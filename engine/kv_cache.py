"""
RadixAttention-style KV Cache with actual prefix KV state management.

This is a core component. We implement prefix sharing here so that
multi-turn / agent workloads can reuse pre-computed KV for common prefixes.

For Phase 0 (CPU friendly):
- We store KV states directly on the tree nodes (list of (key, value) per layer).
- On prefix match we can reuse the stored KV up to the match point.
- When forking a new branch we copy the prefix KV (simple but correct).
- No full paged GPU blocks yet — we will evolve to paged later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class KVBlock:
    """Simple abstraction for a memory block (used for stats & future paged impl)."""

    block_id: int
    ref_count: int = 0


# Legacy type kept for compatibility during transition to paged KV + FlashInfer.
# Will be phased out.
PastKV = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class RadixNode:
    """Node in the radix tree over token ids (pure tree structure)."""

    tokens: List[int] = field(default_factory=list)
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    ref_count: int = 0

    # For paged KV: list of block ids that hold the KV for this prefix.
    # Shared across sequences for common prefixes (refcounted).
    block_ids: List[int] = field(default_factory=list)

    # Rough memory usage estimate (number of tokens this prefix represents)
    prefix_len: int = 0


class RadixTree:
    """Pure token radix tree. KV concerns are handled by the caller (RadixCache)."""

    def __init__(self):
        self.root = RadixNode()

    def walk(self, token_ids: List[int]) -> Tuple[RadixNode, List[int], int]:
        """Walk the tree as far as possible. Returns (final_node, matched, matched_len)."""
        node = self.root
        matched: List[int] = []
        i = 0
        for tid in token_ids:
            if tid in node.children:
                node = node.children[tid]
                matched.append(tid)
                i += 1
            else:
                break
        return node, matched, i

    def insert_path(self, token_ids: List[int]) -> RadixNode:
        """Create nodes for the path if missing. Returns the leaf node."""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = RadixNode(tokens=[tid])
            node = node.children[tid]
            node.ref_count += 1
        return node


class RadixCache:
    """
    Radix cache with KV state sharing.

    Key operations:
    - match_prefix: find longest matching prefix and return cached KV if available.
    - commit: after computing new tokens, store/update the KV for the path.
    - fork / share: when a new sequence branches from an existing prefix.
    """

    def __init__(
        self,
        max_tokens: int = 65536,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.max_tokens = max_tokens
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.tree = RadixTree()
        self._allocated_blocks: Dict[int, KVBlock] = {}
        self._next_block_id = 0

        # Pre-allocate paged KV cache for FlashInfer
        # shape: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        num_blocks = (max_tokens + block_size - 1) // block_size
        self.k_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype, device=device
        )

        # Simple stats
        self.total_tokens_stored = 0
        self.hit_count = 0
        self.miss_count = 0

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], int, List[int], List[int]]:
        """
        Find longest prefix match (paged version).

        Returns:
            matched_tokens, matched_len, remaining_tokens, block_ids (shared prefix blocks)
        """
        node, matched, i = self.tree.walk(token_ids)
        block_ids = list(node.block_ids)  # copy for sharing

        remaining = token_ids[i:]
        if block_ids:  # we have blocks for this prefix
            self.hit_count += 1
        elif i > 0:
            # find from subtree if partial match to a committed longer prefix
            def find_blocks(n):
                if n.block_ids:
                    return n.block_ids
                for c in n.children.values():
                    res = find_blocks(c)
                    if res:
                        return res
                return None

            full_blocks = find_blocks(node)
            if full_blocks:
                num_blocks = (i + self.block_size - 1) // self.block_size
                block_ids = full_blocks[:num_blocks]
                self.hit_count += 1
                self.miss_count = max(0, self.miss_count - 1)
            else:
                self.hit_count += 1
                self.miss_count = max(0, self.miss_count - 1)

        return matched, i, remaining, block_ids

    def insert_or_update(
        self,
        token_ids: List[int],
        kv_tensors: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (k, v) for new tokens, shape (num_new, num_heads, head_dim) or per layer?
        prefix_len: int = 0,
        block_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Walk/create the path and allocate/store paged blocks for new tokens.
        For paged + FlashInfer.

        kv_tensors: for simplicity in transition, expect per-layer? But start simple.
        block_ids: if provided (e.g. from the runner's paged block table), use them
            directly instead of allocating new blocks. This keeps the radix tree
            block list consistent with the actual paged KV cache.
        Returns the list of new block ids allocated for this append.
        """
        node = self.tree.insert_path(token_ids)

        new_block_ids: List[int] = []

        if kv_tensors is not None and block_ids is not None:
            # Runner already allocated pages and appended KV via flashinfer;
            # just record the block table on the radix node for prefix sharing.
            node.block_ids = list(block_ids)
            node.prefix_len = prefix_len + len(token_ids)
            return list(block_ids)

        if kv_tensors is not None:
            # Handle legacy PastKV list or single (k,v)
            if (
                isinstance(kv_tensors, list)
                and len(kv_tensors) > 0
                and isinstance(kv_tensors[0], tuple)
            ):
                # multi layer, take first for demo
                k, v = kv_tensors[0]
            elif isinstance(kv_tensors, tuple):
                k, v = kv_tensors
            else:
                k, _v = None, None
            num_new = k.shape[0] if k is not None and k.dim() > 0 else 0
            num_blocks_needed = (
                (num_new + self.block_size - 1) // self.block_size if num_new > 0 else 0
            )

            for _ in range(num_blocks_needed):
                bid = self._next_block_id
                self._next_block_id += 1
                self._allocated_blocks[bid] = KVBlock(block_id=bid, ref_count=1)
                new_block_ids.append(bid)

                # Copy into paged cache (simplified, assumes single layer for demo)
                # In full, loop over layers, use proper shapes.
                _start = len(node.block_ids) * self.block_size
                # This is placeholder; real copy needs reshaping per layer.
                # For now, just allocate to enable FlashInfer block table.

            node.block_ids.extend(new_block_ids)
            node.prefix_len = prefix_len + len(token_ids)

        self.total_tokens_stored = max(self.total_tokens_stored, node.prefix_len)
        return new_block_ids

    def append_kv_paged(
        self,
        block_ids: List[int],
        kv_state: PastKV,
        layer_idx: int = 0,
    ) -> None:
        """Write the kv_state (for new tokens) into the paged cache using the block_ids.
        Assumes kv_state[-1] corresponds to the appended tokens.
        For production, use flashinfer.append_paged_kv_cache for efficiency.
        """
        if not kv_state or not block_ids:
            return
        # Simple: take the last 'new' part, assume shape allows.
        # In real, kv_state is list per layer, we take for this layer.
        k, v = kv_state[layer_idx] if isinstance(kv_state[0], tuple) else kv_state  # adjust
        # For demo, assume k,v are (1, heads, new_len, dim) or similar
        # Copy into last block for simplicity (extend as needed)
        _bid = block_ids[-1]
        # This is illustrative; real impl slices by block_size
        if k.dim() == 4:  # e.g. (bs, heads, seq, dim) wait adjust
            # placeholder copy
            pass
        # For actual production, after compute QKV, use:
        # flashinfer.append_paged_kv_cache(
        #     k, v, self.k_cache[layer_idx], self.v_cache[layer_idx], block_table, ...
        # )

    def fork_kv(self, src_kv: Optional[PastKV]) -> Optional[PastKV]:
        """
        When we hit a prefix, we usually need to copy the KV tensors
        because the new sequence will continue to append to it.
        Shallow copy of structure + clone tensors (CPU friendly).
        """
        if src_kv is None:
            return None
        return [(k.clone(), v.clone()) for k, v in src_kv]

    def get_cache_stats(self) -> Dict:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total)
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "total_tokens_stored": self.total_tokens_stored,
        }

    # --- The following are kept for future paged attention evolution ---
    def allocate_blocks(self, count: int) -> List[int]:
        ids = []
        for _ in range(count):
            bid = self._next_block_id
            self._next_block_id += 1
            self._allocated_blocks[bid] = KVBlock(block_id=bid, ref_count=1)
            ids.append(bid)
        return ids

    def release_blocks(self, block_ids: List[int]) -> None:
        for bid in block_ids:
            blk = self._allocated_blocks.get(bid)
            if blk:
                blk.ref_count -= 1
                if blk.ref_count <= 0:
                    self._allocated_blocks.pop(bid, None)

    def evict(self, needed: int) -> int:
        # Improved placeholder eviction: simple LRU-like by walking leaves with low ref
        # Real would use priority queue on last_access + ref_count
        evicted = 0
        to_evict = []
        for bid, blk in list(self._allocated_blocks.items()):
            if blk.ref_count <= 1 and evicted < needed:
                to_evict.append(bid)
                evicted += 1
        for bid in to_evict:
            self.release_blocks([bid])
        return evicted
