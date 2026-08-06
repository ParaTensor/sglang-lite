"""Prefix cache for DeepSeek-V4 Hybrid (official Attention in-module KV).

Stores CPU snapshots of Attention / Compressor / Indexer buffers keyed by
prompt token prefixes. On hit, restore into the sequence's batch slot and
skip (exact) or shorten (suffix) prefill.

Phase 0c: entries may also hold dual-pool (SWA + compressed) page ids in
:class:`~sglang_lite.kv_cache.RadixCache`. The cache **owns** a refcount on
those pages; sequences fork for their lifetime. Restore still uses CPU
``buffers`` until a later slice makes pages the attention source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .kv_cache import RadixCache
    from .v4_dual_pool import DualPoolHandle


@dataclass
class V4PrefixEntry:
    token_ids: List[int]
    last_logits: Optional[torch.Tensor]  # CPU float
    # name -> tensor snapshot (CPU)
    buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    # Phase 0c dual-pool page tables (owned by the cache via refcount).
    swa_block_ids: List[int] = field(default_factory=list)
    comp_block_ids: List[int] = field(default_factory=list)
    dual_pool_tokens: int = 0
    dual_pool_layers: int = 0


def snapshot_v4_kv(model: torch.nn.Module, *, batch_slot: int = 0) -> Dict[str, torch.Tensor]:
    """Copy V4 stateful buffers for one batch slot to CPU."""
    out: Dict[str, torch.Tensor] = {}
    seen = set()
    for name, mod in model.named_modules():
        for attr in ("kv_cache", "kv_state", "score_state"):
            if not hasattr(mod, attr):
                continue
            key = f"{name}.{attr}"
            if key in seen:
                continue
            buf = getattr(mod, attr)
            if not torch.is_tensor(buf) or buf.numel() == 0:
                continue
            seen.add(key)
            # [max_batch, ...] — keep only the used slot to shrink CPU RAM.
            # clone() is required: detach().to("cpu") may still share storage.
            if buf.dim() >= 1 and buf.shape[0] > batch_slot:
                piece = buf[batch_slot : batch_slot + 1].detach().to("cpu").contiguous().clone()
            else:
                piece = buf.detach().to("cpu").contiguous().clone()
            out[key] = piece
    return out


def clear_v4_kv_slot(model: torch.nn.Module, *, batch_slot: int = 0) -> int:
    """Zero one batch row of official V4 KV / compressor state.

    Call on cache miss (before prefill) and when a sequence finishes so the
    next occupant of that slot cannot read stale compressor state.
    """
    n = 0
    seen = set()
    for name, mod in model.named_modules():
        for attr in ("kv_cache", "kv_state", "score_state"):
            if not hasattr(mod, attr):
                continue
            key = f"{name}.{attr}"
            if key in seen:
                continue
            buf = getattr(mod, attr)
            if not torch.is_tensor(buf) or buf.numel() == 0:
                continue
            seen.add(key)
            if buf.dim() >= 1 and buf.shape[0] > batch_slot:
                buf[batch_slot].zero_()
                n += 1
            elif batch_slot == 0:
                buf.zero_()
                n += 1
    return n


def restore_v4_kv(
    model: torch.nn.Module,
    buffers: Dict[str, torch.Tensor],
    *,
    batch_slot: int = 0,
) -> int:
    """Restore CPU snapshots into ``batch_slot``. Returns number of tensors written."""
    n = 0
    for key, cpu_t in buffers.items():
        # key = "layers.3.attn.kv_cache"
        if "." not in key:
            continue
        mod_path, attr = key.rsplit(".", 1)
        mod = model
        try:
            for part in mod_path.split("."):
                if not part:
                    continue
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if not hasattr(mod, attr):
            continue
        buf = getattr(mod, attr)
        if not torch.is_tensor(buf):
            continue
        src = cpu_t.to(device=buf.device, dtype=buf.dtype)
        if buf.dim() >= 1 and buf.shape[0] > batch_slot and src.shape[0] == 1:
            buf[batch_slot : batch_slot + 1].copy_(src)
        elif buf.shape == src.shape:
            buf.copy_(src)
        else:
            # Shape drift — skip rather than corrupt.
            continue
        n += 1
    return n


class V4PrefixCache:
    """Longest-prefix store for Hybrid V4 prompts (exact + partial).

    When ``radix`` is bound, dual-pool page ids on entries are refcounted:
    insert forks pages for the cache; eviction/replace releases them.
    """

    def __init__(self, max_entries: int = 64, radix: Optional["RadixCache"] = None):
        self.max_entries = max_entries
        self.radix = radix
        self._entries: List[V4PrefixEntry] = []
        self.dual_store_count = 0
        self.dual_hit_count = 0
        self.dual_release_count = 0

    def bind_radix(self, radix: Optional["RadixCache"]) -> None:
        """Attach (or replace) the page pool used for dual-pool refcounts."""
        self.radix = radix

    def clear(self) -> None:
        for e in self._entries:
            self._release_entry_pages(e)
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def insert(
        self,
        token_ids: List[int],
        *,
        last_logits: Optional[torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        swa_block_ids: Optional[List[int]] = None,
        comp_block_ids: Optional[List[int]] = None,
        dual_pool_tokens: int = 0,
        dual_pool_layers: int = 0,
    ) -> None:
        if not token_ids or not buffers:
            return

        swa = list(swa_block_ids or [])
        comp = list(comp_block_ids or [])
        # Cache takes ownership via an extra ref so sequence finish can release
        # its own fork without dropping the stored prefix pages.
        if self.radix is not None and swa:
            swa = self.radix.fork_blocks(swa)
            # Prefer explicit comp ids; if shared with swa, fork once more only
            # when lists differ by identity/content.
            if comp and comp != list(swa_block_ids or []):
                comp = self.radix.fork_blocks(comp)
            else:
                # Same physical pages as SWA — share the forked list (one release).
                comp = list(swa)
            self.dual_store_count += 1

        entry = V4PrefixEntry(
            token_ids=list(token_ids),
            last_logits=last_logits.detach().float().cpu().clone()
            if last_logits is not None
            else None,
            buffers=buffers,
            swa_block_ids=swa,
            comp_block_ids=comp,
            dual_pool_tokens=int(dual_pool_tokens or 0),
            dual_pool_layers=int(dual_pool_layers or 0),
        )
        # Replace equal-length exact key if present.
        for i, e in enumerate(self._entries):
            if e.token_ids == token_ids:
                self._release_entry_pages(e)
                self._entries[i] = entry
                return
        self._entries.append(entry)
        while len(self._entries) > self.max_entries:
            old = self._entries.pop(0)
            self._release_entry_pages(old)

    def match(self, token_ids: List[int]) -> Tuple[int, Optional[V4PrefixEntry]]:
        """Longest entry whose tokens are an exact prefix of ``token_ids``.

        Snapshots are taken after fully processing ``entry.token_ids``, so a
        partial overlap with a longer stored prompt is not reusable.
        """
        best_len = 0
        best: Optional[V4PrefixEntry] = None
        for e in self._entries:
            elen = len(e.token_ids)
            if elen == 0 or elen > len(token_ids):
                continue
            if token_ids[:elen] == e.token_ids and elen > best_len:
                best_len = elen
                best = e
        return best_len, best

    def fork_dual_pool_for_hit(self, entry: V4PrefixEntry) -> Optional["DualPoolHandle"]:
        """Fork dual-pool pages for a hitting sequence (caller owns the fork)."""
        if self.radix is None or not entry.swa_block_ids:
            return None
        from .v4_dual_pool import DualPoolHandle

        swa = self.radix.fork_blocks(entry.swa_block_ids)
        if entry.comp_block_ids and entry.comp_block_ids != entry.swa_block_ids:
            comp = self.radix.fork_blocks(entry.comp_block_ids)
        else:
            # Same ids as SWA: one fork list is enough (release once).
            comp = list(swa)
        self.dual_hit_count += 1
        if hasattr(self.radix, "dual_hit_count"):
            self.radix.dual_hit_count = getattr(self.radix, "dual_hit_count", 0) + 1
        return DualPoolHandle(
            swa_blocks=swa,
            comp_blocks=comp,
            n_tokens=int(entry.dual_pool_tokens or len(entry.token_ids)),
            n_layers_written=int(entry.dual_pool_layers or 0),
        )

    def _release_entry_pages(self, entry: V4PrefixEntry) -> None:
        if self.radix is None:
            entry.swa_block_ids = []
            entry.comp_block_ids = []
            return
        # Shared swa/comp ids → single release.
        ids = list(entry.swa_block_ids)
        if entry.comp_block_ids and entry.comp_block_ids != entry.swa_block_ids:
            # Distinct tables (unusual): release both.
            if ids:
                self.radix.release_blocks(ids)
            if entry.comp_block_ids:
                self.radix.release_blocks(entry.comp_block_ids)
                self.dual_release_count += 1
        elif ids:
            self.radix.release_blocks(ids)
            self.dual_release_count += 1
        entry.swa_block_ids = []
        entry.comp_block_ids = []

    def get_stats(self) -> Dict[str, int]:
        n_dual = sum(1 for e in self._entries if e.swa_block_ids)
        return {
            "prefix_entries": len(self._entries),
            "prefix_dual_entries": n_dual,
            "dual_store_count": self.dual_store_count,
            "dual_hit_count": self.dual_hit_count,
            "dual_release_count": self.dual_release_count,
        }
