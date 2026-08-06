"""Phase 0c: dual-pool (SWA + compressed) pages for DeepSeek-V4 Hybrid.

Hybrid still runs official ``sparse_attn`` with in-module buffers. This module:

1. **Dual-writes** packed uint8 pages (future FI path) + bf16 restore pages.
2. **Restores** official ``kv_cache`` rows from bf16 pages on prefix hit (0c-3).
3. **Stages** official buffers from pages before decode when page-primary (0c-4):
   pages are the source of truth; official ``kv_cache`` is a staging area for
   TileLang ``sparse_attn`` (FI leaf remains opt-in, not default).
4. Keeps CPU snapshots for ``kv_state`` / ``score_state`` and as fallback.

Ownership: see :class:`~sglang_lite.v4_prefix_cache.V4PrefixCache`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from .dsv4_kv_pack import DSV4_HEAD_DIM, DSV4_PACKED_BYTES, pack_dsv4_kv_bf16
from .kv_cache import RadixCache


@dataclass
class DualPoolHandle:
    """Page tables for one sequence's dual-pool residency."""

    swa_blocks: List[int] = field(default_factory=list)
    comp_blocks: List[int] = field(default_factory=list)
    n_tokens: int = 0
    n_layers_written: int = 0
    # module path keys aligned with layer indices written to restore_bf16_cache
    # e.g. ["layers.0.attn.kv_cache", ...]
    layer_keys: List[str] = field(default_factory=list)


def pages_for_tokens(n_tokens: int, block_size: int) -> int:
    if n_tokens <= 0:
        return 0
    return (n_tokens + block_size - 1) // block_size


def allocate_dual_pool_pages(radix: RadixCache, n_tokens: int) -> DualPoolHandle:
    """Allocate enough pages for ``n_tokens`` (shared block ids across SWA/comp)."""
    n_pages = pages_for_tokens(n_tokens, radix.block_size)
    if n_pages <= 0:
        return DualPoolHandle(n_tokens=0)
    blocks = radix.allocate_blocks(n_pages)
    # Same physical page ids index both packed pools + restore bf16 pool.
    return DualPoolHandle(
        swa_blocks=list(blocks),
        comp_blocks=list(blocks),
        n_tokens=int(n_tokens),
    )


def release_dual_pool_pages(radix: RadixCache, handle: Optional[DualPoolHandle]) -> None:
    if handle is None:
        return
    # swa_blocks and comp_blocks share ids; release once.
    ids = list(handle.swa_blocks) or list(handle.comp_blocks)
    if ids:
        radix.release_blocks(ids)
    handle.swa_blocks = []
    handle.comp_blocks = []
    handle.n_tokens = 0
    handle.n_layers_written = 0
    handle.layer_keys = []


def write_dual_pool_layer(
    radix: RadixCache,
    handle: DualPoolHandle,
    *,
    layer_idx: int,
    swa_bf16: Optional[torch.Tensor] = None,
    comp_bf16: Optional[torch.Tensor] = None,
    swa_packed: Optional[torch.Tensor] = None,
    comp_packed: Optional[torch.Tensor] = None,
    start_pos: int = 0,
    act_quant_fn=None,
    write_restore_bf16: bool = True,
) -> int:
    """Write one layer into SWA/comp packed pools (+ optional bf16 restore pool).

    Accepts either bf16 ``[..., 512]`` (packed here) or pre-packed uint8
    ``[T, 584]``. Returns number of tokens written (max of SWA/comp lengths).
    """
    written = 0
    restore_src: Optional[torch.Tensor] = None
    if swa_bf16 is not None or swa_packed is not None:
        packed = swa_packed
        if packed is None:
            assert swa_bf16 is not None
            flat = _as_token_major_512(swa_bf16)
            restore_src = flat
            packed = pack_dsv4_kv_bf16(flat, act_quant_fn=act_quant_fn)
        radix.write_packed_kv(
            handle.swa_blocks, start_pos, packed, pool="swa", layer_idx=layer_idx
        )
        written = max(written, int(packed.shape[0]))
    if comp_bf16 is not None or comp_packed is not None:
        packed = comp_packed
        if packed is None:
            assert comp_bf16 is not None
            flat = _as_token_major_512(comp_bf16)
            if restore_src is None:
                restore_src = flat
            packed = pack_dsv4_kv_bf16(flat, act_quant_fn=act_quant_fn)
        radix.write_packed_kv(
            handle.comp_blocks, start_pos, packed, pool="comp", layer_idx=layer_idx
        )
        written = max(written, int(packed.shape[0]))
    if (
        write_restore_bf16
        and restore_src is not None
        and radix.restore_bf16_cache is not None
    ):
        radix.write_restore_bf16(
            handle.swa_blocks, start_pos, restore_src, layer_idx=layer_idx
        )
    if written:
        handle.n_layers_written = max(handle.n_layers_written, layer_idx + 1)
        handle.n_tokens = max(handle.n_tokens, start_pos + written)
    return written


def _as_token_major_512(x: torch.Tensor) -> torch.Tensor:
    """Normalize to ``[T, 512]`` bf16."""
    if x.dim() == 1 and x.numel() == DSV4_HEAD_DIM:
        x = x.unsqueeze(0)
    if x.dim() == 3 and x.shape[-1] == DSV4_HEAD_DIM:
        # [B, T, 512] or [T, H, 512] with H=1
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = x.reshape(-1, DSV4_HEAD_DIM)
    if x.dim() != 2 or x.shape[-1] != DSV4_HEAD_DIM:
        raise ValueError(f"expected [T, {DSV4_HEAD_DIM}], got {tuple(x.shape)}")
    return x.to(dtype=torch.bfloat16)


def extract_layer_kv_bf16(
    model: torch.nn.Module,
    *,
    batch_slot: int = 0,
    max_tokens: Optional[int] = None,
) -> List[Tuple[int, str, torch.Tensor]]:
    """Extract per-layer bf16 KV rows shaped ``[T, 512]`` with module keys.

    Returns ``(layer_idx, module_key, kv)`` where ``module_key`` is like
    ``layers.0.attn.kv_cache`` for later restore.
    """
    out: List[Tuple[int, str, torch.Tensor]] = []
    seen = set()
    layer_i = 0
    for name, mod in model.named_modules():
        for attr in ("kv_cache",):
            if not hasattr(mod, attr):
                continue
            key = f"{name}.{attr}" if name else attr
            if key in seen:
                continue
            buf = getattr(mod, attr)
            if not torch.is_tensor(buf) or buf.numel() == 0:
                continue
            seen.add(key)
            row = _slice_batch_row(buf, batch_slot)
            if row is None:
                continue
            flat = _try_as_t512(row, max_tokens=max_tokens)
            if flat is None:
                continue
            out.append((layer_i, key, flat))
            layer_i += 1
    return out


def _slice_batch_row(buf: torch.Tensor, batch_slot: int) -> Optional[torch.Tensor]:
    if buf.dim() >= 1 and buf.shape[0] > batch_slot:
        return buf[batch_slot]
    if batch_slot == 0:
        return buf
    return None


def _try_as_t512(
    row: torch.Tensor, *, max_tokens: Optional[int]
) -> Optional[torch.Tensor]:
    """Interpret a module buffer row as ``[T, 512]`` when possible."""
    if row.dim() == 2 and row.shape[-1] == DSV4_HEAD_DIM:
        t = row
    elif row.dim() == 3 and row.shape[-1] == DSV4_HEAD_DIM:
        # [T, H, 512] or [H, T, 512]
        if row.shape[1] == 1:
            t = row[:, 0, :]
        elif row.shape[0] == 1:
            t = row[0]
        else:
            t = row.reshape(-1, DSV4_HEAD_DIM)
    elif row.dim() == 2 and row.shape[0] == DSV4_HEAD_DIM:
        t = row.transpose(0, 1).contiguous()
    else:
        return None
    if max_tokens is not None and t.shape[0] > max_tokens:
        t = t[:max_tokens]
    if t.shape[0] == 0:
        return None
    return t.detach()


def dual_write_from_model(
    model: torch.nn.Module,
    radix: RadixCache,
    *,
    batch_slot: int = 0,
    n_tokens: int,
    act_quant_fn=None,
) -> Optional[DualPoolHandle]:
    """Allocate dual-pool pages and dual-write extractable layer KVs.

    Also writes bf16 restore pages when ``restore_bf16_cache`` is allocated.
    Returns a handle when at least one layer was written; otherwise releases
    pages and returns ``None``.
    """
    if n_tokens <= 0:
        return None
    if radix.packed_swa_cache is None and radix.packed_kv_cache is None:
        return None
    if radix.packed_comp_cache is None:
        return None

    layers = extract_layer_kv_bf16(model, batch_slot=batch_slot, max_tokens=n_tokens)
    handle = allocate_dual_pool_pages(radix, n_tokens)
    if not handle.swa_blocks:
        return None

    wrote_any = False
    try:
        if layers:
            for layer_idx, key, kv in layers:
                if layer_idx >= radix.num_layers:
                    break
                # Official hybrid often keeps a single fused stream; dual-write
                # the same stream into both packed pools + bf16 restore.
                write_dual_pool_layer(
                    radix,
                    handle,
                    layer_idx=layer_idx,
                    swa_bf16=kv,
                    comp_bf16=kv,
                    start_pos=0,
                    act_quant_fn=act_quant_fn,
                    write_restore_bf16=True,
                )
                # Keep layer_keys aligned with dense layer indices 0..N-1
                while len(handle.layer_keys) < layer_idx:
                    handle.layer_keys.append("")
                if len(handle.layer_keys) == layer_idx:
                    handle.layer_keys.append(key)
                else:
                    handle.layer_keys[layer_idx] = key
                wrote_any = True
        else:
            # No extractable 512-d rows: reserve zero pages for lifecycle only.
            zeros = torch.zeros(
                n_tokens, DSV4_PACKED_BYTES, dtype=torch.uint8, device=radix.device
            )
            for layer_idx in range(min(1, radix.num_layers)):
                write_dual_pool_layer(
                    radix,
                    handle,
                    layer_idx=layer_idx,
                    swa_packed=zeros,
                    comp_packed=zeros,
                    start_pos=0,
                    write_restore_bf16=False,
                )
                wrote_any = True
    except Exception:
        release_dual_pool_pages(radix, handle)
        raise

    if not wrote_any:
        release_dual_pool_pages(radix, handle)
        return None

    radix.dual_write_count += 1
    radix.dual_write_tokens += int(n_tokens)
    return handle


def dual_write_from_bf16(
    radix: RadixCache,
    *,
    swa_layers: List[torch.Tensor],
    comp_layers: Optional[List[torch.Tensor]] = None,
    layer_keys: Optional[List[str]] = None,
    act_quant_fn=None,
) -> DualPoolHandle:
    """Test/helper: dual-write lists of ``[T, 512]`` layers."""
    if not swa_layers:
        raise ValueError("swa_layers required")
    n_tokens = int(swa_layers[0].shape[0])
    handle = allocate_dual_pool_pages(radix, n_tokens)
    comp_layers = comp_layers or swa_layers
    for i, swa in enumerate(swa_layers):
        comp = comp_layers[i] if i < len(comp_layers) else swa
        write_dual_pool_layer(
            radix,
            handle,
            layer_idx=i,
            swa_bf16=swa,
            comp_bf16=comp,
            start_pos=0,
            act_quant_fn=act_quant_fn,
            write_restore_bf16=True,
        )
        key = layer_keys[i] if layer_keys and i < len(layer_keys) else f"layer.{i}.kv_cache"
        handle.layer_keys.append(key)
    radix.dual_write_count += 1
    radix.dual_write_tokens += n_tokens
    return handle


def ensure_dual_pool_capacity(
    radix: RadixCache, handle: DualPoolHandle, n_tokens: int
) -> None:
    """Grow dual-pool block tables so they cover ``n_tokens`` positions."""
    need = pages_for_tokens(n_tokens, radix.block_size)
    have = len(handle.swa_blocks)
    if need <= have:
        return
    shared = (not handle.comp_blocks) or (handle.comp_blocks == handle.swa_blocks)
    extra = radix.allocate_blocks(need - have)
    handle.swa_blocks.extend(extra)
    if shared:
        handle.comp_blocks = list(handle.swa_blocks)
    elif len(handle.comp_blocks) < need:
        more = radix.allocate_blocks(need - len(handle.comp_blocks))
        handle.comp_blocks.extend(more)


def dual_append_from_model(
    model: torch.nn.Module,
    radix: RadixCache,
    handle: DualPoolHandle,
    *,
    batch_slot: int = 0,
    pos: int,
    act_quant_fn=None,
) -> bool:
    """Append one token at ``pos`` into existing dual-pool pages (best-effort)."""
    if pos < 0:
        return False
    if radix.packed_comp_cache is None:
        return False
    if radix.packed_swa_cache is None and radix.packed_kv_cache is None:
        return False

    layers = extract_layer_kv_bf16(model, batch_slot=batch_slot, max_tokens=pos + 1)
    if not layers:
        return False

    ensure_dual_pool_capacity(radix, handle, pos + 1)
    wrote = False
    for layer_idx, key, kv in layers:
        if layer_idx >= radix.num_layers:
            break
        if kv.shape[0] <= pos:
            row = kv[-1:]
        else:
            row = kv[pos : pos + 1]
        write_dual_pool_layer(
            radix,
            handle,
            layer_idx=layer_idx,
            swa_bf16=row,
            comp_bf16=row,
            start_pos=pos,
            act_quant_fn=act_quant_fn,
            write_restore_bf16=True,
        )
        while len(handle.layer_keys) <= layer_idx:
            handle.layer_keys.append("")
        if not handle.layer_keys[layer_idx]:
            handle.layer_keys[layer_idx] = key
        wrote = True
    if wrote:
        handle.n_tokens = max(handle.n_tokens, pos + 1)
        radix.dual_append_count += 1
    return wrote


def _apply_restore_bf16_to_model(
    model: torch.nn.Module,
    radix: RadixCache,
    handle: DualPoolHandle,
    *,
    batch_slot: int = 0,
    n_tokens: Optional[int] = None,
) -> Tuple[int, Set[str]]:
    """Copy bf16 restore pages → model ``kv_cache`` rows (no counters)."""
    if radix.restore_bf16_cache is None or not handle.swa_blocks:
        return 0, set()
    n = int(n_tokens if n_tokens is not None else handle.n_tokens)
    if n <= 0:
        return 0, set()
    restored: Set[str] = set()
    written = 0
    n_layers = min(
        handle.n_layers_written or len(handle.layer_keys) or radix.num_layers,
        radix.num_layers,
        max(len(handle.layer_keys), 1),
    )
    if handle.layer_keys:
        n_layers = min(len(handle.layer_keys), radix.num_layers)
    for layer_idx in range(n_layers):
        key = (
            handle.layer_keys[layer_idx]
            if layer_idx < len(handle.layer_keys)
            else ""
        )
        if not key:
            continue
        try:
            data = radix.read_restore_bf16(
                handle.swa_blocks, n, layer_idx=layer_idx
            )
        except Exception:
            continue
        if _write_module_kv_row(model, key, data, batch_slot=batch_slot):
            restored.add(key)
            written += 1
    return written, restored


def restore_dual_pool_to_model(
    model: torch.nn.Module,
    radix: RadixCache,
    handle: DualPoolHandle,
    *,
    batch_slot: int = 0,
    n_tokens: Optional[int] = None,
) -> Tuple[int, Set[str]]:
    """Prefix-hit restore: pages → official buffers (increments ``dual_restore_count``)."""
    written, restored = _apply_restore_bf16_to_model(
        model,
        radix,
        handle,
        batch_slot=batch_slot,
        n_tokens=n_tokens,
    )
    if written:
        radix.dual_restore_count += 1
    return written, restored


def stage_official_kv_from_pages(
    model: torch.nn.Module,
    radix: RadixCache,
    handle: DualPoolHandle,
    *,
    batch_slot: int = 0,
    n_tokens: Optional[int] = None,
) -> Tuple[int, Set[str]]:
    """Phase 0c-4: re-stage official ``kv_cache`` from pages before decode.

    When page-primary, dual-pool bf16 pages are the source of truth; the
    official module buffers are only a staging area for TileLang
    ``sparse_attn``. Increments ``dual_stage_count`` (distinct from hit restore).
    """
    written, restored = _apply_restore_bf16_to_model(
        model,
        radix,
        handle,
        batch_slot=batch_slot,
        n_tokens=n_tokens,
    )
    if written:
        radix.dual_stage_count += 1
    return written, restored


def _write_module_kv_row(
    model: torch.nn.Module,
    key: str,
    data: torch.Tensor,
    *,
    batch_slot: int,
) -> bool:
    """Copy ``[T, 512]`` into ``model`` buffer identified by ``key``."""
    if "." not in key and key != "kv_cache":
        return False
    if key == "kv_cache":
        mod, attr = model, "kv_cache"
    else:
        mod_path, attr = key.rsplit(".", 1)
        mod = model
        try:
            for part in mod_path.split("."):
                if not part:
                    continue
                mod = getattr(mod, part)
        except AttributeError:
            return False
    if not hasattr(mod, attr):
        return False
    buf = getattr(mod, attr)
    if not torch.is_tensor(buf):
        return False
    src = data.to(device=buf.device, dtype=buf.dtype)
    # Common layouts: [B, T, 512], [B, T, H, 512], [B, H, T, 512]
    try:
        if buf.dim() >= 1 and buf.shape[0] > batch_slot:
            row = buf[batch_slot]
        elif batch_slot == 0:
            row = buf
        else:
            return False
        t = int(src.shape[0])
        if row.dim() == 2 and row.shape[-1] == DSV4_HEAD_DIM:
            n = min(t, row.shape[0])
            row[:n].copy_(src[:n])
            return True
        if row.dim() == 3 and row.shape[-1] == DSV4_HEAD_DIM:
            if row.shape[1] == 1:
                n = min(t, row.shape[0])
                row[:n, 0, :].copy_(src[:n])
                return True
            if row.shape[0] == 1:
                n = min(t, row.shape[1])
                row[0, :n, :].copy_(src[:n])
                return True
        return False
    except Exception:
        return False


def slim_snapshot_buffers(
    buffers: Dict[str, torch.Tensor],
    restored_or_paged_keys: Set[str],
) -> Dict[str, torch.Tensor]:
    """Drop ``kv_cache`` tensors that are page-backed from a snapshot dict.

    Keeps ``kv_state`` / ``score_state`` and any keys not covered by dual-pool.
    """
    if not restored_or_paged_keys:
        return buffers
    out: Dict[str, torch.Tensor] = {}
    for k, v in buffers.items():
        if k in restored_or_paged_keys:
            continue
        # Also drop keys that end with .kv_cache when layer key matches
        if k.endswith(".kv_cache") and k in restored_or_paged_keys:
            continue
        out[k] = v
    return out


def verify_dual_pool_roundtrip(
    radix: RadixCache,
    handle: DualPoolHandle,
    *,
    layer_idx: int = 0,
    n_tokens: Optional[int] = None,
) -> bool:
    """Return True if SWA packed pages for ``layer_idx`` are readable."""
    n = int(n_tokens if n_tokens is not None else handle.n_tokens)
    if n <= 0 or not handle.swa_blocks:
        return False
    try:
        got = radix.read_packed_kv(handle.swa_blocks, n, pool="swa", layer_idx=layer_idx)
        return got.shape == (n, DSV4_PACKED_BYTES)
    except Exception:
        return False
