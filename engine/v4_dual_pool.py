"""Phase 0c: dual-pool (SWA + compressed) page export for DeepSeek-V4 Hybrid.

Hybrid still runs official ``sparse_attn`` with in-module buffers. This module
**dual-writes** packed pages into :class:`RadixCache` so prefix lifecycle can
migrate off whole-buffer CPU snapshots.

Restore path remains official-buffer snapshots until a later slice makes
Radix the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    # Same physical page ids index both packed pools.
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
) -> int:
    """Write one layer into SWA and/or compressed packed pools.

    Accepts either bf16 ``[..., 512]`` (packed here) or pre-packed uint8
    ``[T, 584]``. Returns number of tokens written (max of SWA/comp lengths).
    """
    written = 0
    if swa_bf16 is not None or swa_packed is not None:
        packed = swa_packed
        if packed is None:
            assert swa_bf16 is not None
            flat = _as_token_major_512(swa_bf16)
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
            packed = pack_dsv4_kv_bf16(flat, act_quant_fn=act_quant_fn)
        radix.write_packed_kv(
            handle.comp_blocks, start_pos, packed, pool="comp", layer_idx=layer_idx
        )
        written = max(written, int(packed.shape[0]))
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
) -> List[Tuple[int, torch.Tensor]]:
    """Best-effort extract per-layer bf16 KV rows shaped ``[T, 512]``.

    Official V4 buffers vary by module; we accept tensors whose last dim is
    512 (or 512-contiguous) on the batch row. Returns ``(layer_idx, kv)`` pairs
    in discovery order (layer_idx is a dense counter, not model layer id).
    """
    out: List[Tuple[int, torch.Tensor]] = []
    seen = set()
    layer_i = 0
    for name, mod in model.named_modules():
        for attr in ("kv_cache",):
            if not hasattr(mod, attr):
                continue
            key = f"{name}.{attr}"
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
            out.append((layer_i, flat))
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
            for layer_idx, kv in layers:
                if layer_idx >= radix.num_layers:
                    break
                # Official hybrid often keeps a single fused SWA||compressed stream
                # in one buffer; dual-write the same stream into both pools so
                # page lifecycle is exercised. Later slices split SWA vs compress.
                write_dual_pool_layer(
                    radix,
                    handle,
                    layer_idx=layer_idx,
                    swa_bf16=kv,
                    comp_bf16=kv,
                    start_pos=0,
                    act_quant_fn=act_quant_fn,
                )
                wrote_any = True
        else:
            # No extractable 512-d rows: still reserve pages (lifecycle bookkeeping)
            # so finish/cancel can release dual-pool blocks. Filled with zeros.
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
        )
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
    """Append one token at ``pos`` into existing dual-pool pages (best-effort).

    Returns True if at least one layer was written.
    """
    if pos < 0:
        return False
    if radix.packed_comp_cache is None:
        return False
    if radix.packed_swa_cache is None and radix.packed_kv_cache is None:
        return False

    # Extract up to pos+1 tokens; use only the last row.
    layers = extract_layer_kv_bf16(model, batch_slot=batch_slot, max_tokens=pos + 1)
    if not layers:
        return False

    ensure_dual_pool_capacity(radix, handle, pos + 1)
    wrote = False
    for layer_idx, kv in layers:
        if layer_idx >= radix.num_layers:
            break
        if kv.shape[0] <= pos:
            # Buffer shorter than expected — use last available token.
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
        )
        wrote = True
    if wrote:
        handle.n_tokens = max(handle.n_tokens, pos + 1)
        radix.dual_append_count += 1
    return wrote


def verify_dual_pool_roundtrip(
    radix: RadixCache,
    handle: DualPoolHandle,
    *,
    layer_idx: int = 0,
    n_tokens: Optional[int] = None,
) -> bool:
    """Return True if SWA packed pages for ``layer_idx`` are readable and non-empty."""
    n = int(n_tokens if n_tokens is not None else handle.n_tokens)
    if n <= 0 or not handle.swa_blocks:
        return False
    try:
        got = radix.read_packed_kv(handle.swa_blocks, n, pool="swa", layer_idx=layer_idx)
        return got.shape == (n, DSV4_PACKED_BYTES)
    except Exception:
        return False
