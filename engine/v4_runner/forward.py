"""V4-Flash decode forward helpers (official / vendored Transformer).

Hot path: single-step decode with static token buffer when possible.
Does not import sglang or vllm.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import torch


def extract_logits(out: Any) -> torch.Tensor:
    """Normalize official / Hybrid forward outputs to ``[B, vocab]`` logits."""
    if isinstance(out, torch.Tensor):
        logits = out
    elif isinstance(out, (tuple, list)):
        # Some wrappers return (tokens, logits, hidden); official returns logits.
        logits = out[1] if len(out) > 1 and isinstance(out[1], torch.Tensor) else out[0]
    else:
        logits = out.logits
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    return logits


def model_forward_logits(
    model: Any,
    input_ids: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    """``Transformer.forward(input_ids, start_pos)`` → last-position logits."""
    if hasattr(model, "temperature"):
        try:
            model.temperature = 0.0
        except Exception:
            pass
    out = model(input_ids, start_pos=int(start_pos))
    return extract_logits(out)


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Greedy or Gumbel-max sample (matches official generate.py).

    Returns ``[B]`` int64 token ids on the same device as ``logits``.
    """
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    scaled = logits / max(float(temperature), 1e-5)
    probs = torch.softmax(scaled, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def decode_step(
    model: Any,
    token_ids: Union[Sequence[int], torch.Tensor],
    start_pos: int,
    *,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[int]:
    """One decode step for batch=1 or small batch.

    ``token_ids`` shape: list of int (B=1) or list of lists / ``[B, 1]`` tensor.
    """
    if isinstance(token_ids, torch.Tensor):
        ids = token_ids
        if ids.dim() == 1:
            ids = ids.view(1, -1)
    else:
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            ids = torch.tensor([list(token_ids)], dtype=torch.long)
    if device is not None:
        ids = ids.to(device)
    elif next(model.parameters()).is_cuda:
        ids = ids.cuda()
    logits = model_forward_logits(model, ids, start_pos)
    nxt = sample_token(logits, temperature=temperature)
    return [int(x) for x in nxt.detach().cpu().tolist()]
