"""MoEModelRunner: load MoE models, prefill/decode with prefix-aware KV."""

from __future__ import annotations

from typing import List, Optional

import torch

from .kv_cache import PastKV, RadixCache
from .models import assert_moe_supported, is_fixture_model, register_verified
from .sampling import make_generator, sample_logits
from .scheduler import Sequence

try:
    import flashinfer

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    flashinfer = None


class MoERouter:
    """Placeholder router; real MoE routing is inside the HF model."""

    def route(self, input_ids: List[int]) -> List[int]:
        return [0] * len(input_ids)


class ModelRunner:
    def __init__(
        self,
        model_name: str = "stub",
        device: str = "cpu",
        max_batch: int = 4,
        allow_stub: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.max_batch = max_batch
        self.allow_stub = allow_stub

        self.model = None
        self.tokenizer = None
        self._is_real = False
        self.vocab_size = 32000
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 64
        self.eos_token_id: Optional[int] = 2

        if model_name == "stub":
            if not allow_stub:
                raise ValueError(
                    "model='stub' is only allowed with allow_stub=True "
                    "(explicit demo/test mode). Production must load a real MoE model."
                )
            self._init_tiny_stub_model()
        else:
            self._load_real(model_name)

    def _load_real(self, model_name: str) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        assert_moe_supported(model_name)

        if is_fixture_model(model_name):
            path = model_name.split(":", 1)[1]
            load_id = path
        else:
            load_id = model_name

        print(f"[sglang-lite] Loading MoE model: {load_id}")
        config = AutoConfig.from_pretrained(load_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", None)
        assert_moe_supported(model_name, model_type)

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_id, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            load_id,
            dtype=dtype,
            device_map="cpu" if self.device == "cpu" else "auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self._is_real = True
        self.vocab_size = int(self.model.config.vocab_size)
        self.num_layers = int(self.model.config.num_hidden_layers)
        self.num_kv_heads = int(
            getattr(
                self.model.config,
                "num_key_value_heads",
                self.model.config.num_attention_heads,
            )
        )
        self.head_dim = int(
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None) or getattr(
            self.model.config, "eos_token_id", 2
        )

        if HAS_FLASHINFER and self.device != "cpu":
            self.workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD"
            )
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD"
            )

        register_verified(model_name)
        print(f"[sglang-lite] MoE model '{model_name}' ready on {self.device}")

    def _init_tiny_stub_model(self):
        import torch.nn as nn

        class TinyLM(nn.Module):
            def __init__(self, vocab=32000, dim=256, layers=4, heads=4):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
                        for _ in range(layers)
                    ]
                )
                self.ln = nn.LayerNorm(dim)
                self.head = nn.Linear(dim, vocab, bias=False)
                self.vocab_size = vocab
                self.config = type(
                    "C",
                    (),
                    {
                        "vocab_size": vocab,
                        "num_hidden_layers": layers,
                        "num_attention_heads": heads,
                        "num_key_value_heads": heads,
                        "hidden_size": dim,
                        "eos_token_id": 2,
                    },
                )()

            def forward(self, input_ids, past_key_values=None, use_cache=False):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                x = self.ln(x)
                logits = self.head(x)
                dummy_past = None
                if use_cache:
                    dummy_past = [
                        (x[:, -1:, :].clone(), x[:, -1:, :].clone())
                        for _ in range(len(self.layers))
                    ]
                return type(
                    "Obj",
                    (),
                    {"logits": logits, "past_key_values": dummy_past},
                )()

        self.model = TinyLM(vocab=self.vocab_size).to(self.device)
        self.model.eval()
        self._is_real = False
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 64
        self.eos_token_id = 2
        print("[sglang-lite] Using explicit tiny stub model (allow_stub=True).")

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [hash(c) % self.vocab_size for c in text[:128]]

    def apply_chat_template(self, messages: list) -> List[int]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                return list(ids)
            except Exception:
                pass
        # Fallback: concatenate role:content
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content") or ""
            else:
                role = getattr(m, "role", "user")
                content = getattr(m, "content", "") or ""
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return self.tokenize("\n".join(parts))

    def detokenize(self, token_ids: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return "".join([chr(97 + (t % 26)) for t in token_ids[:20]])

    def detokenize_delta(self, token_ids: List[int], prev_text: str = "") -> str:
        """Unicode-safe incremental decode: return only the new suffix."""
        if not token_ids:
            return ""
        full = self.detokenize(token_ids)
        if full.startswith(prev_text):
            return full[len(prev_text) :]
        return full

    @torch.no_grad()
    def run_batch(
        self,
        batch: List[Sequence],
        radix: RadixCache,
        is_prefill: List[bool],
    ) -> List[Optional[int]]:
        if not batch:
            return []
        results: List[Optional[int]] = []
        for i, seq in enumerate(batch):
            if seq.finished:
                results.append(None)
                continue
            if is_prefill[i]:
                results.append(self._prefill(seq, radix))
            else:
                results.append(self._decode_one(seq, radix))
        return results

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        temperature = float(getattr(seq, "temperature", 0.0) or 0.0)
        top_p = float(getattr(seq, "top_p", 1.0) or 1.0)
        top_k = getattr(seq, "top_k", None)
        seed = getattr(seq, "seed", None)
        # Per-step generator: seed + decode step for reproducibility
        step = len(seq.output_ids)
        gen = None
        if seed is not None:
            gen = make_generator(self.device, int(seed) + step)
        return sample_logits(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=gen,
        )

    def _ensure_blocks(self, seq: Sequence, radix: RadixCache, total_len: int) -> None:
        needed = (total_len + radix.block_size - 1) // radix.block_size
        while len(seq.block_table) < needed:
            seq.block_table.extend(radix.allocate_blocks(1))

    def _prefill(self, seq: Sequence, radix: RadixCache) -> int:
        prompt = seq.input_ids
        start = seq.cached_len
        new_tokens = prompt[start:]
        # Tokens actually skipped by prefix cache (set once at admission)
        if not hasattr(seq, "cache_hit_tokens"):
            seq.cache_hit_tokens = start

        if not new_tokens:
            return self._decode_one(seq, radix)

        self._ensure_blocks(seq, radix, len(prompt))

        # True prefix skip: only forward uncached tokens with past KV.
        # Keep HF Cache objects as-is (transformers >=4.36 rejects legacy lists).
        past = self._as_model_cache(seq.kv_state)
        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        attn = self._attention_mask(input_ids, past)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past,
            use_cache=True,
            attention_mask=attn,
        )
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values

        legacy = self._to_legacy_kv(new_kv)
        if legacy is not None:
            write_kv = self._slice_kv_tail(legacy, len(new_tokens))
            try:
                radix.write_kv(seq.block_table, start, write_kv)
            except Exception:
                pass

        next_id = self._sample(logits, seq)
        seq.kv_state = new_kv
        seq.cached_len = len(prompt)
        seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
        # Radix commit happens in Scheduler.update_after_prefill
        return next_id

    def _decode_one(self, seq: Sequence, radix: RadixCache) -> int:
        last_token = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)

        pos = seq.cached_len
        # COW last page if shared before append
        if seq.block_table:
            page_i = pos // radix.block_size
            while len(seq.block_table) <= page_i:
                seq.block_table.extend(radix.allocate_blocks(1))
            if page_i < len(seq.block_table):
                seq.block_table[page_i] = radix.cow_block_if_shared(seq.block_table[page_i])

        self._ensure_blocks(seq, radix, pos + 1)

        past = self._as_model_cache(seq.kv_state)
        attn = self._attention_mask(input_ids, past)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past,
            use_cache=True,
            attention_mask=attn,
        )
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values

        legacy = self._to_legacy_kv(new_kv)
        if legacy is not None:
            tail = self._slice_kv_tail(legacy, 1)
            try:
                radix.write_kv(seq.block_table, pos, tail)
            except Exception:
                pass

        next_id = self._sample(logits, seq)
        seq.kv_state = new_kv
        # KV now includes the forwarded token at `pos`
        seq.cached_len = pos + 1
        return next_id

    def _slice_kv_tail(self, kv: PastKV, n: int) -> PastKV:
        out = []
        for k, v in kv:
            if k.dim() == 4:
                out.append((k[:, :, -n:, :].contiguous(), v[:, :, -n:, :].contiguous()))
            elif k.dim() == 3:
                out.append((k[-n:, :, :].contiguous(), v[-n:, :, :].contiguous()))
            else:
                out.append((k, v))
        return out

    def _attention_mask(self, input_ids: torch.Tensor, past) -> torch.Tensor:
        past_len = 0
        if past is not None and hasattr(past, "get_seq_length"):
            past_len = int(past.get_seq_length())
        elif isinstance(past, (list, tuple)) and past:
            k0 = past[0][0]
            past_len = int(k0.shape[-2]) if k0.dim() >= 3 else 0
        total = past_len + input_ids.shape[-1]
        return torch.ones((input_ids.shape[0], total), dtype=torch.long, device=input_ids.device)

    def _as_model_cache(self, past):
        """Return a cache object acceptable by current transformers."""
        if past is None:
            return None
        if hasattr(past, "get_seq_length"):
            return past
        if isinstance(past, (list, tuple)):
            try:
                from transformers import DynamicCache

                return DynamicCache.from_legacy_cache(past)
            except Exception:
                return past
        return past

    def _to_legacy_kv(self, past) -> Optional[PastKV]:
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            return list(past.to_legacy_cache())
        if isinstance(past, (list, tuple)):
            return list(past)
        return None

    def sample_next(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> int:
        gen = make_generator(self.device, seed)
        return sample_logits(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
        )


# Alias preferred by architecture docs
MoEModelRunner = ModelRunner
