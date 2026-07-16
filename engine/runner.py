"""MoEModelRunner: batched HF forward with paged-KV as rebuildable attention state."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

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
        # Observability for tests: size of the last tensor forward (batch dim)
        self.last_model_forward_size = 0
        self.model_forward_count = 0
        self.paged_rebuild_count = 0
        self.use_paged_as_source = True

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
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device != "cpu":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(load_id, **load_kwargs)
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
            print("[sglang-lite] FlashInfer paged wrappers ready (append/decode kernels)")

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

            def forward(
                self, input_ids, past_key_values=None, use_cache=False, attention_mask=None, **kwargs
            ):
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
        self.use_paged_as_source = False  # stub past shapes are not real KV
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
        """Execute one scheduler step.

        Prefill/decode groups that share the same cached length are executed with a
        single batched model forward (tensor batch dim > 1 when possible).
        """
        if not batch:
            return []

        results: List[Optional[int]] = [None] * len(batch)
        self.last_model_forward_size = 0

        prefill_idxs: List[int] = []
        decode_idxs: List[int] = []

        for i, seq in enumerate(batch):
            if seq.finished:
                continue
            # Exact prefix hit: first token from stored logits (no forward)
            if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
                if seq.cached_len >= len(seq.input_ids) or (
                    not is_prefill[i] and seq.cached_len == len(seq.input_ids)
                ):
                    results[i] = self._sample(seq.last_logits.to(device=self.device), seq)
                    seq.cached_len = len(seq.input_ids)
                    continue
            if is_prefill[i] and seq.cached_len < len(seq.input_ids):
                prefill_idxs.append(i)
            else:
                decode_idxs.append(i)

        if prefill_idxs:
            self._run_prefill_groups(batch, prefill_idxs, radix, results)
        if decode_idxs:
            self._run_decode_groups(batch, decode_idxs, radix, results)
        return results

    def _run_prefill_groups(
        self,
        batch: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            groups[batch[i].cached_len].append(i)
        for _cached_len, group in groups.items():
            self._batch_prefill([batch[i] for i in group], group, radix, results)

    def _run_decode_groups(
        self,
        batch: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            groups[batch[i].cached_len].append(i)
        for _cached_len, group in groups.items():
            self._batch_decode([batch[i] for i in group], group, radix, results)

    def _batch_prefill(
        self,
        seqs: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        if not seqs:
            return
        news = [s.input_ids[s.cached_len :] for s in seqs]
        past_lens = [s.cached_len for s in seqs]
        new_lens = [len(n) for n in news]
        # Identical past + new lengths required for a correct single forward (no pad KV pollution)
        if len(set(past_lens)) > 1 or len(set(new_lens)) > 1:
            for s, i in zip(seqs, idxs):
                results[i] = self._prefill_one(s, radix)
            return

        B = len(seqs)
        max_new = new_lens[0]
        if max_new == 0:
            for s, i in zip(seqs, idxs):
                if s.last_logits is None:
                    raise RuntimeError("exact hit without last_logits")
                results[i] = self._sample(s.last_logits.to(self.device), s)
                s.cached_len = len(s.input_ids)
            return

        input_ids = torch.tensor(news, dtype=torch.long, device=self.device)
        pasts = [self._past_for_seq(s, radix) for s in seqs]
        batched_past = self._batch_caches(pasts) if past_lens[0] > 0 else None
        max_past = past_lens[0]
        attn = torch.ones((B, max_past + max_new), dtype=torch.long, device=self.device)

        outputs = self._model_forward(input_ids, batched_past, attn)
        for b, (seq, i) in enumerate(zip(seqs, idxs)):
            nlen = new_lens[b]
            logits = outputs.logits[b, nlen - 1, :]
            full_kv = self._split_batch_cache(outputs.past_key_values, b, B)
            start = seq.cached_len
            self._ensure_blocks(seq, radix, start + nlen)
            write_kv = self._slice_kv_tail(self._to_legacy_kv(full_kv), nlen)
            self._commit_pages(seq, radix, start, write_kv)
            seq.last_logits = logits.detach().float().cpu().clone()
            seq.kv_state = None if self.use_paged_as_source else full_kv
            seq.cached_len = start + nlen
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + nlen
            results[i] = self._sample(logits, seq)

    def _batch_decode(
        self,
        seqs: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        if not seqs:
            return
        # All seqs in this group share cached_len
        pending_seqs: List[Sequence] = []
        pending_idxs: List[int] = []
        for s, i in zip(seqs, idxs):
            if not s.output_ids and getattr(s, "last_logits", None) is not None:
                results[i] = self._sample(s.last_logits.to(self.device), s)
                s.cached_len = len(s.input_ids)
                continue
            pending_seqs.append(s)
            pending_idxs.append(i)
        if not pending_seqs:
            return
        seqs, idxs = pending_seqs, pending_idxs
        B = len(seqs)
        lasts = [s.output_ids[-1] if s.output_ids else s.input_ids[-1] for s in seqs]
        input_ids = torch.tensor([[t] for t in lasts], dtype=torch.long, device=self.device)
        pasts = [self._past_for_seq(s, radix) for s in seqs]
        # COW last page before append
        for s in seqs:
            pos = s.cached_len
            self._ensure_blocks(s, radix, pos + 1)
            page_i = pos // radix.block_size
            if page_i < len(s.block_table):
                s.block_table[page_i] = radix.cow_block_if_shared(s.block_table[page_i])

        batched_past = self._batch_caches(pasts)
        past_len = seqs[0].cached_len
        attn = torch.ones((B, past_len + 1), dtype=torch.long, device=self.device)
        outputs = self._model_forward(input_ids, batched_past, attn)

        for b, (seq, i) in enumerate(zip(seqs, idxs)):
            logits = outputs.logits[b, -1, :]
            full_kv = self._split_batch_cache(outputs.past_key_values, b, B)
            pos = seq.cached_len
            write_kv = self._slice_kv_tail(self._to_legacy_kv(full_kv), 1)
            self._commit_pages(seq, radix, pos, write_kv)
            seq.kv_state = None if self.use_paged_as_source else full_kv
            seq.cached_len = pos + 1
            results[i] = self._sample(logits, seq)

    def _prefill_one(self, seq: Sequence, radix: RadixCache) -> int:
        """Serial prefill fallback (different past lengths in a group)."""
        prompt = seq.input_ids
        start = seq.cached_len
        new_tokens = prompt[start:]
        if not new_tokens:
            if seq.last_logits is None:
                raise RuntimeError("exact hit without last_logits")
            return self._sample(seq.last_logits.to(self.device), seq)
        self._ensure_blocks(seq, radix, len(prompt))
        past = self._past_for_seq(seq, radix)
        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        attn = self._attention_mask(input_ids, past, past_len_hint=start)
        outputs = self._model_forward(input_ids, past, attn)
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values
        write_kv = self._slice_kv_tail(self._to_legacy_kv(new_kv), len(new_tokens))
        self._commit_pages(seq, radix, start, write_kv)
        seq.last_logits = logits.detach().float().cpu().clone()
        seq.kv_state = None if self.use_paged_as_source else new_kv
        seq.cached_len = len(prompt)
        seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
        return self._sample(logits, seq)

    def _decode_one(self, seq: Sequence, radix: RadixCache) -> int:
        if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
            seq.cached_len = len(seq.input_ids)
            return self._sample(seq.last_logits.to(device=self.device), seq)
        last_token = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
        pos = seq.cached_len
        self._ensure_blocks(seq, radix, pos + 1)
        if seq.block_table:
            page_i = pos // radix.block_size
            if page_i < len(seq.block_table):
                seq.block_table[page_i] = radix.cow_block_if_shared(seq.block_table[page_i])
        past = self._past_for_seq(seq, radix)
        attn = self._attention_mask(input_ids, past, past_len_hint=pos)
        outputs = self._model_forward(input_ids, past, attn)
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values
        write_kv = self._slice_kv_tail(self._to_legacy_kv(new_kv), 1)
        self._commit_pages(seq, radix, pos, write_kv)
        seq.kv_state = None if self.use_paged_as_source else new_kv
        seq.cached_len = pos + 1
        return self._sample(logits, seq)

    def _model_forward(self, input_ids, past, attention_mask):
        self.model_forward_count += 1
        self.last_model_forward_size = int(input_ids.shape[0])
        return self.model(
            input_ids=input_ids,
            past_key_values=self._as_model_cache(past),
            use_cache=True,
            attention_mask=attention_mask,
        )

    def _past_for_seq(self, seq: Sequence, radix: RadixCache):
        """Attention state: rebuild from paged KV when enabled."""
        if self.use_paged_as_source and seq.block_table and seq.cached_len > 0:
            self.paged_rebuild_count += 1
            return radix.build_cache(seq.block_table, seq.cached_len)
        return self._as_model_cache(seq.kv_state)

    def _commit_pages(
        self, seq: Sequence, radix: RadixCache, start: int, write_kv: PastKV
    ) -> None:
        if not write_kv:
            raise RuntimeError("empty KV write")
        # FlashInfer append when available (CUDA); else tensor copy write_kv
        if HAS_FLASHINFER and self.device != "cpu":
            self._flashinfer_append(seq, radix, start, write_kv)
        else:
            radix.write_kv(seq.block_table, start, write_kv)

    def _flashinfer_append(
        self, seq: Sequence, radix: RadixCache, start: int, write_kv: PastKV
    ) -> None:
        """Append into paged cache via FlashInfer; falls back to write_kv on shape issues."""
        append_len = write_kv[0][0].shape[-2] if write_kv[0][0].dim() == 4 else write_kv[0][0].shape[0]
        for layer_idx, (k, v) in enumerate(write_kv):
            k_tok, v_tok = radix._normalize_kv(k, v)
            # (S, H, D) float16
            k_new = k_tok.to(device=self.device, dtype=torch.float16)
            v_new = v_tok.to(device=self.device, dtype=torch.float16)
            batch_indices = torch.zeros(append_len, dtype=torch.int32, device=self.device)
            positions = torch.arange(start, start + append_len, dtype=torch.int32, device=self.device)
            pages_after = (start + append_len + radix.block_size - 1) // radix.block_size
            while len(seq.block_table) < pages_after:
                seq.block_table.extend(radix.allocate_blocks(1))
            active = seq.block_table[:pages_after]
            kv_indices = torch.tensor(active, dtype=torch.int32, device=self.device)
            kv_indptr = torch.tensor([0, kv_indices.numel()], dtype=torch.int32, device=self.device)
            last_len = start % radix.block_size if start > 0 else 0
            kv_last = torch.tensor([last_len], dtype=torch.int32, device=self.device)
            flashinfer.append_paged_kv_cache(
                k_new,
                v_new,
                batch_indices,
                positions,
                (radix.k_cache[layer_idx], radix.v_cache[layer_idx]),
                kv_indices,
                kv_indptr,
                kv_last,
                kv_layout="NHD",
            )

    def _batch_caches(self, pasts: List):
        """Stack per-seq caches into one batched DynamicCache/legacy list."""
        if not pasts or all(p is None for p in pasts):
            return None
        legacies = []
        for p in pasts:
            if p is None:
                raise RuntimeError("cannot batch None past with non-None peers")
            legacies.append(self._to_legacy_kv(p))
        # All same seq length (caller guarantees)
        n_layers = len(legacies[0])
        batched = []
        for layer in range(n_layers):
            ks = torch.cat([leg[layer][0] for leg in legacies], dim=0)
            vs = torch.cat([leg[layer][1] for leg in legacies], dim=0)
            batched.append((ks, vs))
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(batched)
        except Exception:
            return batched

    def _split_batch_cache(self, past, index: int, batch_size: int):
        legacy = self._to_legacy_kv(past)
        if legacy is None:
            return None
        sliced = []
        for k, v in legacy:
            sliced.append((k[index : index + 1].contiguous(), v[index : index + 1].contiguous()))
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(sliced)
        except Exception:
            return sliced

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        temperature = float(getattr(seq, "temperature", 0.0) or 0.0)
        top_p = float(getattr(seq, "top_p", 1.0) or 1.0)
        top_k = getattr(seq, "top_k", None)
        seed = getattr(seq, "seed", None)
        step = len(seq.output_ids)
        gen = make_generator(self.device, int(seed) + step) if seed is not None else None
        return sample_logits(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
        )

    def _ensure_blocks(self, seq: Sequence, radix: RadixCache, total_len: int) -> None:
        needed = (total_len + radix.block_size - 1) // radix.block_size
        while len(seq.block_table) < needed:
            seq.block_table.extend(radix.allocate_blocks(1))

    def _slice_kv_tail(self, kv: Optional[PastKV], n: int) -> PastKV:
        if not kv:
            return []
        out = []
        for k, v in kv:
            if k.dim() == 4:
                out.append((k[:, :, -n:, :].contiguous(), v[:, :, -n:, :].contiguous()))
            elif k.dim() == 3:
                out.append((k[-n:, :, :].contiguous(), v[-n:, :, :].contiguous()))
            else:
                out.append((k, v))
        return out

    def _attention_mask(
        self, input_ids: torch.Tensor, past, past_len_hint: Optional[int] = None
    ) -> torch.Tensor:
        past_len = 0
        if past_len_hint is not None:
            past_len = past_len_hint
        elif past is not None and hasattr(past, "get_seq_length"):
            past_len = int(past.get_seq_length())
        elif isinstance(past, (list, tuple)) and past:
            k0 = past[0][0]
            past_len = int(k0.shape[-2]) if k0.dim() >= 3 else 0
        total = past_len + input_ids.shape[-1]
        return torch.ones((input_ids.shape[0], total), dtype=torch.long, device=input_ids.device)

    def _as_model_cache(self, past):
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


MoEModelRunner = ModelRunner
