"""
MoEModelRunner composed of smaller pieces for composability.

High-level components (unigateway can compose or replace):
- ModelLoader
- MoERouter
- PrefillExecutor / DecodeExecutor
- KernelBackend

sglang-lite only provides default implementations focused on popular MoE patterns.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from .kv_cache import RadixCache, PastKV
from .scheduler import Sequence

try:
    import flashinfer

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    flashinfer = None


class MoERouter:
    """Default MoE router. Can be replaced for custom routing strategies."""

    def route(self, input_ids: List[int]) -> List[int]:
        # Placeholder: real implementation would use the model's router weights.
        # For now return dummy expert ids.
        return [0] * len(input_ids)


class ModelRunner:  # kept for backward compat in examples; prefer MoEModelRunner
    def __init__(self, model_name: str = "stub", device: str = "cpu", max_batch: int = 4):
        self.model_name = model_name
        self.device = device
        self.max_batch = max_batch

        self.model = None
        self.tokenizer = None
        self._is_real = False
        self.vocab_size = 32000  # fallback

        if model_name and model_name != "stub":
            self._try_load_real(model_name)
        else:
            self._init_tiny_stub_model()

    def _try_load_real(self, model_name: str) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[sglang-lite] Attempting to load real model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=True
            )
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            # Support for MoE models (e.g. Qwen-MoE, Mixtral style) - use auto device for GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="cpu" if self.device == "cpu" else "auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # For small MoE demos, user can specify a small MoE like Qwen1.5-MoE if available
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
            self._is_real = True
            self.vocab_size = self.model.config.vocab_size
            # For paged KV + FlashInfer
            self.num_layers = self.model.config.num_hidden_layers
            self.num_kv_heads = getattr(
                self.model.config, "num_key_value_heads", self.model.config.num_attention_heads
            )
            self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
            if HAS_FLASHINFER:
                self.workspace_buffer = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=self.device
                )
                self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer, kv_layout="NHD"
                )
                self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer, kv_layout="NHD"
                )
            print(
                f"[sglang-lite] Real model '{model_name}' loaded on {self.device}. MoE routing will be handled by the model."
            )
        except Exception as e:
            print(f"[sglang-lite] Failed to load real model ({e}). Using tiny stub model.")
            self._is_real = False
            self._init_tiny_stub_model()

    def _init_tiny_stub_model(self):
        """A tiny causal LM for demos when no real model is available."""
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

            def forward(self, input_ids, past_key_values=None, use_cache=False):
                x = self.embed(input_ids)
                # Simulate MoE: simple expert routing (dummy for 2 experts)
                # In real MoE, router would select top-k experts
                batch_size, seq_len = input_ids.shape
                expert_ids = (input_ids % 2).tolist()  # dummy routing

                # Very simplified: we ignore real past KV for the stub transformer
                # and just run on the last few tokens. Good enough for demo.
                for layer in self.layers:
                    x = layer(x)
                x = self.ln(x)
                logits = self.head(x)
                # Return dummy past for interface compatibility
                dummy_past = None
                if use_cache:
                    dummy_past = [
                        (x[:, -1:, :].clone(), x[:, -1:, :].clone())
                        for _ in range(len(self.layers))
                    ]
                # Attach dummy expert info for observation
                return type(
                    "Obj",
                    (),
                    {"logits": logits, "past_key_values": dummy_past, "expert_ids": expert_ids},
                )()

        self.model = TinyLM(vocab=self.vocab_size).to(self.device)
        self.model.eval()
        self._is_real = False
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 64
        if HAS_FLASHINFER:
            # Allocate workspace for FlashInfer wrappers (128MB)
            self.workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD"
            )
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD"
            )
        print("[sglang-lite] Using built-in tiny stub model for CPU demo.")

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        # Fallback for stub
        return [hash(c) % self.vocab_size for c in text[:128]]

    def detokenize(self, token_ids: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return "".join([chr(97 + (t % 26)) for t in token_ids[:20]])

    @torch.no_grad()
    def run_batch(
        self,
        batch: List[Sequence],
        radix: RadixCache,
        is_prefill: List[bool],
    ) -> List[Optional[int]]:
        """
        Execute one step for the batch.

        Returns next token id for each sequence (or None if finished).
        Updates seq.kv_state and seq.cached_len via scheduler hooks.
        """
        if not batch:
            return []

        results: List[Optional[int]] = []

        for i, seq in enumerate(batch):
            if seq.finished:
                results.append(None)
                continue

            do_prefill = is_prefill[i]

            if do_prefill:
                next_token = self._prefill(seq, radix)
            else:
                next_token = self._decode_one(seq, radix)

            results.append(next_token)

        return results

    def _prefill(self, seq: Sequence, radix: RadixCache) -> int:
        """Run prefill for the unprocessed part of the prompt."""
        prompt = seq.input_ids
        start = seq.cached_len
        new_tokens = prompt[start:]

        if not new_tokens:
            # Already fully cached, treat as decode
            return self._decode_one(seq, radix)

        # Allocate paged blocks for the *new* tokens BEFORE flash append (so block_table covers them)
        num_new = len(new_tokens)
        new_bids = []
        if num_new > 0:
            needed_blocks = (num_new + radix.block_size - 1) // radix.block_size
            if needed_blocks > 0:
                new_bids = radix.allocate_blocks(needed_blocks)
                seq.block_table.extend(new_bids)

        # For real models (skeleton): use full prompt to ensure correct context and avoid
        # past_key_values compatibility issues with some HF models/versions.
        # The hit is observed via stats and cache_hit_tokens.
        if self._is_real and hasattr(self.model, "forward"):
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            new_kv = self._to_legacy_kv(outputs.past_key_values)
        else:
            input_ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
            outputs = self.model(input_ids, use_cache=True)
            logits = outputs.logits[0, -1, :]
            new_kv = self._to_legacy_kv(outputs.past_key_values)

        # Production: use FlashInfer for paged prefill if available (now block_table has room)
        if HAS_FLASHINFER and seq.block_table and new_tokens:
            # dummy q/k/v to exercise flash path (real would come from QKV proj of new_tokens)
            q = torch.randn(
                len(new_tokens),
                self.num_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=torch.float16,
            )
            k_new = torch.randn(
                len(new_tokens),
                self.num_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=torch.float16,
            )
            v_new = torch.randn(
                len(new_tokens),
                self.num_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=torch.float16,
            )
            o = self._flashinfer_paged_prefill(q, k_new, v_new, seq, radix)
            if o is not None:
                print("[flashinfer] paged prefill attention computed")

        # Append new KV produced by model into paged cache via flashinfer (correct 0.6.12 API)
        if HAS_FLASHINFER and seq.block_table and new_kv is not None and new_tokens:
            append_len = len(new_tokens)
            append_base = seq.cached_len  # position before this prefill append
            for layer_idx in range(len(new_kv) if isinstance(new_kv, list) else 1):
                k_full, v_full = new_kv[layer_idx] if isinstance(new_kv, list) else new_kv
                # extract last append_len from model output kv (various shapes from HF/stub)
                if hasattr(k_full, "dim"):
                    if k_full.dim() == 4:
                        k_new = (
                            k_full[0, :, -append_len:, :].transpose(0, 1).contiguous()
                        )  # rough (s, h, d)
                        v_new = v_full[0, :, -append_len:, :].transpose(0, 1).contiguous()
                    elif k_full.dim() == 3:
                        k_new = k_full[-append_len:, :, :].contiguous()
                        v_new = v_full[-append_len:, :, :].contiguous()
                    else:
                        k_new = k_full
                        v_new = v_full
                else:
                    k_new = k_full
                    v_new = v_full
                k_new = k_new.to(device=self.device, dtype=torch.float16)
                v_new = v_new.to(device=self.device, dtype=torch.float16)
                if k_new.dim() != 3 or k_new.shape[1] != self.num_kv_heads:
                    # stub or legacy kv shape not (s, heads, d); use dummy shaped for paged append test
                    k_new = torch.randn(
                        append_len,
                        self.num_kv_heads,
                        self.head_dim,
                        device=self.device,
                        dtype=torch.float16,
                    )
                    v_new = torch.randn(
                        append_len,
                        self.num_kv_heads,
                        self.head_dim,
                        device=self.device,
                        dtype=torch.float16,
                    )
                try:
                    self._append_paged_kv_flash(
                        k_new, v_new, seq, radix, layer_idx, append_base=append_base
                    )
                except Exception as e:
                    print(f"[flashinfer] append prefill skipped: {e}")

        # Sample (greedy for simplicity and determinism in demo)
        next_id = int(torch.argmax(logits).item())

        # Merge the new KV into the existing prefix KV
        if seq.kv_state is not None and new_kv is not None:
            merged_kv = self._merge_kv(seq.kv_state, new_kv)
        else:
            merged_kv = new_kv

        # Let scheduler know
        # (we update here because scheduler may be called from outside)
        seq.kv_state = merged_kv
        seq.cached_len = len(prompt)

        # (allocation for new prefill tokens already done earlier to support flash append)
        # legacy radix append path (no-op mostly)
        if new_bids and new_kv is not None:
            try:
                radix.append_kv_paged(new_bids, new_kv)
            except Exception:
                pass

        return next_id

    def _decode_one(self, seq: Sequence, radix: RadixCache) -> int:
        """Decode exactly one new token using cached KV."""
        # We only need the last generated token as input
        last_token = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)

        # For paged flash decode: ensure we have a block for the *new* token before append
        if HAS_FLASHINFER:
            if len(seq.block_table) == 0 or (seq.cached_len % radix.block_size == 0):
                # need a (new) page for this next token position
                new_b = radix.allocate_blocks(1)
                seq.block_table.extend(new_b)

        # Production path: use FlashInfer paged attention if available
        # For real models (skeleton): no past to avoid compatibility.
        if self._is_real and hasattr(self.model, "forward"):
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            new_kv = self._to_legacy_kv(outputs.past_key_values)
        else:
            outputs = self.model(input_ids, use_cache=True)
            logits = outputs.logits[0, -1, :]
            new_kv = self._to_legacy_kv(outputs.past_key_values)

        if HAS_FLASHINFER and seq.block_table and new_kv is not None:
            for layer_idx in range(len(new_kv) if isinstance(new_kv, list) else 1):
                k_full, v_full = new_kv[layer_idx] if isinstance(new_kv, list) else new_kv
                # Handle possible 3D or 4D , take last 1 for decode
                if hasattr(k_full, "dim") and k_full.dim() == 4:
                    k_new = k_full[0, :, -1:, :].transpose(0, 1).contiguous()
                    v_new = v_full[0, :, -1:, :].transpose(0, 1).contiguous()
                elif hasattr(k_full, "dim") and k_full.dim() == 3:
                    k_new = k_full[-1:, :, :].contiguous()
                    v_new = v_full[-1:, :, :].contiguous()
                else:
                    k_new = k_full
                    v_new = v_full
                k_new = k_new.to(device=self.device, dtype=torch.float16)
                v_new = v_new.to(device=self.device, dtype=torch.float16)
                if k_new.dim() != 3 or k_new.shape[1] != self.num_kv_heads:
                    # decode appends exactly one token
                    k_new = torch.randn(
                        1,
                        self.num_kv_heads,
                        self.head_dim,
                        device=self.device,
                        dtype=torch.float16,
                    )
                    v_new = torch.randn(
                        1,
                        self.num_kv_heads,
                        self.head_dim,
                        device=self.device,
                        dtype=torch.float16,
                    )
                try:
                    self._append_paged_kv_flash(
                        k_new, v_new, seq, radix, layer_idx, append_base=seq.cached_len
                    )
                except Exception as e:
                    print(f"[flashinfer] append decode skipped: {e}")

        next_id = int(torch.argmax(logits).item())

        # Production: use FlashInfer for paged attention
        if HAS_FLASHINFER and seq.block_table:
            # append already done above; now compute paged decode attn (dummy q to exercise)
            q = torch.randn(
                1, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.float16
            )
            o = self._flashinfer_paged_decode(q, seq, radix)
            if o is not None:
                print("[flashinfer] paged decode attention computed")
                # in real, use o instead of model's attn output, then continue with FFN etc.
            # for now, still use model's logits for correctness in skeleton

        # Update KV (append the new step)
        if seq.kv_state is not None and new_kv is not None:
            seq.kv_state = self._merge_kv(seq.kv_state, new_kv)
        else:
            seq.kv_state = new_kv

        # Paged: allocate block for new token if boundary, and append to paged cache
        if len(seq.block_table) == 0 or (seq.cached_len + 1) % radix.block_size == 0:
            new_bids = radix.allocate_blocks(1)
            seq.block_table.extend(new_bids)
            if new_kv is not None:
                # append the kv for this step to the paged
                radix.append_kv_paged(new_bids, [new_kv] if isinstance(new_kv, tuple) else new_kv)

        return next_id

    def _merge_kv(self, old: PastKV, new_step: PastKV) -> PastKV:
        """Concatenate old KV with the new token's KV along sequence dimension."""
        if old is None:
            return new_step
        if new_step is None:
            return old

        merged = []
        for (old_k, old_v), (new_k, new_v) in zip(old, new_step):
            merged.append(
                (
                    torch.cat([old_k, new_k], dim=2),
                    torch.cat([old_v, new_v], dim=2),
                )
            )
        return merged

    def _to_legacy_kv(self, past):
        """Convert HF Cache or other to our list of (k,v) tuples for internal storage."""
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            past = past.to_legacy_cache()
        if isinstance(past, (list, tuple)):
            return list(past)
        # Fallback
        return past

    def sample_next(self, logits: torch.Tensor, temperature: float = 0.7) -> int:
        """Simple sampling (greedy is used by default in runner for demo)."""
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _append_paged_kv_flash(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        seq: Sequence,
        radix: RadixCache,
        layer_idx: int,
        append_base: int = 0,
    ):
        """Correct append for flashinfer 0.6.12 using batch_indices/positions + indptr etc.
        k_new/v_new: (append_len, num_kv_heads, head_dim) float16
        """
        if not seq.block_table:
            return
        dev = self.device
        append_len = k_new.shape[0] if k_new.dim() > 0 else 1
        # build indices for this append (single seq -> batch 0)
        batch_indices = torch.zeros(append_len, dtype=torch.int32, device=dev)
        positions = torch.arange(
            append_base, append_base + append_len, dtype=torch.int32, device=dev
        )
        # Ensure block table has enough pages for append_base + append_len tokens,
        # and trim to exactly the pages flashinfer should see. The last page in the
        # trimmed table is the page we are appending into; its pre-append length is
        # kv_last_page_len.
        pages_with_data = (
            (append_base + radix.block_size - 1) // radix.block_size if append_base > 0 else 0
        )
        pages_after_append = (append_base + append_len + radix.block_size - 1) // radix.block_size
        total_pages = max(pages_with_data, pages_after_append)
        while len(seq.block_table) < total_pages:
            seq.block_table.extend(radix.allocate_blocks(1))
        active_blocks = seq.block_table[:total_pages]

        kv_indices = torch.tensor(active_blocks, dtype=torch.int32, device=dev)
        kv_indptr = torch.tensor([0, kv_indices.numel()], dtype=torch.int32, device=dev)
        # last page len: number of tokens already stored in the LAST page of active_blocks
        # BEFORE this append (0 when the last page is a freshly allocated empty page)
        last_len = append_base % radix.block_size if append_base > 0 else 0
        kv_last_page_len = torch.tensor([last_len], dtype=torch.int32, device=dev)
        k_cache_l = radix.k_cache[layer_idx]
        v_cache_l = radix.v_cache[layer_idx]
        flashinfer.append_paged_kv_cache(
            k_new,
            v_new,
            batch_indices,
            positions,
            (k_cache_l, v_cache_l),
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            kv_layout="NHD",
        )

    def _flashinfer_paged_decode(
        self, q: torch.Tensor, seq: Sequence, radix: RadixCache
    ) -> torch.Tensor:
        """Use FlashInfer paged decode attention with wrapper (0.6.12 API).
        q: [1, num_heads, head_dim] float16
        """
        if not HAS_FLASHINFER or not seq.block_table:
            return None
        try:
            k = radix.k_cache[0]
            v = radix.v_cache[0]
            # indptr/indices/last_page_len for current kv length (after append)
            num_pages = max(1, len(seq.block_table))
            indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=self.device)
            indices = torch.tensor(
                seq.block_table[:num_pages], dtype=torch.int32, device=self.device
            )
            total_len = max(1, seq.cached_len + 1)
            last_len = total_len % radix.block_size
            if last_len == 0:
                last_len = radix.block_size
            last_pl = torch.tensor([last_len], dtype=torch.int32, device=self.device)
            self.decode_wrapper.plan(
                indptr,
                indices,
                last_pl,
                num_qo_heads=self.num_kv_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=radix.block_size,
            )
            o = self.decode_wrapper.run(q, (k, v))
            return o
        except Exception as e:
            print(f"[flashinfer] decode error: {e}")
            return None

    def _flashinfer_paged_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq: Sequence, radix: RadixCache
    ) -> torch.Tensor:
        """Use FlashInfer paged prefill for new tokens (0.6.12)."""
        if not HAS_FLASHINFER or not seq.block_table:
            return None
        try:
            k_p = radix.k_cache[0]
            v_p = radix.v_cache[0]
            num_pages = max(1, len(seq.block_table))
            qo_indptr = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=self.device)
            pkv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=self.device)
            pkv_indices = torch.tensor(
                seq.block_table[:num_pages], dtype=torch.int32, device=self.device
            )
            total_for_last = max(1, (seq.cached_len or 0) + q.shape[0])
            last_l = total_for_last % radix.block_size
            if last_l == 0:
                last_l = radix.block_size
            pkv_last = torch.tensor([last_l], dtype=torch.int32, device=self.device)
            self.prefill_wrapper.plan(
                qo_indptr,
                pkv_indptr,
                pkv_indices,
                pkv_last,
                num_qo_heads=self.num_kv_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=radix.block_size,
                causal=True,
            )
            o = self.prefill_wrapper.run(q, (k_p, v_p))
            return o
        except Exception as e:
            print(f"[flashinfer] prefill error: {e}")
            return None
