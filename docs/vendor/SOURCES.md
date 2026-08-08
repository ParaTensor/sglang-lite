# Vendor sources (DeepSeek-V4-Flash only)

This tree holds **copied** code from upstream projects. Runtime must **not**
`import sglang` or `import vllm` as packages.

In-tree layout: `engine/vendor/{deepseek_infer,sglang_v4,vllm_v4}/`.
Helper: `python scripts/vendor_v4_slice.py ...`.

| Path | Upstream | Pin (commit / tag) | License | Notes |
|------|----------|--------------------|---------|-------|
| `engine/vendor/deepseek_infer/` | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) `inference/` + `encoding/` | **`hf@60d8d70770c6776ff598c94bb586a859a38244f1`** | Apache-2.0 / model terms | **Live graph**: `model.py`, `kernel.py`, `encoding_dsv4.py`, convert/generate |
| `engine/vendor/sglang_v4/` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | **`e732c0a9dc071ca06026dba93887e8d77e631d04`** | Apache-2.0 | **REFERENCE ONLY** under `reference/` (full srt deps; not imported) |
| `engine/vendor/vllm_v4/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | _empty skeleton_ | Apache-2.0 | Optional if faster leaf for SM120 |

See also `engine/vendor/NOTICE_VENDOR.md` and each tree’s `VENDOR_PIN.txt`.

## Rules

1. Every copied file keeps original copyright headers.
2. Record exact upstream commit SHA when adding or refreshing files.
3. Rewrite imports to `sglang_lite.vendor.*` only when making a slice **runtime-importable**.
4. Delete non-V4 branches and multi-model registries after copy.
5. Prefer fewer files on the decode hot path over “clean” layering.
6. Prefer vendor graph over external `SGLANG_LITE_DSV4_INFER` when `model.py` is present.

## Live vs reference

| Tree | Runtime? | How loaded |
|------|----------|------------|
| `deepseek_infer/` | **Yes** | `sys.path` insert → `import model` / `import kernel` (matches upstream) |
| `sglang_v4/reference/` | **No** | Read for porting CUDA-graph / dsv4 leaf ops into `v4_runner` |
| `vllm_v4/` | **No** until filled | — |

## Refresh procedure

```bash
# Official graph (primary)
python scripts/vendor_v4_slice.py deepseek-infer \
  --src "$HF_CKPT/inference" \
  --pin "hf:deepseek-ai/DeepSeek-V4-Flash@<sha>" \
  --files model.py kernel.py config.json convert.py generate.py \
  --update-sources
# also copy encoding/encoding_dsv4.py next to model.py

# SGLang reference (manual rsync of listed paths; see sglang_v4/README.md)
```

## Pin log

- `deepseek-infer` pin=`hf:deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1` files=[model.py, kernel.py, config.json, convert.py, generate.py, requirements.txt, encoding_dsv4.py, test_encoding_dsv4.py]
- `sglang-v4` pin=`sgl-project/sglang@e732c0a9dc071ca06026dba93887e8d77e631d04` reference tree: deepseek_v4.py, decode_cuda_graph_runner.py, dsv4 ops, configs, DeepSeek-V4.mdx
