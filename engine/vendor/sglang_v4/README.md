# sglang_v4 vendor

## Runtime status

**REFERENCE ONLY.** The live decode graph is `../deepseek_infer/` (DeepSeek official).

SGLang's `deepseek_v4.py` pulls the full `sglang.srt` stack (distributed, eplb,
compilation, NPU backends). Copying it into the import path would re-introduce
the giant package we refuse to depend on.

## What is here

| Path | Role |
|------|------|
| `reference/python/sglang/srt/models/deepseek_v4.py` | Full SGLang V4 model (import map for future leaf ports) |
| `reference/.../decode_cuda_graph_runner.py` | Capture/replay patterns to mirror in `v4_runner/cuda_graph.py` |
| `reference/.../kernels/ops/attention/dsv4/*` | Op call sites (sgl-kernel / Triton) to wire as leaves |
| `reference/docs/.../DeepSeek-V4.mdx` | Launch / perf cookbook |

## Integration plan (V2)

1. Port **leaf ops only** from `dsv4/*` behind `KernelBackend` (no srt imports).
2. Mirror **static buffer + CUDA graph** capture from `decode_cuda_graph_runner.py`
   into `sglang_lite.v4_runner.cuda_graph` around the **official** Transformer.
3. Never `import sglang` at runtime.
