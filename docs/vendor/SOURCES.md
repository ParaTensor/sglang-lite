# Vendor sources (DeepSeek-V4-Flash only)

This tree holds **copied** code from upstream projects. Runtime must **not**
`import sglang` or `import vllm` as packages.

| Path (planned) | Upstream | Pin (commit / tag) | License | Notes |
|----------------|----------|--------------------|---------|-------|
| `engine/vendor/deepseek_infer/` | DeepSeek official `inference/` under HF ckpt | _TBD after first vendor PR_ | Apache-2.0 / model license | Model graph subset |
| `engine/vendor/sglang_v4/` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | _TBD_ | Apache-2.0 | V4 decode / graph / MLA-MoE call sites only |
| `engine/vendor/vllm_v4/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | _TBD_ | Apache-2.0 | Only if faster/clearer than SGLang slice for SM120 |

## Rules

1. Every copied file keeps original copyright headers.
2. Record exact upstream commit SHA when adding or refreshing files.
3. Rewrite imports to `sglang_lite.vendor.*` (or relative package paths).
4. Delete non-V4 branches and multi-model registries after copy.
5. Prefer fewer files on the decode hot path over “clean” layering.

## Refresh procedure

```bash
# Example (do not commit secrets)
# 1. clone upstream at PIN
# 2. copy listed files into engine/vendor/<name>/
# 3. fix imports; run V4 smoke + thruput vs SGLang
# 4. update this table + NOTICE
```
