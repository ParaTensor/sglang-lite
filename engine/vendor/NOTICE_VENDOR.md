# Vendor notice — DeepSeek-V4-Flash slices

This directory holds **copied** source from upstream projects for the
sglang-lite V4-Flash-only product. Runtime must **not** `import sglang` or
`import vllm` as packages.

## License inventory

| Tree | Upstream | License | Status |
|------|----------|---------|--------|
| `deepseek_infer/` | DeepSeek official `inference/` + `encoding/` (HF) | Apache-2.0 / model terms | **Pinned** `60d8d707…` — live graph |
| `sglang_v4/` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | Apache-2.0 | **Pinned** `e732c0a9…` — reference only |
| `vllm_v4/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 | Skeleton |

Exact commit SHAs live in `docs/vendor/SOURCES.md` when files are added.

## Rules

1. Keep original copyright headers on every copied file.
2. Rewrite imports to `sglang_lite.vendor.*` (or package-local relative imports).
3. Delete non-V4 branches after copy; do not preserve multi-model registries.
4. Refresh only with a deliberate pin bump recorded in SOURCES.md.
