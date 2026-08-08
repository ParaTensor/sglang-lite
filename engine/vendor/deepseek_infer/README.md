# vendor/deepseek_infer — DeepSeek official inference graph

**Status**: placeholder (no `model.py` yet).

Copy the **minimal runnable subset** of the official `inference/` tree from a
DeepSeek-V4-Flash HF checkpoint (or release tarball):

```text
model.py          # Transformer + ModelArgs (required)
config.json       # optional ModelArgs-shaped defaults
kernel/ or ops/   # only if model.py imports them relatively
```

Do **not** vendor training scripts, unrelated demos, or the full weight files.

## How to add

```bash
# After downloading HF ckpt with inference/ next to config.json:
python scripts/vendor_v4_slice.py deepseek-infer \
  --src "$HF_CKPT/inference" \
  --pin "hf:DeepSeek-V4-Flash@<revision>"
```

Then update `docs/vendor/SOURCES.md` with the pin.

Load prefers this directory over `SGLANG_LITE_DSV4_INFER` when `model.py` exists
(`sglang_lite.model_loader.import_official_inference`).
