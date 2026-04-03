# Kokoro-82M-8bit Migration Design

## Overview

Migrate the TTS pipeline from `kokoro` + PyTorch to `mlx-audio` + MLX, using the quantized `mlx-community/Kokoro-82M-8bit` model. This targets Apple Silicon Macs only.

## Dependencies

**Remove:** `kokoro>=0.9.4`, `torch`

**Add:** `mlx-audio`

**Keep:** `misaki[ja]`, `misaki[zh]`, `en-core-web-sm`, `numpy`, `soundfile`, `streamlit`, `scipy`

Dropping PyTorch (~2GB) in favor of the lighter MLX backend, native to Apple Silicon.

## Constants & Imports

- `MODEL_NAME`: `"Kokoro-82M"` → `"Kokoro-82M-8bit"`
- `REPO_ID`: `"hexgrad/Kokoro-82M"` → `"mlx-community/Kokoro-82M-8bit"`
- Remove `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`
- Replace `from kokoro import KPipeline` with `from mlx_audio.tts.utils import load_model`

## Model Loading

**Before:** One `KPipeline` cached per language via `@st.cache_resource` on `load_pipeline(lang_code)`.

**After:** One model cached globally via `load_model(REPO_ID)`. The `lang_code` is passed at generation time to `model.generate()`, not at model load time. This simplifies caching — one model instance serves all languages.

The `load_pipeline` function signature changes from `load_pipeline(lang_code: str) -> KPipeline` to `load_pipeline() -> model` (no lang_code parameter).

## Generation

`generate_speech` gains a `lang_code` parameter. The pipeline call changes from:

```python
for result in pipeline(text, voice=voice, speed=speed):
    audio = result.audio.cpu().numpy().astype(np.float32)
```

to:

```python
for result in model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code):
    audio = np.array(result.audio, dtype=np.float32)
```

Audio conversion changes from PyTorch tensor (`.cpu().numpy()`) to MLX array (`np.array()`).

## Tokenization

Replace `KPipeline(lang_code=..., model=False)` with direct `misaki` usage. The `misaki` package is already a transitive dependency and provides the underlying phoneme tokenization.

The `load_tokenizer` and `tokenize_text` functions will call misaki's tokenizer API directly instead of wrapping through kokoro's pipeline.

## Voice Discovery

Keep using `huggingface_hub.list_repo_tree` for dynamic voice discovery. Verify during implementation whether voice files live in the `mlx-community/Kokoro-82M-8bit` repo or the original `hexgrad/Kokoro-82M` repo. Voice names and language codes are identical in both cases.

A separate `VOICE_REPO_ID` constant may be needed if voices are served from a different repo than the model.

## Tests

- Update `conftest.py` to mock `mlx_audio` instead of `kokoro`
- Mock `load_model` to return a mock with a `.generate()` method
- Test structure stays the same; `generate_speech` tests gain `lang_code` parameter
- Tokenization tests updated to match direct misaki usage

## CLAUDE.md

Update to reflect:
- New model name and repo
- `mlx-audio` replacing `kokoro` and `torch` in dependencies
- Mac-only target platform
- Any API changes in key functions
