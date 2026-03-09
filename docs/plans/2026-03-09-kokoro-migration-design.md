# Kokoro Migration Design

## Summary

Replace Chatterbox Multilingual (500M params, 7 monkey-patches) with Kokoro-82M (82M params, zero patches) for text-to-speech generation.

## Approach

Clean rewrite of `streamlit_app.py`. Remove all monkey-patches, voice cloning, and Chatterbox dependencies. Replace with Kokoro's `KPipeline` API.

## Dependencies

### Remove
- `chatterbox-tts`
- `peft`
- `setuptools<71`
- `uv` constraint dependencies (`ml-dtypes`, `pandas`)
- `uv` extra build dependencies (`pkuseg`)

### Add
- `kokoro>=0.9.4`
- `soundfile`
- `misaki[ja]` (Japanese phonemization)
- `misaki[zh]` (Chinese phonemization)

### Keep
- `numpy`, `streamlit`, `scipy`, `torch`

### System dependency
- `espeak-ng` (required by Kokoro for phonemization)

## Languages (9)

| Code | Language |
|------|----------|
| `a` | American English |
| `b` | British English |
| `e` | Spanish |
| `f` | French |
| `h` | Hindi |
| `i` | Italian |
| `j` | Japanese |
| `p` | Brazilian Portuguese |
| `z` | Mandarin Chinese |

## Architecture

### Model loading
- `KPipeline(lang_code=...)` cached with `@st.cache_resource`
- Sample rate: 24000 Hz (hardcoded by Kokoro)
- Device handled internally by Kokoro; set `PYTORCH_ENABLE_MPS_FALLBACK=1` via `os.environ` before torch import for Apple Silicon

### Generation
- `pipeline(text, voice=..., speed=...)` returns generator of `(graphemes, phonemes, audio)` chunks
- Concatenate chunks with `np.concatenate` into single array

### Voice selection
- Dynamically discover available voices from installed `kokoro` package
- Filter voices by selected language (first letter of voice name = language code)
- Default to first available voice for selected language

## UI Changes

### Remove
- Voice cloning file uploader
- CFG Weight slider
- Exaggeration slider

### Add
- Voice selector (dropdown, filtered by language)
- Speed slider (0.5–2.0, default 1.0)

### Keep
- Text input (300 char max)
- Language selector
- Generate button
- Audio player + WAV download
- Metrics (model name, input chars, output duration, generation time)

## Error Handling

- `espeak-ng` not installed: caught by existing `st.exception()` pattern
- MPS fallback: `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` before torch import
- Audio chunking: concatenate generator output before playback

## Tests

Rewrite to mock `kokoro.KPipeline` instead of `chatterbox`. Same structure: languages, model loading, generation, dependency setup.

## Dropped Features

- Voice cloning (not supported by Kokoro)
- 15 languages: Arabic, Danish, Dutch, Finnish, German, Greek, Hebrew, Korean, Malay, Norwegian, Polish, Russian, Swahili, Swedish, Turkish
