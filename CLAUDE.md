# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for generating multilingual speech using [Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16), a text-to-speech model optimized for Apple Silicon. Mac only.

## Installation

Requires `espeak-ng` system dependency.

```bash
uv sync --group dev
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

**System:** `espeak-ng`

**Runtime:** `en-core-web-sm` (pinned URL; update wheel URL if spaCy is upgraded), `misaki[ja]`, `misaki[zh]`, `mlx-audio`, `numpy`, `soundfile`, `streamlit`, `scipy`

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`).

## Architecture

### Files

- `streamlit_app.py` — single-file app: text input, language/voice selection, speed control, audio playback, voice comparison, phoneme tokenization, sample texts, long samples, character limit, pronunciation tips, session-based generation history
- `tests/conftest.py` — mocks `streamlit`, `mlx_audio`, `misaki`, and `huggingface_hub` for import
- `tests/test_app.py` — unit tests

### Key Functions

- `generate_speech` — generator yielding `(audio, phonemes)` tuples per chunk; takes `lang_code` parameter
- `load_pipeline` — cached global model via `mlx_audio.tts.utils.load_model` (no lang_code parameter)
- `load_tokenizer` — cached G2P tokenizer via direct `misaki` usage per language
- `_create_g2p` — creates language-specific misaki G2P object
- `tokenize_text` — returns phoneme string without running inference
- `add_to_history` — manages generation history (max 20 entries, newest first)
- `_wav_bytes` — converts a NumPy audio array to WAV bytes
- `render_output` — displays audio player, metrics, download button, phoneme expander

### Model

[Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) (`load_model` from `mlx_audio.tts.utils`), 82M params in bf16 precision. Sample rate: 24000 Hz. MLX backend for Apple Silicon.

### Supported Languages

a=American English, b=British English, e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Brazilian Portuguese, z=Mandarin Chinese — 9 languages

### Voice Discovery

Voices are discovered dynamically from the HuggingFace Hub (`mlx-community/Kokoro-82M-bf16`) via `huggingface_hub.list_repo_tree`. Voice files follow the naming convention `{lang}{gender}_{name}` (e.g. `af_heart` — American English, female, "heart") with `.safetensors` extension. Voices are cached per language code with `@st.cache_data`.

### Performance

- MLX backend runs natively on Apple Silicon (no PyTorch or MPS fallback needed)
- `@st.cache_resource` to cache model globally and tokenizers per language
- `@st.cache_data(ttl=3600)` to cache voice lists (1-hour TTL)
- `time.perf_counter()` for timing

### UI

- Text input with 5000-character limit (`CHAR_LIMIT`), live character count (red when exceeded)
- Random Sample button: populates text area with a language-aware sample from `SAMPLES` dict via session state wiring
- Long Sample button: populates text area with a literary excerpt from `LONG_SAMPLES` dict via session state wiring
- Pronunciation Tips expander: collapsed by default, shows Kokoro pronunciation syntax (`PRONUNCIATION_TIPS` constant)
- Language selection (9 languages via `LANGUAGES` dict)
- Voice selector (dynamically populated from HuggingFace Hub)
- Compare Voices toggle: switches voice selector to multiselect (max 3 voices)
- Speed slider (0.5–2.0, default 1.0)
- Four-button row: Generate (primary), Tokenize, Random Sample, Long Sample
- Chunk-by-chunk generation progress via `st.status` (per-voice in compare mode)
- Tokenize button: shows phoneme tokens without generating audio (uses model-free pipeline)
- Phoneme token expander (`st.expander` + `st.code`) below audio output; shared in compare mode
- Generated audio displayed in browser player via `st.audio`
- WAV download via `st.download_button` (saved with `scipy.io.wavfile.write`)
- Metrics via `st.metric`: model name, input characters, output duration, generation time
- Compare mode: shared Model + Input Characters metrics, per-voice Duration + Generation Time
- Errors shown with `st.exception()`
- Session state (`st.session_state`) persists current output and generation history across reruns
- Sidebar generation history (max 20 entries, newest first) with Load buttons

## Resources

- [MLX Model](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Original Model](https://github.com/hexgrad/kokoro)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)
