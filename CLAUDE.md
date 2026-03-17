# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for generating multilingual speech using [Kokoro-82M](https://github.com/hexgrad/kokoro), a lightweight text-to-speech model by [Hexgrad](https://github.com/hexgrad).

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

**Runtime:** `en-core-web-sm` (pinned URL; update wheel URL if spaCy is upgraded), `kokoro>=0.9.4`, `misaki[ja]`, `misaki[zh]`, `numpy`, `soundfile`, `streamlit`, `scipy`, `torch`

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`).

## Architecture

### Files

- `streamlit_app.py` — single-file app: text input, language/voice selection, speed control, audio playback, voice comparison, phoneme tokenization, session-based generation history
- `tests/conftest.py` — mocks `streamlit`, `kokoro`, and `huggingface_hub` for import
- `tests/test_app.py` — unit tests

### Key Functions

- `generate_speech` — generator yielding `(audio, phonemes)` tuples per chunk
- `load_pipeline` / `load_tokenizer` — cached pipeline (with model) and tokenizer (model-free)
- `tokenize_text` — returns joined phoneme string without running inference
- `add_to_history` — manages generation history (max 20 entries, newest first)
- `render_output` — displays audio player, metrics, download button, phoneme expander

### Model

[Kokoro-82M](https://github.com/hexgrad/kokoro) (`KPipeline` from `kokoro` package), 82M params. Sample rate: 24000 Hz. `repo_id` is passed explicitly to suppress default warnings.

### Supported Languages

a=American English, b=British English, e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Brazilian Portuguese, z=Mandarin Chinese — 9 languages

### Voice Discovery

Voices are discovered dynamically from the HuggingFace Hub (`hexgrad/Kokoro-82M`) via `huggingface_hub.list_repo_tree`. Voice files follow the naming convention `{lang}{gender}_{name}` (e.g. `af_heart` — American English, female, "heart"). Voices are cached per language code with `@st.cache_data`.

### Performance

- Device selection handled internally by Kokoro; `PYTORCH_ENABLE_MPS_FALLBACK=1` set via `os.environ` for Apple Silicon compatibility
- `@st.cache_resource` to cache pipeline and tokenizer per language
- `@st.cache_data(ttl=3600)` to cache voice lists (1-hour TTL)
- `time.perf_counter()` for timing

### UI

- Text input (no character limit, live character count)
- Language selection (9 languages via `LANGUAGES` dict)
- Voice selector (dynamically populated from HuggingFace Hub)
- Compare Voices toggle: switches voice selector to multiselect (max 3 voices)
- Speed slider (0.5–2.0, default 1.0)
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

- [GitHub Repo](https://github.com/hexgrad/kokoro)
- [Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M)
- [PyPI](https://pypi.org/project/kokoro/)
