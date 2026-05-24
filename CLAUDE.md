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

**Runtime:** `en-core-web-sm` (pinned URL; update wheel URL if spaCy is upgraded), `misaki[ja]`, `misaki[zh]`, `mlx-audio`, `numpy`, `streamlit`

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`).

## Architecture

### Files

- `streamlit_app.py` — main app: text input, language/gender/voice selection (one voice per request), speed control, audio playback, phoneme tokenization, pronunciation tips
- `voice_grades.py` — quality-grade table (`VOICE_GRADES`), rank table (`_GRADE_RANK`), and `_grade_rank` helper extracted from the Kokoro model card; consumed by the voice picker for sorting and labeling
- `tests/conftest.py` — mocks `streamlit`, `mlx_audio`, `misaki`, and `huggingface_hub` for import
- `tests/test_app.py` — unit tests

### Key Functions

- `ensure_repo_downloaded` — calls `huggingface_hub.snapshot_download` once per process; tries `local_files_only=True` first and only shows the download spinner when files are missing
- `get_voices` — walks the local snapshot's `voices/` directory and returns voice IDs for the given language, sorted by quality grade (best first, ties alphabetical)
- `generate_speech` — generator yielding audio arrays per chunk; takes `lang_code` parameter
- `generate_one` — runs `generate_speech` for the chosen voice inside an `st.status` block, concatenates chunks, and returns a single `VoiceResult`
- `load_pipeline` — cached global model via `mlx_audio.tts.utils.load_model` (no lang_code parameter); called lazily on first Generate click
- `load_tokenizer` — cached G2P tokenizer via direct `misaki` usage per language
- `_create_g2p` — creates language-specific misaki G2P object
- `tokenize_text` — returns phoneme string without running inference
- `_format_voice` — formats a raw voice ID into a display label with optional grade suffix (e.g. `af_heart` → `"Heart (female) — A"`); ungraded voices show just `"Name (gender)"`. Used as `format_func` on voice widgets.
- `_grade_rank` — maps a voice ID to its numeric sort rank via `VOICE_GRADES` + `_GRADE_RANK` (both in `voice_grades.py`); ungraded voices get a sentinel rank that sorts last
- `_filter_voices_by_gender` — narrows a voice list to one gender (`"f"` or `"m"`), or returns unchanged for `None` (i.e. "All")
- `_default_voice` — returns a default voice for the current language/gender combination: prefers the `DEFAULT_VOICE_BY_LANG` entry when it matches the gender filter (e.g. `"af_heart"` for American English), otherwise falls back to the highest-graded voice in the gender-filtered list. Returns `None` only when no voices match.
- `_reset_selected_voice` — `on_change` callback for the Gender selectbox; resets `selected_voice` to the new default but leaves `current_output` intact
- `_on_language_change` — `on_change` callback for the Language selectbox; resets `selected_voice` and clears `current_output` (prior audio belongs to a different language context)
- `render_phonemes` — renders the `Phoneme Tokens` expander with `st.code`; `expanded` flag toggles open state
- `render_output` — displays the audio player for the current `VoiceResult` and calls `render_phonemes` once; returns early if `result is None`

### Model

[Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) (`load_model` from `mlx_audio.tts.utils`), 82M params in bf16 precision. Sample rate: 24000 Hz. MLX backend for Apple Silicon.

### Supported Languages

a=American English, b=British English, e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Brazilian Portuguese, z=Mandarin Chinese — 9 languages

### Voice Discovery

On first launch, `ensure_repo_downloaded` calls `huggingface_hub.snapshot_download` to fetch the model and all voice files in one event (~160 MB), with a spinner shown only when the local HuggingFace cache is incomplete (detected via `snapshot_download(..., local_files_only=True)` raising `LocalEntryNotFoundError`). `get_voices` then walks the local snapshot's `voices/` directory and sorts results by quality grade (best first, ties broken alphabetically) using the `VOICE_GRADES` table in `voice_grades.py` (sourced from [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)). Voice files follow the naming convention `{lang}{gender}_{name}` (e.g. `af_heart` — American English, female, "heart") with `.safetensors` extension. Ungraded voices (Spanish, Brazilian Portuguese, or any future addition not in `VOICE_GRADES`) sort to the end. After the initial download the app is fully offline; voices added upstream require clearing the HuggingFace cache to pick up.

### Performance

- MLX backend runs natively on Apple Silicon (no PyTorch or MPS fallback needed)
- `@st.cache_resource` caches the model globally, tokenizers per language, and the snapshot path returned by `ensure_repo_downloaded`
- `@st.cache_data` caches voice lists per language code (local filesystem walk, no TTL needed)
- `ensure_repo_downloaded` first attempts `snapshot_download(..., local_files_only=True)` and only shows the spinner + downloads when the local cache is incomplete
- `load_pipeline()` is deferred until the user clicks Generate, so initial page render is not blocked by model load
- `generate_speech` uses `np.asarray(..., dtype=np.float32)` to avoid copying chunks that are already float32

### UI

- Text input with `label_visibility="collapsed"` (no visible label) and no character cap
- Language, Gender, and Voice selectors rendered in a 3-column row with `label_visibility="collapsed"` (no visible labels)
- Gender selectbox offers `All` / `Female` / `Male` (mapped via the `GENDERS` constant)
- Voice display uses `_format_voice` to transform raw IDs (e.g. `af_heart`) into labels with quality grade (e.g. `"Heart (female) — A"`); ungraded voices show without the grade suffix
- Voices from `get_voices` are sorted by quality grade (best first, ties alphabetical); `_filter_voices_by_gender` narrows them to the selected gender while preserving the quality order
- Voice is a single selectbox (`index=None` with a "Select a voice" placeholder when no default applies). Defaults via `_default_voice`: prefers `DEFAULT_VOICE_BY_LANG` (`af_heart` for American English), otherwise the highest-graded voice for the current language/gender. Changing Language calls `_on_language_change` (resets voice and clears prior output); changing Gender calls `_reset_selected_voice` (resets voice only, preserves output). When the filter yields no voices, an `st.info("No voices match this filter.")` renders in place of the selectbox.
- Speed slider (0.5–2.0, default 1.0); `disabled=not selected_voice` so it's inert when nothing is picked
- Two-button row: Generate (primary), Tokenize
- On initial render, `ensure_repo_downloaded` may show `st.spinner("Downloading Kokoro model and voices (one-time, ~160 MB)...")` if the local HuggingFace cache is incomplete; otherwise no spinner appears
- If the first-launch download fails (e.g. offline with no cache), the script shows `st.error(...)` and halts via `st.stop()` instead of leaking a Python traceback
- Chunk-by-chunk generation progress via `st.status` (rendered inside `generate_one`)
- Model loads lazily on the first Generate click, shown via `st.spinner("Loading model...")`
- Tokenize button: shows phoneme tokens without generating audio (uses misaki G2P directly via `render_phonemes`)
- Phoneme token expander rendered below the audio output via `render_phonemes`, using the result's phonemes
- Generated audio displayed in browser player via `st.audio` (built-in download available from the player's menu)
- Errors shown with `st.exception()`
- "Tips" expander at the bottom of the page shows Kokoro pronunciation syntax (`PRONUNCIATION_TIPS` constant)
- Session state (`st.session_state`) persists current output across reruns

## Resources

- [MLX Model](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Original Model](https://github.com/hexgrad/kokoro)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)
