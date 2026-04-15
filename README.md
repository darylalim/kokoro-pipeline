# Kokoro Pipeline

Streamlit web app for generating multilingual speech using [Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16), a text-to-speech model optimized for Apple Silicon via [mlx-audio](https://github.com/Blaizzy/mlx-audio). Mac only.

## Features

- 9 supported languages (American English, British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin Chinese)
- Voices discovered dynamically from HuggingFace Hub, displayed with human-readable labels (e.g. "Heart (female)")
- Gender filter (All / Female / Male) narrows the voice list
- Multi-voice generation (up to 3 voices, played back side by side)
- Adjustable speech speed (0.5x–2.0x)
- Inline 5000-character input limit (enforced as you type)
- In-browser audio playback (download via the audio player's menu)
- Chunk-by-chunk generation progress (per-voice when multiple voices are selected)
- Phoneme token display with standalone Tokenize button
- Tips panel with Kokoro-specific pronunciation syntax (custom phonemes, stress, intonation)

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng)

## Installation

```bash
uv sync --group dev
uv run streamlit run streamlit_app.py
```

> **Note:** The spaCy model `en_core_web_sm` (required for English G2P) is installed automatically via `uv sync`.

## Development

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`
