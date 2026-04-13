# Kokoro Pipeline

Streamlit web app for generating multilingual speech using [Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16), a text-to-speech model optimized for Apple Silicon via [mlx-audio](https://github.com/Blaizzy/mlx-audio). Mac only.

## Features

- 9 supported languages (American English, British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin Chinese)
- Voices discovered dynamically from HuggingFace Hub
- Voice comparison mode (up to 3 voices side by side)
- Adjustable speech speed (0.5x-2.0x)
- Pronunciation tips with Kokoro-specific syntax (custom phonemes, stress, intonation)
- 5000-character input limit with visual indicator
- In-browser audio playback and WAV download
- Chunk-by-chunk generation progress (per-voice in compare mode)
- Phoneme token display with standalone Tokenize button
- Generation metrics: model name, input characters, output duration, generation time

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
