# Kokoro Pipeline

Generate multilingual speech with the [Hexgrad Kokoro model](https://huggingface.co/hexgrad/Kokoro-82M) on Apple Silicon with MLX.

## Features

- 9 supported languages (American English, British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin Chinese)
- One-time model + voice download on first launch (~160 MB); fully offline thereafter
- Voice cards sorted by quality grade (best first), with grade in each title (e.g. "Heart (female) — A"); top 6 visible, rest behind a "Show All Voices" expander
- Gender filter via Female / Male checkboxes; either alone filters, neither or both shows all
- Per-card Play button generates audio on demand and embeds an audio player inline — A/B-compare multiple voices on the same text
- Per-card speed dropdown (0.7x–1.5x in 0.1 steps); cached audio invalidates when speed changes
- Chunk-by-chunk generation progress via `st.status` inside the active card
- Phoneme token display via standalone Tokenize button (renders inline below the text input)
- Pronunciation note always visible with Kokoro-specific syntax (custom phonemes, stress, intonation)
- In-browser audio playback per card (download via the audio player's menu)

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
