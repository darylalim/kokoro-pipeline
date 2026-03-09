# Kokoro Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Chatterbox TTS with Kokoro-82M, removing all monkey-patches and simplifying the app.

**Architecture:** Single-file Streamlit app using Kokoro's `KPipeline` for TTS. Voices discovered dynamically from HuggingFace Hub. No monkey-patches needed.

**Tech Stack:** Kokoro (`kokoro` package), Streamlit, NumPy, SciPy, `huggingface_hub` (transitive dep, used for voice discovery)

---

### Task 1: Update dependencies in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Replace dependencies**

```toml
[project]
name = "text-to-speech-pipeline"
version = "0.3.0"
description = "Streamlit web app for multilingual text-to-speech using Kokoro"
requires-python = ">=3.12"
dependencies = [
    "kokoro>=0.9.4",
    "misaki[ja]",
    "misaki[zh]",
    "numpy>=1.26.0",
    "soundfile",
    "streamlit",
    "scipy",
    "torch",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "ty",
]

[tool.ruff.lint.isort]
combine-as-imports = true

[tool.pytest.ini_options]
pythonpath = ["."]

[tool.ty.environment]
python-version = "3.12"
```

Remove: `chatterbox-tts`, `peft`, `setuptools<71`, `[tool.uv]` constraint-dependencies section, `[tool.uv.extra-build-dependencies]` section.

**Step 2: Sync dependencies**

Run: `uv sync --group dev`
Expected: Clean install without chatterbox/peft/diffusers.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: replace chatterbox deps with kokoro"
```

---

### Task 2: Rewrite test conftest

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Replace conftest with Kokoro mocks**

```python
import sys
from unittest.mock import MagicMock

# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = lambda f: f
_st.cache_data = lambda f: f
_st.selectbox.side_effect = lambda label, **_kw: {
    "Language": "American English",
    "Voice": "af_heart",
}.get(label, MagicMock())
_st.slider.side_effect = lambda label, **_kw: {
    "Speed": 1.0,
}.get(label, MagicMock())
_st.button.return_value = False
_st.text_area.return_value = ""
_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
sys.modules["streamlit"] = _st

# Mock kokoro to prevent model downloads on import
_kokoro = MagicMock()
sys.modules["kokoro"] = _kokoro

# Mock huggingface_hub to prevent network calls on import
_hf_hub = MagicMock()
_hf_hub.list_repo_tree.return_value = [
    MagicMock(rfilename="voices/af_heart.pt"),
    MagicMock(rfilename="voices/af_bella.pt"),
    MagicMock(rfilename="voices/am_adam.pt"),
    MagicMock(rfilename="voices/bf_alice.pt"),
    MagicMock(rfilename="voices/bm_daniel.pt"),
    MagicMock(rfilename="voices/jf_alpha.pt"),
    MagicMock(rfilename="voices/zf_xiaobei.pt"),
    MagicMock(rfilename="voices/ef_dora.pt"),
    MagicMock(rfilename="voices/ff_siwis.pt"),
    MagicMock(rfilename="voices/hf_alpha.pt"),
    MagicMock(rfilename="voices/if_sara.pt"),
    MagicMock(rfilename="voices/pf_dora.pt"),
]
sys.modules["huggingface_hub"] = _hf_hub
```

**Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: rewrite conftest mocks for kokoro"
```

---

### Task 3: Rewrite tests

**Files:**
- Modify: `tests/test_app.py`

**Step 1: Write all tests**

```python
from unittest.mock import MagicMock, patch

import numpy as np

from streamlit_app import (
    LANGUAGES,
    MODEL_NAME,
    SAMPLE_RATE,
    generate_speech,
    get_voices,
    load_pipeline,
)

EXPECTED_LANGUAGES = [
    "American English",
    "Brazilian Portuguese",
    "British English",
    "French",
    "Hindi",
    "Italian",
    "Japanese",
    "Mandarin Chinese",
    "Spanish",
]

EXPECTED_CODES = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}


class TestLanguages:
    def test_all_languages_present(self) -> None:
        assert sorted(LANGUAGES.keys()) == EXPECTED_LANGUAGES

    def test_language_codes(self) -> None:
        codes = set(LANGUAGES.values())
        assert codes == EXPECTED_CODES

    def test_language_count(self) -> None:
        assert len(LANGUAGES) == 9


class TestModelConstants:
    def test_model_name(self) -> None:
        assert MODEL_NAME == "Kokoro-82M"

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 24000


class TestGetVoices:
    def test_returns_voices_for_language(self) -> None:
        voices = get_voices("a")
        assert all(v[1] == "a" for v in voices)

    def test_returns_empty_for_unknown_language(self) -> None:
        voices = get_voices("x")
        assert voices == []

    def test_voices_are_sorted(self) -> None:
        voices = get_voices("a")
        assert voices == sorted(voices)


class TestLoadPipeline:
    def test_returns_pipeline(self) -> None:
        pipeline = load_pipeline("a")
        assert pipeline is not None

    def test_called_with_lang_code(self) -> None:
        from kokoro import KPipeline

        load_pipeline("a")
        KPipeline.assert_called_with(lang_code="a")


class TestGenerateSpeech:
    def _mock_pipeline(self, *, audio_length: int = 48000) -> MagicMock:
        pipeline = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.random.randn(audio_length).astype(np.float32)
        pipeline.return_value = [chunk]
        return pipeline

    def test_returns_audio_array(self) -> None:
        pipeline = self._mock_pipeline()

        audio = generate_speech("hello", "af_heart", pipeline)

        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)

    def test_calls_pipeline_with_correct_args(self) -> None:
        pipeline = self._mock_pipeline()

        generate_speech("test text", "af_heart", pipeline, speed=1.5)

        pipeline.assert_called_once_with("test text", voice="af_heart", speed=1.5)

    def test_default_speed(self) -> None:
        pipeline = self._mock_pipeline()

        generate_speech("test", "af_heart", pipeline)

        pipeline.assert_called_once_with("test", voice="af_heart", speed=1.0)

    def test_concatenates_multiple_chunks(self) -> None:
        pipeline = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = np.ones(100, dtype=np.float32)
        chunk2 = MagicMock()
        chunk2.audio = np.zeros(200, dtype=np.float32)
        pipeline.return_value = [chunk1, chunk2]

        audio = generate_speech("long text", "af_heart", pipeline)

        assert audio.shape == (300,)
        assert audio[:100].sum() == 100.0
        assert audio[100:].sum() == 0.0

    def test_output_is_float32(self) -> None:
        pipeline = self._mock_pipeline()

        audio = generate_speech("test", "af_heart", pipeline)

        assert audio.dtype == np.float32
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ -v`
Expected: FAIL — `streamlit_app` exports don't exist yet.

**Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test: rewrite tests for kokoro migration"
```

---

### Task 4: Rewrite streamlit_app.py

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Write the complete new app**

```python
import io
import os
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from huggingface_hub import list_repo_tree
from kokoro import KPipeline

MODEL_NAME = "Kokoro-82M"
SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"

LANGUAGES: dict[str, str] = {
    "American English": "a",
    "British English": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Japanese": "j",
    "Brazilian Portuguese": "p",
    "Mandarin Chinese": "z",
}


@st.cache_data
def get_voices(lang_code: str) -> list[str]:
    entries = list_repo_tree(REPO_ID, path_in_repo="voices")
    voices = []
    for entry in entries:
        name = entry.rfilename
        if name.endswith(".pt") and name.startswith("voices/"):
            voice = name.removeprefix("voices/").removesuffix(".pt")
            if len(voice) >= 2 and voice[1] == lang_code:
                voices.append(voice)
    return sorted(voices)


@st.cache_resource
def load_pipeline(lang_code: str) -> KPipeline:
    return KPipeline(lang_code=lang_code)


def generate_speech(
    text: str,
    voice: str,
    pipeline: KPipeline,
    speed: float = 1.0,
) -> np.ndarray:
    chunks = list(pipeline(text, voice=voice, speed=speed))
    audio = np.concatenate([c.audio for c in chunks])
    return audio.astype(np.float32)


st.title("Text to Speech Pipeline")
st.write("Generate multilingual speech with Kokoro.")

st.subheader("Text")
text_input = st.text_area(
    "Text",
    placeholder="Enter text...",
    max_chars=300,
    height=150,
    help="Maximum 300 characters per generation.",
)

st.subheader("Voice")
voice_col1, voice_col2 = st.columns(2)

with voice_col1:
    language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        help="Select a language for speech generation.",
    )

lang_code = LANGUAGES[language]

with voice_col2:
    voices = get_voices(lang_code)
    voice = st.selectbox(
        "Voice",
        options=voices,
        help="Select a voice. Names starting with 'f' are female, 'm' are male.",
    )

st.subheader("Style")
speed = st.slider(
    "Speed",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Speech rate multiplier. 1.0 is normal speed.",
)

with st.spinner("Loading model..."):
    pipeline = load_pipeline(lang_code)

if st.button("Generate", type="primary"):
    if text_input.strip():
        try:
            with st.spinner("Generating speech..."):
                start = time.perf_counter()
                audio_array = generate_speech(
                    text_input, voice, pipeline, speed=speed
                )
                eval_duration = round(time.perf_counter() - start, 2)
                output_duration = len(audio_array) / SAMPLE_RATE

            st.audio(audio_array, sample_rate=SAMPLE_RATE)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model", MODEL_NAME)
            col2.metric("Input Characters", len(text_input))
            col3.metric("Output Duration", f"{output_duration:.2f}s")
            col4.metric("Generation Time", f"{eval_duration}s")

            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, SAMPLE_RATE, audio_array)
            st.download_button(
                label="Download Audio",
                data=wav_buffer.getvalue(),
                file_name="speech.wav",
                mime="audio/wav",
            )

        except Exception as e:
            st.exception(e)
    else:
        st.warning("Enter text.")
```

**Step 2: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean.

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: replace chatterbox with kokoro TTS"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Rewrite CLAUDE.md to reflect Kokoro**

Update all sections: project overview, dependencies, architecture, supported languages, UI description. Remove the entire "Dependency Patches" section. Update resources links.

Key changes:
- Model: Kokoro-82M (82M params) from `kokoro` package
- Languages: 9 (a, b, e, f, h, i, j, p, z)
- Dependencies: `kokoro`, `misaki[ja]`, `misaki[zh]`, `soundfile` replace `chatterbox-tts`, `peft`, `setuptools`
- System dependency: `espeak-ng`
- UI: Voice selector + speed slider replace voice cloning + CFG/exaggeration sliders
- No dependency patches section

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for kokoro migration"
```

---

### Task 6: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 2: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean.

**Step 3: Run typecheck**

Run: `uv run ty check`
Expected: No new errors introduced.
