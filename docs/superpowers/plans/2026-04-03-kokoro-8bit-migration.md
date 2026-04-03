# Kokoro-82M-8bit Migration Plan

> **Implementation note:** The final implementation uses `Kokoro-82M-bf16` instead of `Kokoro-82M-8bit` due to a shape mismatch bug in mlx-audio's sanitize function ([Blaizzy/mlx-audio#623](https://github.com/Blaizzy/mlx-audio/issues/623)). Additionally, `generate_speech` yields audio arrays only (not tuples) because mlx-audio's `GenerationResult` lacks a `phonemes` attribute; phonemes are obtained via `tokenize_text` separately.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the TTS pipeline from `kokoro` + PyTorch to `mlx-audio` + MLX, using the quantized `mlx-community/Kokoro-82M-8bit` model for Apple Silicon Macs.

**Architecture:** Replace `KPipeline` with `mlx_audio.tts.utils.load_model` for inference, use `misaki` directly for phoneme tokenization (instead of wrapping through kokoro), and update voice discovery to use `.safetensors` files from the 8bit model repo. One global model instance serves all languages (lang_code passed at generation time).

**Tech Stack:** `mlx-audio`, `misaki`, `streamlit`, `numpy`, `scipy`, `huggingface_hub`

**Spec:** `docs/superpowers/specs/2026-04-03-kokoro-8bit-migration-design.md`

---

### Task 1: Update pyproject.toml dependencies

**Files:**
- Modify: `pyproject.toml:6-16`

- [ ] **Step 1: Update dependencies**

In `pyproject.toml`, replace the `dependencies` list. Remove `kokoro>=0.9.4` and `torch`. Add `mlx-audio`.

```toml
dependencies = [
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    "misaki[ja]",
    "misaki[zh]",
    "mlx-audio",
    "numpy>=1.26.0",
    "soundfile",
    "streamlit",
    "scipy",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: swap kokoro+torch for mlx-audio in dependencies"
```

---

### Task 2: Migrate conftest, imports, constants, get_voices, and load_pipeline

This is the foundational swap. All these pieces are interconnected through imports and `REPO_ID`, so they change together. After this task, all existing tests pass with the new mocks.

**Files:**
- Modify: `tests/conftest.py` (full rewrite of mocks)
- Modify: `streamlit_app.py:1-14` (imports/constants)
- Modify: `streamlit_app.py:266-281` (get_voices, load_pipeline)
- Modify: `streamlit_app.py:449-451` (load_pipeline call site)
- Modify: `tests/test_app.py:7-25` (imports)
- Modify: `tests/test_app.py:54-68` (TestModelConstants)
- Modify: `tests/test_app.py:102-131` (TestGetVoices)
- Modify: `tests/test_app.py:133-142` (TestLoadPipeline)

- [ ] **Step 1: Update conftest.py**

Replace the kokoro mock with mlx_audio and misaki mocks. Update voice file entries from `.pt` to `.safetensors`.

Full new `tests/conftest.py`:

```python
import sys
from unittest.mock import MagicMock

# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = lambda f: f
_st.cache_data = lambda **_kw: lambda f: f
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
_st.toggle.return_value = False
_st.multiselect.return_value = []
_st.session_state = {}
sys.modules["streamlit"] = _st

# Mock mlx_audio to prevent model downloads on import
_mlx_audio = MagicMock()
_mlx_audio_tts = MagicMock()
_mlx_audio_tts_utils = MagicMock()
sys.modules["mlx_audio"] = _mlx_audio
sys.modules["mlx_audio.tts"] = _mlx_audio_tts
sys.modules["mlx_audio.tts.utils"] = _mlx_audio_tts_utils

# Mock misaki to prevent espeak-ng dependency in tests
_misaki = MagicMock()
_misaki_en = MagicMock()
_misaki_ja = MagicMock()
_misaki_zh = MagicMock()
_misaki_espeak = MagicMock()
_misaki.en = _misaki_en
_misaki.ja = _misaki_ja
_misaki.zh = _misaki_zh
_misaki.espeak = _misaki_espeak
sys.modules["misaki"] = _misaki
sys.modules["misaki.en"] = _misaki_en
sys.modules["misaki.ja"] = _misaki_ja
sys.modules["misaki.zh"] = _misaki_zh
sys.modules["misaki.espeak"] = _misaki_espeak

# Mock huggingface_hub to prevent network calls on import
_hf_hub = MagicMock()
_hf_hub.list_repo_tree.return_value = [
    MagicMock(rfilename="voices/af_heart.safetensors"),
    MagicMock(rfilename="voices/af_bella.safetensors"),
    MagicMock(rfilename="voices/am_adam.safetensors"),
    MagicMock(rfilename="voices/bf_alice.safetensors"),
    MagicMock(rfilename="voices/bm_daniel.safetensors"),
    MagicMock(rfilename="voices/jf_alpha.safetensors"),
    MagicMock(rfilename="voices/zf_xiaobei.safetensors"),
    MagicMock(rfilename="voices/ef_dora.safetensors"),
    MagicMock(rfilename="voices/ff_siwis.safetensors"),
    MagicMock(rfilename="voices/hf_alpha.safetensors"),
    MagicMock(rfilename="voices/if_sara.safetensors"),
    MagicMock(rfilename="voices/pf_dora.safetensors"),
]
sys.modules["huggingface_hub"] = _hf_hub
```

- [ ] **Step 2: Update streamlit_app.py imports and constants**

Replace lines 1-18 of `streamlit_app.py` with:

```python
import io
import random
import time
from collections.abc import Generator

import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from huggingface_hub import list_repo_tree
from mlx_audio.tts.utils import load_model

MODEL_NAME = "Kokoro-82M-8bit"
SAMPLE_RATE = 24000
REPO_ID = "mlx-community/Kokoro-82M-8bit"
HISTORY_MAX = 20
CHAR_LIMIT = 5000
```

Changes: removed `import os`, removed `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`, replaced `from kokoro import KPipeline` with `from mlx_audio.tts.utils import load_model`, updated `MODEL_NAME` and `REPO_ID`.

- [ ] **Step 3: Update get_voices to use .safetensors**

Replace the `get_voices` function (lines 266-276) with:

```python
@st.cache_data(ttl=3600)
def get_voices(lang_code: str) -> list[str]:
    return sorted(
        voice
        for entry in list_repo_tree(REPO_ID, path_in_repo="voices")
        if (name := getattr(entry, "rfilename", ""))
        and name.startswith("voices/")
        and name.endswith(".safetensors")
        and len(voice := name.removeprefix("voices/").removesuffix(".safetensors")) >= 2
        and voice[0] == lang_code
    )
```

Only change: `.pt` replaced with `.safetensors` in two places.

- [ ] **Step 4: Update load_pipeline to use load_model**

Replace the `load_pipeline` function (lines 279-281) with:

```python
@st.cache_resource
def load_pipeline() -> object:
    return load_model(REPO_ID)
```

No `lang_code` parameter — one global model serves all languages.

- [ ] **Step 5: Update load_pipeline call site**

Replace line 450-451:

```python
with st.spinner("Loading model..."):
    pipeline = load_pipeline(lang_code)
```

with:

```python
with st.spinner("Loading model..."):
    pipeline = load_pipeline()
```

- [ ] **Step 6: Update test imports**

In `tests/test_app.py`, update the imports (lines 7-25). Remove `load_tokenizer` from the import (it will be re-added in Task 3 after the signature changes). The import block becomes:

```python
from streamlit_app import (
    CHAR_LIMIT,
    HISTORY_MAX,
    LANGUAGES,
    LONG_SAMPLES,
    MODEL_NAME,
    PRONUNCIATION_TIPS,
    REPO_ID,
    SAMPLE_RATE,
    SAMPLES,
    _wav_bytes,
    add_to_history,
    generate_speech,
    get_voices,
    load_pipeline,
    render_output,
    tokenize_text,
)
```

- [ ] **Step 7: Update TestModelConstants**

In `tests/test_app.py`, update the two assertions in `TestModelConstants`:

```python
class TestModelConstants:
    def test_model_name(self) -> None:
        assert MODEL_NAME == "Kokoro-82M-8bit"

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 24000

    def test_repo_id(self) -> None:
        assert REPO_ID == "mlx-community/Kokoro-82M-8bit"

    def test_history_max(self) -> None:
        assert HISTORY_MAX == 20

    def test_char_limit(self) -> None:
        assert CHAR_LIMIT == 5000
```

- [ ] **Step 8: Update TestGetVoices**

The `test_returns_correct_voices` test stays the same (voice names unchanged). The `test_skips_entries_without_rfilename` test needs its fixture entry updated to use `.safetensors`:

No change needed — this test adds a `MagicMock(spec=[])` (no rfilename) and checks it's skipped. The conftest already provides `.safetensors` entries. Test stays as-is.

- [ ] **Step 9: Update TestLoadPipeline**

Replace the `TestLoadPipeline` class:

```python
class TestLoadPipeline:
    def test_returns_pipeline(self) -> None:
        pipeline = load_pipeline()
        assert pipeline is not None

    def test_called_with_repo_id(self) -> None:
        from mlx_audio.tts.utils import load_model

        load_pipeline()
        load_model.assert_called_with(REPO_ID)
```

- [ ] **Step 10: Run tests**

```bash
uv run pytest tests/test_app.py -v
```

Expected: all tests pass except `TestTokenizeText::test_load_tokenizer_passes_model_false` (this test checks the old KPipeline API — will be fixed in Task 3).

- [ ] **Step 11: Commit**

```bash
git add streamlit_app.py tests/conftest.py tests/test_app.py
git commit -m "refactor: migrate to mlx-audio for model loading and voice discovery"
```

---

### Task 3: Migrate tokenization to direct misaki

Replace `KPipeline(lang_code=..., model=False)` with direct misaki G2P usage. The misaki package provides language-specific G2P (grapheme-to-phoneme) classes.

**Files:**
- Modify: `streamlit_app.py:30-40` (add ESPEAK_LANGUAGES constant after LANGUAGES)
- Modify: `streamlit_app.py:283-290` (load_tokenizer, tokenize_text, add _create_g2p)
- Modify: `tests/test_app.py:7-25` (add load_tokenizer back to imports)
- Modify: `tests/test_app.py:145-191` (TestTokenizeText)

- [ ] **Step 1: Update TestTokenizeText for new misaki API**

The misaki G2P returns `(phonemes_string, tokens)` as a tuple from a single call (no iteration). Replace `TestTokenizeText` in `tests/test_app.py`:

```python
class TestTokenizeText:
    def _mock_g2p(self, phonemes: str) -> MagicMock:
        from misaki import en

        mock_g2p = MagicMock(return_value=(phonemes, None))
        en.G2P.return_value = mock_g2p
        return mock_g2p

    def test_returns_phonemes(self) -> None:
        self._mock_g2p("hɛlˈoʊ wˈɜːld")

        result = tokenize_text("hello world", "a")

        assert result == "hɛlˈoʊ wˈɜːld"

    def test_single_word(self) -> None:
        self._mock_g2p("hɛlˈoʊ")

        result = tokenize_text("hello", "a")

        assert result == "hɛlˈoʊ"

    def test_returns_empty_for_empty_phonemes(self) -> None:
        self._mock_g2p("")

        result = tokenize_text("", "a")

        assert result == ""

    def test_returns_empty_for_none_phonemes(self) -> None:
        from misaki import en

        en.G2P.return_value = MagicMock(return_value=(None, None))

        result = tokenize_text("", "a")

        assert result == ""

    def test_british_english_uses_british_g2p(self) -> None:
        self._mock_g2p("hɛlˈəʊ")

        tokenize_text("hello", "b")

        from misaki import en

        call_kwargs = en.G2P.call_args[1]
        assert call_kwargs["british"] is True

    def test_japanese_uses_ja_g2p(self) -> None:
        from misaki import ja

        ja.JAG2P.return_value = MagicMock(return_value=("konniʧiwa", None))

        result = tokenize_text("こんにちは", "j")

        assert result == "konniʧiwa"
        ja.JAG2P.assert_called_once()

    def test_spanish_uses_espeak_g2p(self) -> None:
        from misaki import espeak

        espeak.EspeakG2P.return_value = MagicMock(return_value=("ola", None))

        result = tokenize_text("hola", "e")

        assert result == "ola"
        espeak.EspeakG2P.assert_called_with(language="es")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_app.py::TestTokenizeText -v
```

Expected: FAIL — `tokenize_text` still uses the old KPipeline API.

- [ ] **Step 3: Add ESPEAK_LANGUAGES constant and _create_g2p helper**

In `streamlit_app.py`, add `ESPEAK_LANGUAGES` after the `LANGUAGES` dict (after line 40):

```python
ESPEAK_LANGUAGES: dict[str, str] = {
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}
```

Then replace `load_tokenizer` and `tokenize_text` (the block after `load_pipeline`) with:

```python
def _create_g2p(lang_code: str) -> object:
    if lang_code in ("a", "b"):
        from misaki import en, espeak as mespeak

        british = lang_code == "b"
        fallback = mespeak.EspeakFallback(british=british)
        return en.G2P(trf=False, british=british, fallback=fallback, unk="")
    if lang_code == "j":
        from misaki import ja

        return ja.JAG2P()
    if lang_code == "z":
        from misaki import zh

        return zh.ZHG2P()
    from misaki import espeak as mespeak

    return mespeak.EspeakG2P(language=ESPEAK_LANGUAGES[lang_code])


@st.cache_resource
def load_tokenizer(lang_code: str) -> object:
    return _create_g2p(lang_code)


def tokenize_text(text: str, lang_code: str) -> str:
    phonemes, _ = load_tokenizer(lang_code)(text)
    return phonemes or ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_app.py::TestTokenizeText -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "refactor: migrate tokenization from kokoro to direct misaki usage"
```

---

### Task 4: Migrate generate_speech and UI call site

Update `generate_speech` to use the mlx-audio `model.generate()` API. Add `lang_code` parameter. Update audio conversion from PyTorch tensor to numpy-compatible MLX array.

**Files:**
- Modify: `streamlit_app.py:293-305` (generate_speech function)
- Modify: `streamlit_app.py:481-483` (generate_speech call site)
- Modify: `tests/test_app.py:193-283` (TestGenerateSpeech)

- [ ] **Step 1: Update TestGenerateSpeech for new API**

The key changes: `pipeline` parameter becomes `model`, audio is a plain numpy array (no `.cpu().numpy()` chain), and `lang_code` is a new parameter.

Replace `TestGenerateSpeech` in `tests/test_app.py`:

```python
class TestGenerateSpeech:
    def _mock_model(
        self, *, audio_length: int = 48000, phonemes: str = "hɛlˈoʊ"
    ) -> MagicMock:
        model = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.random.randn(audio_length).astype(np.float32)
        chunk.phonemes = phonemes
        model.generate.return_value = [chunk]
        return model

    def test_yields_audio_and_phonemes(self) -> None:
        model = self._mock_model()

        results = list(generate_speech("hello", "af_heart", model, lang_code="a"))

        assert len(results) == 1
        audio, phonemes = results[0]
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)
        assert phonemes == "hɛlˈoʊ"

    def test_calls_model_generate_with_correct_args(self) -> None:
        model = self._mock_model()

        list(generate_speech("test text", "af_heart", model, speed=1.5, lang_code="b"))

        model.generate.assert_called_once_with(
            text="test text", voice="af_heart", speed=1.5, lang_code="b"
        )

    def test_default_speed_and_lang_code(self) -> None:
        model = self._mock_model()

        list(generate_speech("test", "af_heart", model))

        model.generate.assert_called_once_with(
            text="test", voice="af_heart", speed=1.0, lang_code="a"
        )

    def test_yields_multiple_chunks(self) -> None:
        model = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = np.ones(100, dtype=np.float32)
        chunk1.phonemes = "wˈʌn"
        chunk2 = MagicMock()
        chunk2.audio = np.zeros(200, dtype=np.float32)
        chunk2.phonemes = "tˈuː"
        model.generate.return_value = [chunk1, chunk2]

        results = list(generate_speech("long text", "af_heart", model, lang_code="a"))

        assert len(results) == 2
        assert results[0][0].shape == (100,)
        assert results[1][0].shape == (200,)
        assert results[0][1] == "wˈʌn"
        assert results[1][1] == "tˈuː"

    def test_output_is_float32(self) -> None:
        model = self._mock_model()

        results = list(generate_speech("test", "af_heart", model, lang_code="a"))

        assert results[0][0].dtype == np.float32

    def test_raises_on_empty_chunks(self) -> None:
        model = MagicMock()
        model.generate.return_value = []

        with pytest.raises(ValueError, match="No audio generated"):
            list(generate_speech("test", "af_heart", model, lang_code="a"))

    def test_skips_chunks_with_none_audio(self) -> None:
        model = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = None
        chunk1.phonemes = "skipped"
        chunk2 = MagicMock()
        chunk2.audio = np.ones(100, dtype=np.float32)
        chunk2.phonemes = "kˈɛpt"
        model.generate.return_value = [chunk1, chunk2]

        results = list(generate_speech("test", "af_heart", model, lang_code="a"))

        assert len(results) == 1
        assert results[0][0].shape == (100,)
        assert results[0][1] == "kˈɛpt"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_app.py::TestGenerateSpeech -v
```

Expected: FAIL — `generate_speech` still uses the old pipeline API.

- [ ] **Step 3: Update generate_speech implementation**

Replace the `generate_speech` function in `streamlit_app.py`:

```python
def generate_speech(
    text: str,
    voice: str,
    model: object,
    speed: float = 1.0,
    lang_code: str = "a",
) -> Generator[tuple[np.ndarray, str], None, None]:
    generated = False
    for result in model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code):
        if result.audio is not None:
            generated = True
            yield np.array(result.audio, dtype=np.float32), result.phonemes or ""
    if not generated:
        raise ValueError("No audio generated. Check your input text.")
```

- [ ] **Step 4: Update generate_speech call site**

In the UI section of `streamlit_app.py`, update the `generate_speech` call (inside the `for v in selected_voices:` loop) to pass `lang_code`:

Replace:
```python
                    for i, (audio_chunk, phonemes) in enumerate(
                        generate_speech(text_input, v, pipeline, speed=speed), 1
                    ):
```

with:
```python
                    for i, (audio_chunk, phonemes) in enumerate(
                        generate_speech(text_input, v, pipeline, speed=speed, lang_code=lang_code), 1
                    ):
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_app.py::TestGenerateSpeech -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/test_app.py -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "refactor: migrate generate_speech to mlx-audio model.generate API"
```

---

### Task 5: Update CLAUDE.md

Update documentation to reflect the migration.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Apply these changes to `CLAUDE.md`:

1. **Project Overview** — change `Kokoro-82M` to `Kokoro-82M-8bit`, add "(Mac only)" note:
   ```
   Streamlit web app for generating multilingual speech using [Kokoro-82M-8bit](https://huggingface.co/mlx-community/Kokoro-82M-8bit), a quantized text-to-speech model optimized for Apple Silicon. Mac only.
   ```

2. **Dependencies section** — replace Runtime line:
   ```
   **Runtime:** `en-core-web-sm` (pinned URL; update wheel URL if spaCy is upgraded), `misaki[ja]`, `misaki[zh]`, `mlx-audio`, `numpy`, `soundfile`, `streamlit`, `scipy`
   ```

3. **Architecture > Files** — update conftest description:
   ```
   - `tests/conftest.py` — mocks `streamlit`, `mlx_audio`, `misaki`, and `huggingface_hub` for import
   ```

4. **Architecture > Key Functions** — update `load_pipeline` and `load_tokenizer` descriptions:
   ```
   - `generate_speech` — generator yielding `(audio, phonemes)` tuples per chunk; takes `lang_code` parameter
   - `load_pipeline` — cached global model via `mlx_audio.tts.utils.load_model` (no lang_code parameter)
   - `load_tokenizer` — cached G2P tokenizer via direct `misaki` usage per language
   - `_create_g2p` — creates language-specific misaki G2P object
   - `tokenize_text` — returns phoneme string without running inference
   ```

5. **Architecture > Model** — update:
   ```
   [Kokoro-82M-8bit](https://huggingface.co/mlx-community/Kokoro-82M-8bit) (`load_model` from `mlx_audio.tts.utils`), 82M params quantized to 8-bit. Sample rate: 24000 Hz. MLX backend for Apple Silicon.
   ```

6. **Architecture > Voice Discovery** — update file extension:
   ```
   Voice files follow the naming convention `{lang}{gender}_{name}` (e.g. `af_heart` — American English, female, "heart") with `.safetensors` extension. Voices are cached per language code with `@st.cache_data`.
   ```

7. **Architecture > Performance** — replace first bullet:
   ```
   - MLX backend runs natively on Apple Silicon (no PyTorch or MPS fallback needed)
   - `@st.cache_resource` to cache model globally and tokenizers per language
   ```

8. **Resources** — update links:
   ```
   - [MLX Model](https://huggingface.co/mlx-community/Kokoro-82M-8bit)
   - [Original Model](https://github.com/hexgrad/kokoro)
   - [mlx-audio](https://github.com/Blaizzy/mlx-audio)
   ```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for kokoro-82M-8bit migration"
```

---

### Task 6: Final verification

Run the full quality suite to confirm everything works.

**Files:** none (read-only)

- [ ] **Step 1: Run tests**

```bash
uv run pytest tests/test_app.py -v
```

Expected: all tests PASS

- [ ] **Step 2: Run linter**

```bash
uv run ruff check .
```

Expected: no errors. If there are unused import warnings (e.g., `os` was removed), they should already be resolved from Task 2.

- [ ] **Step 3: Run formatter**

```bash
uv run ruff format .
```

Expected: no changes (code should already be formatted).

- [ ] **Step 4: Run type checker**

```bash
uv run ty check
```

Expected: no new errors. Note: `object` return types on `load_pipeline` and `load_tokenizer` are intentionally loose since the mlx-audio and misaki types don't share a common base.
