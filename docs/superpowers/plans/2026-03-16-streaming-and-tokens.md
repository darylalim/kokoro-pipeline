# Streaming Progress & Phoneme Token Display Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add chunk-by-chunk progress feedback during speech generation and phoneme token display (post-generation expander + standalone Tokenize button).

**Architecture:** Refactor `generate_speech` from returning a single array to a generator yielding `(audio, phonemes)` tuples. Add `tokenize_text` using a model-free pipeline. Progress shown via `st.status`; phonemes shown via `st.expander` + `st.code`.

**Tech Stack:** Python, Streamlit (`st.status`, `st.expander`, `st.code`), Kokoro (`KPipeline`), NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-streaming-and-tokens-design.md`

---

## File Structure

All changes are in existing files — no new files created.

| File | Changes |
|---|---|
| `streamlit_app.py` | Refactor `generate_speech`, add `load_tokenizer`, `tokenize_text`, update `render_output`, update Generate handler, add Tokenize button |
| `tests/conftest.py` | No changes needed — existing `MagicMock` handles new `st.status`, `st.expander`, `st.code` automatically |
| `tests/test_app.py` | Update `TestGenerateSpeech` mocks/assertions, update `TestRenderOutput`, add `TestTokenizeText` |

---

## Chunk 1: Refactor `generate_speech` to Generator

### Task 1: Update `generate_speech` tests for generator behavior

**Files:**
- Modify: `tests/test_app.py:7-18` (imports)
- Modify: `tests/test_app.py:104-161` (TestGenerateSpeech class)

- [ ] **Step 1: Update imports in test_app.py**

Add `load_tokenizer` and `tokenize_text` to the import block. These don't exist yet but will be needed by later tasks — importing them now will cause an `ImportError` until Task 3, so defer this to Task 3. For now, no import changes are needed.

- [ ] **Step 2: Rewrite `_mock_pipeline` helper and all tests**

The mock pipeline must return objects with `.audio` (a mock torch tensor supporting `.cpu().numpy()`) and `.phonemes` (a string). Replace the entire `TestGenerateSpeech` class in `tests/test_app.py:104-161` with:

```python
class TestGenerateSpeech:
    def _mock_tensor(self, data: np.ndarray) -> MagicMock:
        tensor = MagicMock()
        tensor.cpu.return_value = tensor
        tensor.numpy.return_value = data
        return tensor

    def _mock_pipeline(
        self, *, audio_length: int = 48000, phonemes: str = "hɛlˈoʊ"
    ) -> MagicMock:
        pipeline = MagicMock()
        chunk = MagicMock()
        chunk.audio = self._mock_tensor(
            np.random.randn(audio_length).astype(np.float32)
        )
        chunk.phonemes = phonemes
        pipeline.return_value = [chunk]
        return pipeline

    def test_yields_audio_and_phonemes(self) -> None:
        pipeline = self._mock_pipeline()

        results = list(generate_speech("hello", "af_heart", pipeline))

        assert len(results) == 1
        audio, phonemes = results[0]
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)
        assert phonemes == "hɛlˈoʊ"

    def test_calls_pipeline_with_correct_args(self) -> None:
        pipeline = self._mock_pipeline()

        list(generate_speech("test text", "af_heart", pipeline, speed=1.5))

        pipeline.assert_called_once_with("test text", voice="af_heart", speed=1.5)

    def test_default_speed(self) -> None:
        pipeline = self._mock_pipeline()

        list(generate_speech("test", "af_heart", pipeline))

        pipeline.assert_called_once_with("test", voice="af_heart", speed=1.0)

    def test_concatenates_multiple_chunks(self) -> None:
        pipeline = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = self._mock_tensor(np.ones(100, dtype=np.float32))
        chunk1.phonemes = "wˈʌn"
        chunk2 = MagicMock()
        chunk2.audio = self._mock_tensor(np.zeros(200, dtype=np.float32))
        chunk2.phonemes = "tˈuː"
        pipeline.return_value = [chunk1, chunk2]

        results = list(generate_speech("long text", "af_heart", pipeline))

        assert len(results) == 2
        assert results[0][0].shape == (100,)
        assert results[1][0].shape == (200,)
        assert results[0][1] == "wˈʌn"
        assert results[1][1] == "tˈuː"

    def test_output_is_float32(self) -> None:
        pipeline = self._mock_pipeline()

        results = list(generate_speech("test", "af_heart", pipeline))

        assert results[0][0].dtype == np.float32

    def test_raises_on_empty_chunks(self) -> None:
        pipeline = MagicMock()
        pipeline.return_value = []

        with pytest.raises(ValueError, match="No audio generated"):
            list(generate_speech("test", "af_heart", pipeline))

    def test_skips_chunks_with_none_audio(self) -> None:
        pipeline = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = None
        chunk1.phonemes = "skipped"
        chunk2 = MagicMock()
        chunk2.audio = self._mock_tensor(np.ones(100, dtype=np.float32))
        chunk2.phonemes = "kˈɛpt"
        pipeline.return_value = [chunk1, chunk2]

        results = list(generate_speech("test", "af_heart", pipeline))

        assert len(results) == 1
        assert results[0][0].shape == (100,)
        assert results[0][1] == "kˈɛpt"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestGenerateSpeech -v`
Expected: FAIL — `generate_speech` still returns `np.ndarray`, not a generator.

### Task 2: Implement `generate_speech` as generator

**Files:**
- Modify: `streamlit_app.py:1-3` (imports)
- Modify: `streamlit_app.py:49-60` (generate_speech function)

- [ ] **Step 1: Add `Generator` import**

Add `from collections.abc import Generator` to the imports at `streamlit_app.py:1`:

```python
import io
import os
import time
from collections.abc import Generator
```

- [ ] **Step 2: Replace `generate_speech` function**

Replace `streamlit_app.py:49-60` with:

```python
def generate_speech(
    text: str,
    voice: str,
    pipeline: KPipeline,
    speed: float = 1.0,
) -> Generator[tuple[np.ndarray, str], None, None]:
    generated = False
    for result in pipeline(text, voice=voice, speed=speed):
        if result.audio is not None:
            generated = True
            yield result.audio.cpu().numpy().astype(np.float32), result.phonemes or ""
    if not generated:
        raise ValueError("No audio generated. Check your input text.")
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py::TestGenerateSpeech -v`
Expected: All 7 tests PASS.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: Some tests outside `TestGenerateSpeech` may still pass (constants, voices, etc.). The Generate button handler in the module-level code calls `generate_speech` and expects an array, but since `st.button.return_value = False` in conftest, that code path is never hit during import. All tests should PASS.

- [ ] **Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "refactor: change generate_speech to generator yielding (audio, phonemes) tuples"
```

---

## Chunk 2: Add `tokenize_text` and phoneme display in `render_output`

### Task 3: Add `tokenize_text` tests and implementation

**Files:**
- Modify: `tests/test_app.py:7-18` (imports — add `load_tokenizer`, `tokenize_text`)
- Modify: `tests/test_app.py` (add `TestTokenizeText` class after `TestLoadPipeline`)
- Modify: `streamlit_app.py:44-47` (add `load_tokenizer` after `load_pipeline`)
- Modify: `streamlit_app.py` (add `tokenize_text` after `load_tokenizer`)

- [ ] **Step 1: Write `TestTokenizeText` tests**

First, update the import block in `tests/test_app.py:7-18` to add the new functions:

```python
from streamlit_app import (
    HISTORY_MAX,
    LANGUAGES,
    MODEL_NAME,
    REPO_ID,
    SAMPLE_RATE,
    add_to_history,
    generate_speech,
    get_voices,
    load_pipeline,
    load_tokenizer,
    render_output,
    tokenize_text,
)
```

Then add the following class after `TestLoadPipeline` (after line 101):

```python
class TestTokenizeText:
    def _mock_tokenizer_pipeline(self, phoneme_chunks: list[str]) -> MagicMock:
        results = []
        for p in phoneme_chunks:
            r = MagicMock()
            r.phonemes = p
            results.append(r)
        from kokoro import KPipeline

        KPipeline.return_value = MagicMock(return_value=results)  # type: ignore[union-attribute]
        return KPipeline.return_value  # type: ignore[union-attribute]

    def test_returns_joined_phonemes(self) -> None:
        self._mock_tokenizer_pipeline(["hɛlˈoʊ", "wˈɜːld"])

        result = tokenize_text("hello world", "a")

        assert result == "hɛlˈoʊ wˈɜːld"

    def test_single_chunk(self) -> None:
        self._mock_tokenizer_pipeline(["hɛlˈoʊ"])

        result = tokenize_text("hello", "a")

        assert result == "hɛlˈoʊ"

    def test_skips_empty_phonemes(self) -> None:
        self._mock_tokenizer_pipeline(["hɛlˈoʊ", "", "wˈɜːld"])

        result = tokenize_text("hello world", "a")

        assert result == "hɛlˈoʊ wˈɜːld"

    def test_returns_empty_for_no_phonemes(self) -> None:
        self._mock_tokenizer_pipeline([])

        result = tokenize_text("", "a")

        assert result == ""

    def test_load_tokenizer_passes_model_false(self) -> None:
        from kokoro import KPipeline

        load_tokenizer("a")

        KPipeline.assert_called_with(lang_code="a", model=False)  # type: ignore[union-attribute]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestTokenizeText -v`
Expected: FAIL — `ImportError: cannot import name 'load_tokenizer'` and `'tokenize_text'`.

- [ ] **Step 3: Implement `load_tokenizer` and `tokenize_text`**

Add after `load_pipeline` in `streamlit_app.py` (after line 47):

```python
@st.cache_resource
def load_tokenizer(lang_code: str) -> KPipeline:
    return KPipeline(lang_code=lang_code, model=False)


def tokenize_text(text: str, lang_code: str) -> str:
    tokenizer = load_tokenizer(lang_code)
    phonemes = []
    for result in tokenizer(text):
        if result.phonemes:
            phonemes.append(result.phonemes)
    return " ".join(phonemes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py::TestTokenizeText -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "feat: add load_tokenizer and tokenize_text for phoneme extraction"
```

### Task 4: Add phoneme display to `render_output`

**Files:**
- Modify: `tests/test_app.py` (TestRenderOutput — update `_make_result`, `_reset_st_mocks`, add new tests)
- Modify: `streamlit_app.py` (render_output function)

- [ ] **Step 1: Update `_make_result` and `_reset_st_mocks`, write new tests**

Update `_make_result` to include `"phonemes"` in `tests/test_app.py`:

```python
@staticmethod
def _make_result(
    voice: str = "af_heart", text: str = "hello", phonemes: str = "hɛlˈoʊ"
) -> dict[str, object]:
    return {
        "audio": np.ones(24000, dtype=np.float32),
        "voice": voice,
        "text": text,
        "speed": 1.0,
        "duration": 1.0,
        "generation_time": 0.5,
        "phonemes": phonemes,
    }
```

Update `_reset_st_mocks` to also reset `st.expander` and `st.code`:

```python
def _reset_st_mocks(self) -> None:
    st.audio.reset_mock()  # type: ignore[union-attribute]
    st.download_button.reset_mock()  # type: ignore[union-attribute]
    st.markdown.reset_mock()  # type: ignore[union-attribute]
    st.metric.reset_mock()  # type: ignore[union-attribute]
    st.expander.reset_mock()  # type: ignore[union-attribute]
    st.code.reset_mock()  # type: ignore[union-attribute]
```

Add the following tests at the end of `TestRenderOutput`:

```python
def test_single_result_shows_phoneme_expander(self) -> None:
    self._reset_st_mocks()
    render_output([self._make_result()])
    st.expander.assert_called_once_with("Phoneme Tokens")  # type: ignore[union-attribute]

def test_single_result_shows_phonemes_in_code(self) -> None:
    self._reset_st_mocks()
    render_output([self._make_result(phonemes="hɛlˈoʊ")])
    st.code.assert_called_once_with("hɛlˈoʊ")  # type: ignore[union-attribute]

def test_compare_shows_single_shared_phoneme_expander(self) -> None:
    self._reset_st_mocks()
    results = [
        self._make_result("af_heart", phonemes="hɛlˈoʊ"),
        self._make_result("af_bella", phonemes="hɛlˈoʊ"),
    ]
    render_output(results)
    st.expander.assert_called_once_with("Phoneme Tokens")  # type: ignore[union-attribute]
    st.code.assert_called_once_with("hɛlˈoʊ")  # type: ignore[union-attribute]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestRenderOutput::test_single_result_shows_phoneme_expander tests/test_app.py::TestRenderOutput::test_single_result_shows_phonemes_in_code tests/test_app.py::TestRenderOutput::test_compare_shows_single_shared_phoneme_expander -v`
Expected: FAIL — `render_output` does not call `st.expander` or `st.code`.

- [ ] **Step 3: Add phoneme expander to `render_output`**

In `streamlit_app.py`, update the `render_output` function.

For the **compare mode branch** (the `if len(results) > 1:` block), add after the `for result in results:` loop's download button (after the existing line with `key=f"download_{result['voice']}"`):

```python
        with st.expander("Phoneme Tokens"):
            st.code(results[0].get("phonemes", ""))
```

This goes at the same indentation level as the `for result in results:` loop (inside the `if len(results) > 1:` block but outside the for loop).

For the **single mode branch** (the `else:` block), add after the download button:

```python
        with st.expander("Phoneme Tokens"):
            st.code(result.get("phonemes", ""))
```

Using `.get("phonemes", "")` ensures backward compatibility if old history entries (without the `"phonemes"` key) are loaded.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py::TestRenderOutput -v`
Expected: All 10 tests PASS (7 existing + 3 new).

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS. Existing `_make_result` callers that don't pass `phonemes=` get the default `"hɛlˈoʊ"`.

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: Clean.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "feat: add phoneme token expander to render_output"
```

---

## Chunk 3: Update Generate handler and add Tokenize button

### Task 5: Update Generate handler and add Tokenize button

**Files:**
- Modify: `streamlit_app.py:193-220` (Generate button handler + add Tokenize handler)

- [ ] **Step 1: Replace the Generate button handler and add Tokenize handler**

Replace `streamlit_app.py:193-220` (from `if st.button("Generate"...` through the `except` block) and add the Tokenize handler before the `render_output` call. The full replacement:

```python
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    generate_clicked = st.button("Generate", type="primary")
with btn_col2:
    tokenize_clicked = st.button("Tokenize")

if generate_clicked:
    if not text_input.strip():
        st.warning("Enter text.")
    elif compare_mode and not selected_voices:
        st.warning("Select at least one voice.")
    else:
        try:
            results = []
            for v in selected_voices:
                start = time.perf_counter()
                with st.status(f"Generating {v}...", expanded=True) as status:
                    audio_chunks = []
                    phoneme_chunks = []
                    for i, (audio_chunk, phonemes) in enumerate(
                        generate_speech(text_input, v, pipeline, speed=speed), 1
                    ):
                        audio_chunks.append(audio_chunk)
                        phoneme_chunks.append(phonemes)
                        st.write(f"Chunk {i}...")
                    status.update(label=f"{v} complete!", state="complete")
                gen_time = round(time.perf_counter() - start, 2)
                audio_array = np.concatenate(audio_chunks)
                all_phonemes = " ".join(phoneme_chunks)
                results.append(
                    {
                        "audio": audio_array,
                        "voice": v,
                        "text": text_input,
                        "speed": speed,
                        "duration": len(audio_array) / SAMPLE_RATE,
                        "generation_time": gen_time,
                        "phonemes": all_phonemes,
                    }
                )
            st.session_state["current_output"] = results
            add_to_history(st.session_state["history"], results)
            st.rerun()
        except Exception as e:
            st.exception(e)

if tokenize_clicked:
    if not text_input.strip():
        st.warning("Enter text.")
    else:
        phonemes = tokenize_text(text_input, lang_code)
        with st.expander("Phoneme Tokens", expanded=True):
            st.code(phonemes)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS. The handler code runs at import time but `st.button.return_value = False` in conftest means neither `generate_clicked` nor `tokenize_clicked` triggers.

- [ ] **Step 3: Lint, format, typecheck**

Run: `uv run ruff check . && uv run ruff format . && uv run ty check`
Expected: Clean (ty may have existing warnings unrelated to these changes).

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add st.status progress, phoneme collection, and Tokenize button"
```

### Task 6: Final verification

- [ ] **Step 1: Run full test suite one final time**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 2: Verify lint and format are clean**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean.

- [ ] **Step 3: Review git log**

Run: `git log --oneline -4`
Expected: 4 new commits:
1. `refactor: change generate_speech to generator yielding (audio, phonemes) tuples`
2. `feat: add load_tokenizer and tokenize_text for phoneme extraction`
3. `feat: add phoneme token expander to render_output`
4. `feat: add st.status progress, phoneme collection, and Tokenize button`
