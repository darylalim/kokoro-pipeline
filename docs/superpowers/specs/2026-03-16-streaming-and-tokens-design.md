# Streaming Progress & Phoneme Token Display

## Overview

Add two features to the Kokoro TTS Streamlit app:

1. **Streaming progress** — real-time feedback during speech generation, showing chunk-by-chunk progress via `st.status`
2. **Phoneme token display** — show the phonetic tokens used for generation, both post-generation (expander) and on-demand (Tokenize button)

## Context

The app currently blocks with a spinner during generation and provides no visibility into phoneme tokenization. The official HF demo (Gradio-based) offers streaming audio and a Tokenize button. Streamlit does not support progressive audio playback, so we use a progress indicator approach instead.

## Design

### 1. `generate_speech` refactor

Change from returning a single `np.ndarray` to a **generator** that yields `(audio_chunk, phonemes)` tuples.

**Before:**

```python
def generate_speech(
    text: str,
    voice: str,
    pipeline: KPipeline,
    speed: float = 1.0,
) -> np.ndarray:
    chunks = list(pipeline(text, voice=voice, speed=speed))
    if not chunks:
        raise ValueError("No audio generated. Check your input text.")
    audio = np.concatenate([c.audio for c in chunks])
    return audio.astype(np.float32)
```

**After:**

```python
def generate_speech(
    text: str,
    voice: str,
    pipeline: KPipeline,
    speed: float = 1.0,
) -> Generator[tuple[np.ndarray, str], None, None]:  # requires: from collections.abc import Generator
    generated = False
    for result in pipeline(text, voice=voice, speed=speed):
        if result.audio is not None:
            generated = True
            yield result.audio.cpu().numpy().astype(np.float32), result.phonemes
    if not generated:
        raise ValueError("No audio generated. Check your input text.")
```

Callers collect chunks, concatenate audio, and join phonemes.

**Note:** `result.audio` is assumed to be a `torch.FloatTensor` as returned by `KModel.Output`. The `.cpu()` call is necessary because tensors may reside on MPS (Apple Silicon). Since `generate_speech` is only called when audio is generated (not in tokenize-only mode), `result.audio` is always a tensor when not `None`.

**Note:** Because `generate_speech` is now a generator, the `ValueError` for empty output is raised only after the generator is fully consumed. Callers that iterate lazily will not see the error until iteration completes. Tests must consume the generator (e.g., `list(generate_speech(...))`) to trigger the error.

### 2. `load_tokenizer` — new cached function

```python
@st.cache_resource
def load_tokenizer(lang_code: str) -> KPipeline:
    return KPipeline(lang_code=lang_code, model=False)
```

A lightweight pipeline with no model weights. Used exclusively by the Tokenize button for fast phoneme extraction.

### 3. `tokenize_text` — new function

```python
def tokenize_text(text: str, lang_code: str) -> str:
    tokenizer = load_tokenizer(lang_code)
    phonemes = []
    for result in tokenizer(text):
        if result.phonemes:
            phonemes.append(result.phonemes)
    return " ".join(phonemes)
```

Returns the joined phoneme string for display. No audio inference is performed.

### 4. Progress feedback during generation

Replace `st.spinner` with `st.status` in the Generate button handler:

```python
with st.status("Generating speech...", expanded=True) as status:
    chunks = []
    for i, (audio_chunk, phonemes) in enumerate(generate_speech(...), 1):
        chunks.append((audio_chunk, phonemes))
        st.write(f"Generated chunk {i}...")
    status.update(label="Generation complete!", state="complete")
```

**Compare mode:** One `st.status` per voice. Each voice gets its own status context that opens, shows chunk progress ("Generating af_heart, chunk 2..."), and collapses when that voice is done. This avoids interleaving messages from multiple voices in a single status container.

```python
for v in selected_voices:
    with st.status(f"Generating {v}...", expanded=True) as status:
        chunks = []
        for i, (audio_chunk, phonemes) in enumerate(generate_speech(...), 1):
            chunks.append((audio_chunk, phonemes))
            st.write(f"Chunk {i}...")
        status.update(label=f"{v} complete!", state="complete")
```

After all statuses complete, audio is concatenated per voice and results are stored in session state. The existing `st.rerun()` call is kept — on rerun, the `st.status` widgets are gone and `render_output` displays the final results cleanly.

### 5. Phoneme token display in `render_output`

Add an expander below the audio output.

**Single mode:** One expander after the audio player, metrics, and download button:

```python
with st.expander("Phoneme Tokens"):
    st.code(result["phonemes"])
```

**Compare mode:** One shared expander at the bottom. Phonemes are determined by `lang_code` + text via the `g2p` module, not by voice — verified in Kokoro's `KPipeline.__call__` where phoneme generation happens before voice pack loading. Since compare mode only compares voices within the same language, phonemes are identical across all results:

```python
with st.expander("Phoneme Tokens"):
    st.code(results[0]["phonemes"])
```

`st.code` provides monospace, copyable display suited for phoneme strings.

### 6. Tokenize button

Placed next to the Generate button using columns:

```python
col1, col2 = st.columns(2)
with col1:
    generate_clicked = st.button("Generate", type="primary")
with col2:
    tokenize_clicked = st.button("Tokenize")
```

When clicked:
- Validates that text is non-empty (uses `st.warning("Enter text.")` for consistency with Generate)
- Calls `tokenize_text(text_input, lang_code)`
- Displays result in `st.expander("Phoneme Tokens")` with `st.code`
- Does not generate audio or affect session history
- Output is ephemeral (not stored in session state) — it disappears on the next rerun, which is acceptable since tokenize-only results are for quick inspection, not persistent reference

### 7. Result dict changes

Each result dict gains a `"phonemes"` key:

```python
results.append({
    "audio": concatenated_audio,
    "voice": v,
    "text": text_input,
    "speed": speed,
    "duration": len(concatenated_audio) / SAMPLE_RATE,
    "generation_time": gen_time,
    "phonemes": all_phonemes,  # new
})
```

Phonemes are persisted in history and restored when a history entry is loaded.

### 8. Timing

`time.perf_counter()` wraps the entire chunk-collection loop (including the `st.status` block) so generation time reflects total inference, not just the last chunk:

```python
start = time.perf_counter()
with st.status(...) as status:
    for i, (audio_chunk, phonemes) in enumerate(generate_speech(...), 1):
        ...
gen_time = round(time.perf_counter() - start, 2)
```

### 9. Test updates

**Conftest changes (`tests/conftest.py`):**
- `st.status` needs to be mocked as a context manager with an `.update` method. `MagicMock` handles `__enter__`/`__exit__` automatically, but `.update` should be explicitly available.
- `st.expander` needs to work as a context manager — same auto-handling by `MagicMock`.
- `st.code` is already available as a `MagicMock` attribute.
- No changes needed for `KPipeline(model=False)` — the existing `MagicMock` for kokoro accepts any kwargs.

**Modified tests:**
- `TestGenerateSpeech` — update all tests for generator behavior. The `_mock_pipeline` helper must return an iterable of objects with `.audio` (as a mock torch tensor with `.cpu().numpy()` chain) and `.phonemes` (string) attributes. Tests iterate the generator and collect `(audio, phonemes)` tuples instead of receiving a direct array.

**New tests:**
- `TestTokenizeText` — verify phoneme extraction returns joined string, handles empty input
- `TestRenderOutput` — verify `st.expander` and `st.code` are called with phoneme data for both single and compare modes

**Unchanged tests:**
- `TestLanguages`, `TestModelConstants`, `TestGetVoices`, `TestLoadPipeline`, `TestAddToHistory` — no changes needed

## Files changed

- `streamlit_app.py` — all feature changes (single file app)
- `tests/test_app.py` — test updates and additions
- `tests/conftest.py` — may need mock updates for `KPipeline(model=False)` if tokenizer tests require it

## No new dependencies

All functionality uses existing `kokoro` and `streamlit` APIs.
