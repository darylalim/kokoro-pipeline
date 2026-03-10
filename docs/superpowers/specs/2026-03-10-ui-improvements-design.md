# UI Improvements Design

**Date:** 2026-03-10
**Context:** Internal team tool (2–5 people), mixed content lengths, fast iteration with voice comparison, frequent multilingual use across all 9 languages.
**Approach:** Enhanced Single-Page — evolve the existing layout with targeted improvements.

## 1. Text Input & Long-Form Support

- Remove the 300-character hard limit from `st.text_area`.
- Add a live character count displayed below the text area using `st.caption` (raw count, no threshold or warning).
- Increase default text area height from 150px to 200px.
- Keep the existing `st.spinner("Generating speech...")` for progress feedback. No per-chunk progress tracking — `generate_speech` remains unchanged.

## 2. Voice Comparison (Inline Multi-Voice Generate)

- Add a **"Compare Voices"** `st.toggle` placed full-width below the two-column (Language | Voice) row, outside the columns block.
- When enabled, replace the single voice `st.selectbox` in column 2 with a `st.multiselect` (`max_selections=3`). Selecting 1 voice is valid. Selecting 0 voices and clicking Generate shows `st.warning("Select at least one voice.")`.
- When the user changes language while in Compare mode, the multiselect options update to the new language's voices. Streamlit automatically clears selections that are no longer in the options list.
- On Generate, call `generate_speech` once per selected voice sequentially, using the same text and speed.
- Display results stacked vertically. Shared metrics shown once above all results: Model, Input Characters. Then each voice result block includes:
  - Voice name label (`st.subheader` or `st.markdown`)
  - Audio player (`st.audio`)
  - Output Duration and Generation Time (`st.metric` in two columns)
  - Download button with voice name in filename (e.g., `speech_af_heart.wav`)
- When the toggle is off, the UI behaves as it does today: single voice, full 4-metric row (Model, Input Characters, Output Duration, Generation Time), filename `speech.wav`.
- `generate_speech` function signature and behavior are unchanged. Multi-voice looping happens at the call site.

## 3. Audio Persistence with Session State

Session state holds two distinct items:

1. **Current output** (`st.session_state.current_output`) — the result of the most recent Generate click. A list of dicts, one per voice, each containing: `audio` (np.ndarray), `voice` (str), `text` (str), `speed` (float), `duration` (float), `generation_time` (float). Replaced entirely on each new Generate click.

2. **History list** (`st.session_state.history`) — a list of past outputs, appended to on each Generate click (see Section 4). Uses the same schema: each history entry is a list of dicts (same structure as `current_output`).

Re-render the current output (audio players, metrics, download buttons) from `st.session_state.current_output` on every rerun so output survives widget interactions.

## 4. Generation History (Sidebar)

- Use `st.sidebar` to display past generations from the current session, newest first.
- Each history entry shows:
  - Truncated text preview (first ~50 characters)
  - Voice name(s) — comma-separated if multi-voice comparison
  - Audio player(s) for each voice in the entry
- Each entry has a `st.button` ("Load"). Clicking it copies that history entry into `st.session_state.current_output`, re-rendering it in the main output area. This does **not** restore input fields (text, voice, speed) — it only restores the audio output.
- One history entry per Generate click, even if multiple voices were compared. Stored in the same schema as `current_output` (list of dicts).
- Maximum 20 entries. When the cap is reached, the oldest entry is dropped.
- History is session-scoped — clears on page refresh.

Note: at 24 kHz float32, a 30-second clip is ~2.9 MB. With 20 entries and up to 3 voices each, worst-case memory is ~175 MB. Acceptable for 2–5 concurrent users on a dedicated machine.

## 5. Overall Layout

```
┌──────────────┬──────────────────────────────────┐
│   Sidebar    │         Main Area                │
│              │                                  │
│  Generation  │  Title + Description             │
│  History     │                                  │
│  (newest     │  Text Input (no char limit)      │
│   first)     │  Character count (st.caption)    │
│              │                                  │
│  - entry 1   │  ┌─────────┬─────────┐           │
│    [Load]    │  │Language  │Voice    │           │
│  - entry 2   │  │selectbox│selectbox│           │
│    [Load]    │  │         │or multi │           │
│  - ...       │  └─────────┴─────────┘           │
│              │  [Compare Voices toggle]          │
│              │                                  │
│              │  Speed Slider                    │
│              │                                  │
│              │  [Generate]                      │
│              │                                  │
│              │  Output (persisted via state):   │
│              │  Single mode:                    │
│              │   - Audio player                 │
│              │   - 4 metrics in row             │
│              │   - Download (speech.wav)        │
│              │  Compare mode:                   │
│              │   - Model + Input Chars (once)   │
│              │   - Per voice:                   │
│              │     - Voice label                │
│              │     - Audio player               │
│              │     - Duration + Gen Time        │
│              │     - Download (speech_voice.wav) │
└──────────────┴──────────────────────────────────┘
```

## What Stays the Same

- Core flow: type text, pick language/voice, set speed, generate.
- Single-file app (`streamlit_app.py`).
- `generate_speech` function signature and behavior.
- Caching strategy (`@st.cache_resource` for pipeline, `@st.cache_data` for voices).
- All 9 languages and dynamic voice discovery from HuggingFace Hub.
- WAV export format and sample rate (24000 Hz).
- Progress feedback: `st.spinner` (unchanged).

## What Changes

| Area | Current | Proposed |
|------|---------|----------|
| Text limit | 300 chars hard | No limit, raw character count |
| Text area height | 150px | 200px |
| Voice selection | Single voice | Single or multi-voice compare (max 3) |
| Output persistence | Lost on rerun | Stored in session state |
| History | None | Sidebar, newest first (max 20) |
| Download filename | `speech.wav` | `speech.wav` or `speech_{voice}.wav` in compare mode |
| Metrics (compare) | N/A | Model + Input Chars shared; Duration + Gen Time per voice |

## Post-Implementation

- Update `CLAUDE.md` to reflect: removed character limit, session state usage, sidebar history, compare voices feature.
- Update tests: mock `st.multiselect`, `st.sidebar`, `st.toggle`, `st.session_state`. Add tests for multi-voice generation flow and history management.
