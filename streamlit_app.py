from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import streamlit as st
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx_audio.tts.utils import load_model

from voice_grades import VOICE_GRADES, _grade_rank


class VoiceResult(TypedDict):
    audio: np.ndarray
    voice: str
    phonemes: str


SAMPLE_RATE = 24000
REPO_ID = "mlx-community/Kokoro-82M-bf16"
PRONUNCIATION_TIPS = """\
**Custom pronunciation:** Use `[word](/phonemes/)` syntax, e.g. `[Kokoro](/kˈOkəɹO/)`

**Intonation:** Adjust with punctuation `;` `:` `,` `.` `!` `?` `—` `…` `"` `(` `)` `"` `"`

**Lower stress:** `[word](-1)` or `[word](-2)`

**Raise stress:** `[word](+1)` or `[word](+2)` (works best on less-stressed, usually short words)\
"""

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

ESPEAK_LANGUAGES: dict[str, str] = {
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

GENDERS: dict[str, str | None] = {
    "All": None,
    "Female": "f",
    "Male": "m",
}

_GENDER_LABELS = {code: label.lower() for label, code in GENDERS.items() if code}

DEFAULT_VOICE_BY_LANG: dict[str, str] = {
    "a": "af_heart",
}

@st.cache_resource
def ensure_repo_downloaded() -> str:
    try:
        return snapshot_download(REPO_ID, local_files_only=True)
    except LocalEntryNotFoundError:
        with st.spinner("Downloading Kokoro model and voices (one-time, ~160 MB)..."):
            return snapshot_download(REPO_ID)


@st.cache_data
def get_voices(lang_code: str) -> list[str]:
    voices_dir = Path(ensure_repo_downloaded()) / "voices"
    voices = [
        p.stem
        for p in voices_dir.iterdir()
        if p.suffix == ".safetensors" and len(p.stem) >= 2 and p.stem[0] == lang_code
    ]
    return sorted(voices, key=lambda v: (_grade_rank(v), v))


@st.cache_resource
def load_pipeline() -> Any:
    return load_model(REPO_ID)  # type: ignore[arg-type]


def _create_g2p(lang_code: str) -> Any:
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
def load_tokenizer(lang_code: str) -> Any:
    return _create_g2p(lang_code)


def tokenize_text(text: str, lang_code: str) -> str:
    phonemes, _ = load_tokenizer(lang_code)(text)
    return phonemes or ""


def _format_voice(voice: str) -> str:
    if "_" not in voice:
        return voice
    name = voice.split("_", 1)[1].replace("_", " ").title()
    gender = _GENDER_LABELS.get(voice[1], "")
    grade = VOICE_GRADES.get(voice, "")
    base = f"{name} ({gender})" if gender else name
    return f"{base} — {grade}" if grade else base


def _filter_voices_by_gender(voices: list[str], gender_code: str | None) -> list[str]:
    if gender_code is None:
        return voices
    return [v for v in voices if v[1] == gender_code]


def _default_voice(lang_code: str, gender_code: str | None) -> str | None:
    explicit = DEFAULT_VOICE_BY_LANG.get(lang_code)
    if explicit is not None and (gender_code is None or explicit[1] == gender_code):
        return explicit
    voices = _filter_voices_by_gender(get_voices(lang_code), gender_code)
    return min(voices, key=_grade_rank) if voices else None


def generate_speech(
    text: str,
    voice: str,
    pipeline: Any,
    speed: float = 1.0,
    lang_code: str = "a",
) -> Generator[np.ndarray, None, None]:
    generated = False
    for result in pipeline.generate(
        text=text, voice=voice, speed=speed, lang_code=lang_code
    ):
        if result.audio is not None:
            generated = True
            yield np.asarray(result.audio, dtype=np.float32)
    if not generated:
        raise ValueError("No audio generated. Check your input text.")


def generate_one(
    text: str,
    voice: str,
    pipeline: Any,
    speed: float,
    lang_code: str,
) -> VoiceResult:
    phonemes = tokenize_text(text, lang_code)
    with st.status(f"Generating {voice}...", expanded=True) as status:
        chunks = []
        for i, chunk in enumerate(
            generate_speech(text, voice, pipeline, speed=speed, lang_code=lang_code), 1
        ):
            chunks.append(chunk)
            st.write(f"Chunk {i}...")
        status.update(label=f"{voice} complete!", state="complete")
    return {"audio": np.concatenate(chunks), "voice": voice, "phonemes": phonemes}


def _reset_selected_voice() -> None:
    lang_code = LANGUAGES[st.session_state["language"]]
    gender_code = GENDERS[st.session_state["gender"]]
    st.session_state["selected_voice"] = _default_voice(lang_code, gender_code)


def _on_language_change() -> None:
    _reset_selected_voice()
    st.session_state["current_output"] = None


def render_phonemes(phonemes: str, *, expanded: bool = False) -> None:
    with st.expander("Phoneme Tokens", expanded=expanded):
        st.code(phonemes)


def render_output(result: VoiceResult | None) -> None:
    if result is None:
        return
    st.audio(result["audio"], sample_rate=SAMPLE_RATE)
    render_phonemes(result["phonemes"])


st.session_state.setdefault("current_output", None)

st.title("Kokoro Pipeline")

try:
    ensure_repo_downloaded()
except Exception:
    st.error("Could not download the Kokoro model. Connect to the internet and reload the page.")
    st.stop()

text_input = st.text_area(
    label="Text",
    placeholder="Enter text to generate speech...",
    height=200,
    key="text_input",
    label_visibility="collapsed",
)

lang_col, gender_col, voice_col = st.columns(3)

with lang_col:
    language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        label_visibility="collapsed",
        key="language",
        on_change=_on_language_change,
    )

lang_code = LANGUAGES[language]

with gender_col:
    gender_label = st.selectbox(
        "Gender",
        options=list(GENDERS.keys()),
        label_visibility="collapsed",
        key="gender",
        on_change=_reset_selected_voice,
    )

voices = _filter_voices_by_gender(get_voices(lang_code), GENDERS[gender_label])

if "selected_voice" not in st.session_state:
    st.session_state["selected_voice"] = _default_voice(lang_code, GENDERS[gender_label])

with voice_col:
    if voices:
        selected_voice = st.selectbox(
            "Voice",
            options=voices,
            index=None,
            format_func=_format_voice,
            placeholder="Select a voice",
            label_visibility="collapsed",
            key="selected_voice",
        )
    else:
        st.info("No voices match this filter.")
        selected_voice = None

speed = st.slider(
    "Speed",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Speech rate multiplier. 1.0 is normal speed.",
    key="speed",
    disabled=not selected_voice,
)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    generate_clicked = st.button("Generate", type="primary")
with btn_col2:
    tokenize_clicked = st.button("Tokenize")

if generate_clicked:
    if not text_input.strip():
        st.warning("Enter text.")
    elif not selected_voice:
        st.warning("Select a voice.")
    else:
        try:
            with st.spinner("Loading model..."):
                pipeline = load_pipeline()
            st.session_state["current_output"] = generate_one(
                text_input, selected_voice, pipeline, speed, lang_code
            )
            st.rerun()
        except Exception as e:
            st.exception(e)

if tokenize_clicked:
    if not text_input.strip():
        st.warning("Enter text.")
    else:
        render_phonemes(tokenize_text(text_input, lang_code), expanded=True)

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])

with st.expander("Tips"):
    st.markdown(PRONUNCIATION_TIPS)
