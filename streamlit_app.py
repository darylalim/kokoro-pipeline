import io
import time
from collections.abc import Generator
from typing import Any

import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from huggingface_hub import list_repo_tree
from mlx_audio.tts.utils import load_model

MODEL_NAME = "Kokoro-82M-bf16"
SAMPLE_RATE = 24000
REPO_ID = "mlx-community/Kokoro-82M-bf16"
CHAR_LIMIT = 5000
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


@st.cache_data(ttl=3600)
def get_voices(lang_code: str) -> list[str]:
    voices = [
        voice
        for entry in list_repo_tree(REPO_ID, path_in_repo="voices")
        if (name := getattr(entry, "rfilename", ""))
        and name.startswith("voices/")
        and name.endswith(".safetensors")
        and len(voice := name.removeprefix("voices/").removesuffix(".safetensors")) >= 2
        and voice[0] == lang_code
    ]
    females = sorted(v for v in voices if v[1] == "f")
    males = sorted(v for v in voices if v[1] == "m")
    return females + males


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


_VOICE_GENDERS: dict[str, str] = {"f": "female", "m": "male"}


def _format_voice(voice: str) -> str:
    if "_" not in voice:
        return voice
    name = voice.split("_", 1)[1].replace("_", " ").title()
    gender = _VOICE_GENDERS.get(voice[1], "")
    return f"{name} ({gender})" if gender else name


def _filter_voices_by_gender(voices: list[str], gender_code: str | None) -> list[str]:
    if gender_code is None:
        return voices
    return [v for v in voices if len(v) >= 2 and v[1] == gender_code]


def generate_speech(
    text: str,
    voice: str,
    model: Any,
    speed: float = 1.0,
    lang_code: str = "a",
) -> Generator[np.ndarray, None, None]:
    generated = False
    for result in model.generate(
        text=text, voice=voice, speed=speed, lang_code=lang_code
    ):
        if result.audio is not None:
            generated = True
            yield np.array(result.audio, dtype=np.float32)
    if not generated:
        raise ValueError("No audio generated. Check your input text.")


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio)
    return buf.getvalue()


def _validate_input(text: str) -> str | None:
    if not text.strip():
        return "Enter text."
    return None


def _reset_selected_voices() -> None:
    st.session_state["selected_voices"] = []


def render_output(results: list[dict[str, object]]) -> None:
    if not results:
        return
    show_heading = len(results) > 1
    for result in results:
        voice = result["voice"]
        audio = np.asarray(result["audio"])
        if show_heading:
            st.markdown(f"### {voice}")
        st.audio(audio, sample_rate=SAMPLE_RATE)
        st.download_button(
            label="Download",
            data=_wav_bytes(audio),
            file_name=f"speech_{voice}.wav",
            mime="audio/wav",
            key=f"download_{voice}",
        )
    with st.expander("Phoneme Tokens"):
        st.code(results[0]["phonemes"])


st.session_state.setdefault("current_output", None)

st.title("Kokoro Pipeline")

text_input = st.text_area(
    label="Text",
    placeholder="Enter text to generate speech...",
    height=200,
    max_chars=CHAR_LIMIT,
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
        on_change=_reset_selected_voices,
    )

lang_code = LANGUAGES[language]

with gender_col:
    gender_label = st.selectbox(
        "Gender",
        options=list(GENDERS.keys()),
        label_visibility="collapsed",
        key="gender",
        on_change=_reset_selected_voices,
    )

gender_code = GENDERS[gender_label]
voices = _filter_voices_by_gender(get_voices(lang_code), gender_code)

with voice_col:
    selected_voices = st.multiselect(
        "Voices",
        options=voices,
        max_selections=3,
        format_func=_format_voice,
        label_visibility="collapsed",
        help="Select up to 3 voices.",
        key="selected_voices",
    )

speed = st.slider(
    "Speed",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Speech rate multiplier. 1.0 is normal speed.",
)

with st.spinner("Loading model..."):
    pipeline = load_pipeline()

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    generate_clicked = st.button("Generate", type="primary")
with btn_col2:
    tokenize_clicked = st.button("Tokenize")

if generate_clicked:
    warning = _validate_input(text_input)
    if warning:
        st.warning(warning)
    elif not selected_voices:
        st.warning("Select at least one voice.")
    else:
        try:
            results = []
            phonemes = tokenize_text(text_input, lang_code)
            for v in selected_voices:
                start = time.perf_counter()
                with st.status(f"Generating {v}...", expanded=True) as status:
                    audio_chunks = []
                    for i, audio_chunk in enumerate(
                        generate_speech(
                            text_input, v, pipeline, speed=speed, lang_code=lang_code
                        ),
                        1,
                    ):
                        audio_chunks.append(audio_chunk)
                        st.write(f"Chunk {i}...")
                    status.update(label=f"{v} complete!", state="complete")
                gen_time = round(time.perf_counter() - start, 2)
                audio_array = np.concatenate(audio_chunks)
                results.append(
                    {
                        "audio": audio_array,
                        "voice": v,
                        "text": text_input,
                        "duration": len(audio_array) / SAMPLE_RATE,
                        "generation_time": gen_time,
                        "phonemes": phonemes,
                    }
                )
            st.session_state["current_output"] = results
            st.rerun()
        except Exception as e:
            st.exception(e)

if tokenize_clicked:
    warning = _validate_input(text_input)
    if warning:
        st.warning(warning)
    else:
        phonemes = tokenize_text(text_input, lang_code)
        with st.expander("Phoneme Tokens", expanded=True):
            st.code(phonemes)

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])

with st.expander("Tips"):
    st.markdown(PRONUNCIATION_TIPS)
