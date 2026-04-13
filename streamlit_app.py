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
HISTORY_MAX = 20
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


def add_to_history(
    history: list[list[dict[str, object]]],
    entry: list[dict[str, object]],
    max_entries: int = HISTORY_MAX,
) -> None:
    history.insert(0, entry)
    if len(history) > max_entries:
        history.pop()


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio)
    return buf.getvalue()


def render_output(results: list[dict[str, object]]) -> None:
    if not results:
        return
    if len(results) > 1:
        col1, col2 = st.columns(2)
        col1.metric("Model", MODEL_NAME)
        col2.metric("Input Characters", len(str(results[0]["text"])))
        for result in results:
            st.markdown(f"### {result['voice']}")
            audio = np.asarray(result["audio"])
            st.audio(audio, sample_rate=SAMPLE_RATE)
            mc1, mc2 = st.columns(2)
            mc1.metric("Output Duration", f"{result['duration']:.2f}s")
            mc2.metric("Generation Time", f"{result['generation_time']}s")
            st.download_button(
                label=f"Download {result['voice']}",
                data=_wav_bytes(audio),
                file_name=f"speech_{result['voice']}.wav",
                mime="audio/wav",
                key=f"download_{result['voice']}",
            )
    else:
        result = results[0]
        audio = np.asarray(result["audio"])
        st.audio(audio, sample_rate=SAMPLE_RATE)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", MODEL_NAME)
        col2.metric("Input Characters", len(str(result["text"])))
        col3.metric("Output Duration", f"{result['duration']:.2f}s")
        col4.metric("Generation Time", f"{result['generation_time']}s")
        st.download_button(
            label="Download Audio",
            data=_wav_bytes(audio),
            file_name="speech.wav",
            mime="audio/wav",
        )
    with st.expander("Phoneme Tokens"):
        st.code(results[0].get("phonemes", ""))


st.session_state.setdefault("current_output", None)
st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("Generation History")
    history = st.session_state["history"]
    if not history:
        st.caption("No generations yet.")
    for i, entry in enumerate(history):
        text = str(entry[0]["text"])
        text_preview = text[:50] + ("..." if len(text) > 50 else "")
        voice_names = ", ".join(str(r["voice"]) for r in entry)
        st.markdown(f"**{text_preview}**")
        st.caption(voice_names)
        for result in entry:
            st.audio(np.asarray(result["audio"]), sample_rate=SAMPLE_RATE)
        if st.button("Load", key=f"load_{i}"):
            st.session_state["current_output"] = entry

st.title("Text to Speech Pipeline")
st.write("Generate multilingual speech with Kokoro.")

st.subheader("Text")
text_input = st.text_area(
    "Text",
    placeholder="Enter text...",
    height=200,
    help="Enter text for speech generation.",
    key="text_input",
)
if len(text_input) > CHAR_LIMIT:
    st.caption(
        f'<span style="color: red">{len(text_input)} / {CHAR_LIMIT} characters</span>',
        unsafe_allow_html=True,
    )
else:
    st.caption(f"{len(text_input)} / {CHAR_LIMIT} characters")

with st.expander("Pronunciation Tips"):
    st.markdown(PRONUNCIATION_TIPS)

st.subheader("Voice")
compare_mode = st.session_state.get("compare_mode", False)
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
    if compare_mode:
        selected_voices = st.multiselect(
            "Voices",
            options=voices,
            max_selections=3,
            help="Select up to 3 voices to compare.",
        )
    else:
        voice = st.selectbox(
            "Voice",
            options=voices,
            help="The second letter indicates gender: 'f' for female, 'm' for male.",
        )
        selected_voices = [voice]

st.toggle("Compare Voices", key="compare_mode")

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
    pipeline = load_pipeline()

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    generate_clicked = st.button("Generate", type="primary")
with btn_col2:
    tokenize_clicked = st.button("Tokenize")

if generate_clicked:
    if not text_input.strip():
        st.warning("Enter text.")
    elif len(text_input) > CHAR_LIMIT:
        st.warning(f"Text exceeds {CHAR_LIMIT} character limit.")
    elif compare_mode and not selected_voices:
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
                        "speed": speed,
                        "duration": len(audio_array) / SAMPLE_RATE,
                        "generation_time": gen_time,
                        "phonemes": phonemes,
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
    elif len(text_input) > CHAR_LIMIT:
        st.warning(f"Text exceeds {CHAR_LIMIT} character limit.")
    else:
        phonemes = tokenize_text(text_input, lang_code)
        with st.expander("Phoneme Tokens", expanded=True):
            st.code(phonemes)

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])
