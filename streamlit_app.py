import io
import os
import time
from collections.abc import Generator

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from huggingface_hub import list_repo_tree
from kokoro import KPipeline

MODEL_NAME = "Kokoro-82M"
SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"
HISTORY_MAX = 20

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


@st.cache_data(ttl=3600)
def get_voices(lang_code: str) -> list[str]:
    entries = list_repo_tree(REPO_ID, path_in_repo="voices")
    voices = []
    for entry in entries:
        name = getattr(entry, "rfilename", None)
        if name is not None and name.endswith(".pt") and name.startswith("voices/"):
            voice = name.removeprefix("voices/").removesuffix(".pt")
            if len(voice) >= 2 and voice[0] == lang_code:
                voices.append(voice)
    return sorted(voices)


@st.cache_resource
def load_pipeline(lang_code: str) -> KPipeline:
    return KPipeline(lang_code=lang_code, repo_id=REPO_ID)


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


def add_to_history(
    history: list[list[dict[str, object]]],
    entry: list[dict[str, object]],
    max_entries: int = HISTORY_MAX,
) -> None:
    history.insert(0, entry)
    if len(history) > max_entries:
        history.pop()


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
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, SAMPLE_RATE, audio)
            st.download_button(
                label=f"Download {result['voice']}",
                data=wav_buffer.getvalue(),
                file_name=f"speech_{result['voice']}.wav",
                mime="audio/wav",
                key=f"download_{result['voice']}",
            )
        with st.expander("Phoneme Tokens"):
            st.code(results[0].get("phonemes", ""))
    else:
        result = results[0]
        audio = np.asarray(result["audio"])
        st.audio(audio, sample_rate=SAMPLE_RATE)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", MODEL_NAME)
        col2.metric("Input Characters", len(str(result["text"])))
        col3.metric("Output Duration", f"{result['duration']:.2f}s")
        col4.metric("Generation Time", f"{result['generation_time']}s")
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, SAMPLE_RATE, audio)
        st.download_button(
            label="Download Audio",
            data=wav_buffer.getvalue(),
            file_name="speech.wav",
            mime="audio/wav",
        )
        with st.expander("Phoneme Tokens"):
            st.code(result.get("phonemes", ""))


if "current_output" not in st.session_state:
    st.session_state["current_output"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []

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
)
st.caption(f"{len(text_input)} characters")

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
    pipeline = load_pipeline(lang_code)

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
                        if phonemes:
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

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])
