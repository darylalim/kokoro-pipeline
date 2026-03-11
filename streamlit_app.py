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


def generate_speech(
    text: str,
    voice: str,
    pipeline: KPipeline,
    speed: float = 1.0,
) -> np.ndarray:
    chunks = list(pipeline(text, voice=voice, speed=speed))
    if not chunks:
        raise ValueError("No audio generated. Check your input text.")
    # ty cannot infer audio type from KPipeline generator output
    audio = np.concatenate([c.audio for c in chunks])  # type: ignore[no-matching-overload]
    return audio.astype(np.float32)


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
            st.audio(result["audio"], sample_rate=SAMPLE_RATE)
            mc1, mc2 = st.columns(2)
            mc1.metric("Output Duration", f"{result['duration']:.2f}s")
            mc2.metric("Generation Time", f"{result['generation_time']}s")
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, SAMPLE_RATE, result["audio"])  # type: ignore[arg-type]
            st.download_button(
                label=f"Download {result['voice']}",
                data=wav_buffer.getvalue(),
                file_name=f"speech_{result['voice']}.wav",
                mime="audio/wav",
                key=f"download_{result['voice']}",
            )
    else:
        result = results[0]
        st.audio(result["audio"], sample_rate=SAMPLE_RATE)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", MODEL_NAME)
        col2.metric("Input Characters", len(str(result["text"])))
        col3.metric("Output Duration", f"{result['duration']:.2f}s")
        col4.metric("Generation Time", f"{result['generation_time']}s")
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, SAMPLE_RATE, result["audio"])  # type: ignore[arg-type]
        st.download_button(
            label="Download Audio",
            data=wav_buffer.getvalue(),
            file_name="speech.wav",
            mime="audio/wav",
        )



if "current_output" not in st.session_state:
    st.session_state["current_output"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []

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
        help="The second letter indicates gender: 'f' for female, 'm' for male.",
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
            results = []
            with st.spinner("Generating speech..."):
                start = time.perf_counter()
                audio_array = generate_speech(text_input, voice, pipeline, speed=speed)
                gen_time = round(time.perf_counter() - start, 2)
                results.append({
                    "audio": audio_array,
                    "voice": voice,
                    "text": text_input,
                    "speed": speed,
                    "duration": len(audio_array) / SAMPLE_RATE,
                    "generation_time": gen_time,
                })
            st.session_state["current_output"] = results
            add_to_history(st.session_state["history"], results)
            st.rerun()
        except Exception as e:
            st.exception(e)
    else:
        st.warning("Enter text.")

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])
