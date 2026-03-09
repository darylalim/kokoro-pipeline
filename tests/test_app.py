from unittest.mock import MagicMock

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
        assert len(voices) > 0
        assert all(v[0] == "a" for v in voices)

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
