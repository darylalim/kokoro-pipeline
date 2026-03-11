from unittest.mock import MagicMock

import numpy as np
import pytest
import streamlit as st

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
    render_output,
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

    def test_repo_id(self) -> None:
        assert REPO_ID == "hexgrad/Kokoro-82M"

    def test_history_max(self) -> None:
        assert HISTORY_MAX == 20


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

    def test_returns_correct_voices(self) -> None:
        voices = get_voices("a")
        assert voices == ["af_bella", "af_heart", "am_adam"]

    def test_skips_entries_without_rfilename(self) -> None:
        from huggingface_hub import list_repo_tree

        original = list_repo_tree.return_value  # type: ignore[union-attribute]
        folder = MagicMock(spec=[])  # no rfilename attribute
        list_repo_tree.return_value = [folder] + list(original)  # type: ignore[union-attribute]
        try:
            voices = get_voices("a")
            assert "af_heart" in voices
        finally:
            list_repo_tree.return_value = original  # type: ignore[union-attribute]


class TestLoadPipeline:
    def test_returns_pipeline(self) -> None:
        pipeline = load_pipeline("a")
        assert pipeline is not None

    def test_called_with_lang_code(self) -> None:
        from kokoro import KPipeline

        load_pipeline("a")
        KPipeline.assert_called_with(lang_code="a", repo_id=REPO_ID)  # type: ignore[union-attribute]


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

    def test_raises_on_empty_chunks(self) -> None:
        pipeline = MagicMock()
        pipeline.return_value = []

        with pytest.raises(ValueError, match="No audio generated"):
            generate_speech("test", "af_heart", pipeline)


class TestAddToHistory:
    def test_adds_entry_to_empty_history(self) -> None:
        history: list[list[dict[str, object]]] = []
        entry: list[dict[str, object]] = [{"voice": "af_heart", "text": "hello"}]
        add_to_history(history, entry)
        assert len(history) == 1
        assert history[0] is entry

    def test_newest_first(self) -> None:
        old: list[dict[str, object]] = [{"voice": "af_bella"}]
        history: list[list[dict[str, object]]] = [old]
        new: list[dict[str, object]] = [{"voice": "af_heart"}]
        add_to_history(history, new)
        assert history[0] is new
        assert history[1] is old

    def test_caps_at_max_entries(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(20)
        ]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new, max_entries=20)
        assert len(history) == 20

    def test_drops_oldest_when_full(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(20)
        ]
        oldest = history[-1]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new, max_entries=20)
        assert oldest not in history
        assert history[0] is new

    def test_custom_max_entries(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(3)
        ]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new, max_entries=3)
        assert len(history) == 3
        assert history[0] is new

    def test_default_max_uses_history_max(self) -> None:
        history: list[list[dict[str, object]]] = [
            [{"voice": f"v{i}"}] for i in range(HISTORY_MAX)
        ]
        new: list[dict[str, object]] = [{"voice": "new"}]
        add_to_history(history, new)
        assert len(history) == HISTORY_MAX
        assert history[0] is new


class TestRenderOutput:
    @staticmethod
    def _make_result(voice: str = "af_heart", text: str = "hello") -> dict[str, object]:
        return {
            "audio": np.ones(24000, dtype=np.float32),
            "voice": voice,
            "text": text,
            "speed": 1.0,
            "duration": 1.0,
            "generation_time": 0.5,
        }

    def _reset_st_mocks(self) -> None:
        st.audio.reset_mock()  # type: ignore[union-attribute]
        st.download_button.reset_mock()  # type: ignore[union-attribute]
        st.markdown.reset_mock()  # type: ignore[union-attribute]
        st.metric.reset_mock()  # type: ignore[union-attribute]

    def test_empty_results_returns_early(self) -> None:
        self._reset_st_mocks()
        render_output([])
        st.audio.assert_not_called()  # type: ignore[union-attribute]
        st.download_button.assert_not_called()  # type: ignore[union-attribute]

    def test_single_result_renders_audio(self) -> None:
        self._reset_st_mocks()
        render_output([self._make_result()])
        st.audio.assert_called_once()  # type: ignore[union-attribute]

    def test_single_result_download_filename(self) -> None:
        self._reset_st_mocks()
        render_output([self._make_result()])
        st.download_button.assert_called_once()  # type: ignore[union-attribute]
        call_kwargs = st.download_button.call_args[1]  # type: ignore[union-attribute]
        assert call_kwargs["file_name"] == "speech.wav"

    def test_single_result_download_label(self) -> None:
        self._reset_st_mocks()
        render_output([self._make_result()])
        call_kwargs = st.download_button.call_args[1]  # type: ignore[union-attribute]
        assert call_kwargs["label"] == "Download Audio"

    def test_compare_renders_audio_per_voice(self) -> None:
        self._reset_st_mocks()
        results = [self._make_result("af_heart"), self._make_result("af_bella")]
        render_output(results)
        assert st.audio.call_count == 2  # type: ignore[union-attribute]

    def test_compare_download_filenames(self) -> None:
        self._reset_st_mocks()
        results = [self._make_result("af_heart"), self._make_result("af_bella")]
        render_output(results)
        assert st.download_button.call_count == 2  # type: ignore[union-attribute]
        filenames = [
            call[1]["file_name"]
            for call in st.download_button.call_args_list  # type: ignore[union-attribute]
        ]
        assert "speech_af_heart.wav" in filenames
        assert "speech_af_bella.wav" in filenames

    def test_compare_voice_labels(self) -> None:
        self._reset_st_mocks()
        results = [self._make_result("af_heart"), self._make_result("af_bella")]
        render_output(results)
        markdown_calls = [
            call[0][0]
            for call in st.markdown.call_args_list  # type: ignore[union-attribute]
        ]
        assert "### af_heart" in markdown_calls
        assert "### af_bella" in markdown_calls

    def test_compare_download_labels_include_voice(self) -> None:
        self._reset_st_mocks()
        results = [self._make_result("af_heart"), self._make_result("am_adam")]
        render_output(results)
        labels = [
            call[1]["label"]
            for call in st.download_button.call_args_list  # type: ignore[union-attribute]
        ]
        assert "Download af_heart" in labels
        assert "Download am_adam" in labels
