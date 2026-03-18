from unittest.mock import MagicMock

import numpy as np
import pytest
import streamlit as st

from streamlit_app import (
    CHAR_LIMIT,
    HISTORY_MAX,
    LANGUAGES,
    MODEL_NAME,
    REPO_ID,
    SAMPLE_RATE,
    SAMPLES,
    add_to_history,
    generate_speech,
    get_voices,
    load_pipeline,
    load_tokenizer,
    render_output,
    tokenize_text,
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

    def test_char_limit(self) -> None:
        assert CHAR_LIMIT == 5000


class TestSamples:
    def test_has_all_language_codes(self) -> None:
        assert set(SAMPLES.keys()) == EXPECTED_CODES

    def test_each_language_has_three_samples(self) -> None:
        for code, samples in SAMPLES.items():
            assert len(samples) == 3, f"Language '{code}' should have 3 samples"

    def test_samples_are_nonempty_strings(self) -> None:
        for code, samples in SAMPLES.items():
            for sample in samples:
                assert isinstance(sample, str) and len(sample) > 0, (
                    f"Empty or non-string sample in '{code}'"
                )

    def test_samples_within_char_limit(self) -> None:
        for code, samples in SAMPLES.items():
            for sample in samples:
                assert len(sample) <= CHAR_LIMIT, f"Sample in '{code}' exceeds limit"


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

    def test_yields_multiple_chunks(self) -> None:
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

    def _reset_st_mocks(self) -> None:
        st.audio.reset_mock()  # type: ignore[union-attribute]
        st.download_button.reset_mock()  # type: ignore[union-attribute]
        st.markdown.reset_mock()  # type: ignore[union-attribute]
        st.metric.reset_mock()  # type: ignore[union-attribute]
        st.expander.reset_mock()  # type: ignore[union-attribute]
        st.code.reset_mock()  # type: ignore[union-attribute]

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
