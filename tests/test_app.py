from unittest.mock import MagicMock

import numpy as np
import pytest
import streamlit as st

from streamlit_app import (
    CHAR_LIMIT,
    HISTORY_MAX,
    LANGUAGES,
    LONG_SAMPLES,
    MODEL_NAME,
    PRONUNCIATION_TIPS,
    REPO_ID,
    SAMPLE_RATE,
    SAMPLES,
    _wav_bytes,
    add_to_history,
    generate_speech,
    get_voices,
    load_pipeline,
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
        assert MODEL_NAME == "Kokoro-82M-bf16"

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 24000

    def test_repo_id(self) -> None:
        assert REPO_ID == "mlx-community/Kokoro-82M-bf16"

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

    def test_samples_keys_match_language_values(self) -> None:
        assert set(SAMPLES.keys()) == set(LANGUAGES.values())

    def test_samples_have_no_leading_trailing_whitespace(self) -> None:
        for code, samples in SAMPLES.items():
            for sample in samples:
                assert sample == sample.strip(), (
                    f"Sample in '{code}' has leading/trailing whitespace"
                )


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
        pipeline = load_pipeline()
        assert pipeline is not None

    def test_called_with_repo_id(self) -> None:
        from mlx_audio.tts.utils import load_model

        load_pipeline()
        load_model.assert_called_with(REPO_ID)  # type: ignore[union-attribute]


class TestTokenizeText:
    def _mock_g2p(self, phonemes: str) -> MagicMock:
        from misaki import en

        mock_g2p = MagicMock(return_value=(phonemes, None))
        en.G2P.return_value = mock_g2p  # type: ignore[union-attribute]
        return mock_g2p

    def test_returns_phonemes(self) -> None:
        self._mock_g2p("hɛlˈoʊ wˈɜːld")

        result = tokenize_text("hello world", "a")

        assert result == "hɛlˈoʊ wˈɜːld"

    def test_single_word(self) -> None:
        self._mock_g2p("hɛlˈoʊ")

        result = tokenize_text("hello", "a")

        assert result == "hɛlˈoʊ"

    def test_returns_empty_for_empty_phonemes(self) -> None:
        self._mock_g2p("")

        result = tokenize_text("", "a")

        assert result == ""

    def test_returns_empty_for_none_phonemes(self) -> None:
        from misaki import en

        en.G2P.return_value = MagicMock(return_value=(None, None))  # type: ignore[union-attribute]

        result = tokenize_text("", "a")

        assert result == ""

    def test_british_english_uses_british_g2p(self) -> None:
        self._mock_g2p("hɛlˈəʊ")

        tokenize_text("hello", "b")

        from misaki import en

        call_kwargs = en.G2P.call_args[1]  # type: ignore[union-attribute]
        assert call_kwargs["british"] is True

    def test_japanese_uses_ja_g2p(self) -> None:
        from misaki import ja

        ja.JAG2P.return_value = MagicMock(return_value=("konniʧiwa", None))  # type: ignore[union-attribute]

        result = tokenize_text("こんにちは", "j")

        assert result == "konniʧiwa"
        ja.JAG2P.assert_called_once()  # type: ignore[union-attribute]

    def test_spanish_uses_espeak_g2p(self) -> None:
        from misaki import espeak

        espeak.EspeakG2P.return_value = MagicMock(return_value=("ola", None))  # type: ignore[union-attribute]

        result = tokenize_text("hola", "e")

        assert result == "ola"
        espeak.EspeakG2P.assert_called_with(language="es")  # type: ignore[union-attribute]


class TestGenerateSpeech:
    def _mock_model(
        self, *, audio_length: int = 48000, phonemes: str = "hɛlˈoʊ"
    ) -> MagicMock:
        model = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.random.randn(audio_length).astype(np.float32)
        chunk.phonemes = phonemes
        model.generate.return_value = [chunk]
        return model

    def test_yields_audio_and_phonemes(self) -> None:
        model = self._mock_model()

        results = list(generate_speech("hello", "af_heart", model, lang_code="a"))

        assert len(results) == 1
        audio, phonemes = results[0]
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (48000,)
        assert phonemes == "hɛlˈoʊ"

    def test_calls_model_generate_with_correct_args(self) -> None:
        model = self._mock_model()

        list(generate_speech("test text", "af_heart", model, speed=1.5, lang_code="b"))

        model.generate.assert_called_once_with(
            text="test text", voice="af_heart", speed=1.5, lang_code="b"
        )

    def test_default_speed_and_lang_code(self) -> None:
        model = self._mock_model()

        list(generate_speech("test", "af_heart", model))

        model.generate.assert_called_once_with(
            text="test", voice="af_heart", speed=1.0, lang_code="a"
        )

    def test_yields_multiple_chunks(self) -> None:
        model = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = np.ones(100, dtype=np.float32)
        chunk1.phonemes = "wˈʌn"
        chunk2 = MagicMock()
        chunk2.audio = np.zeros(200, dtype=np.float32)
        chunk2.phonemes = "tˈuː"
        model.generate.return_value = [chunk1, chunk2]

        results = list(generate_speech("long text", "af_heart", model, lang_code="a"))

        assert len(results) == 2
        assert results[0][0].shape == (100,)
        assert results[1][0].shape == (200,)
        assert results[0][1] == "wˈʌn"
        assert results[1][1] == "tˈuː"

    def test_output_is_float32(self) -> None:
        model = self._mock_model()

        results = list(generate_speech("test", "af_heart", model, lang_code="a"))

        assert results[0][0].dtype == np.float32

    def test_raises_on_empty_chunks(self) -> None:
        model = MagicMock()
        model.generate.return_value = []

        with pytest.raises(ValueError, match="No audio generated"):
            list(generate_speech("test", "af_heart", model, lang_code="a"))

    def test_skips_chunks_with_none_audio(self) -> None:
        model = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = None
        chunk1.phonemes = "skipped"
        chunk2 = MagicMock()
        chunk2.audio = np.ones(100, dtype=np.float32)
        chunk2.phonemes = "kˈɛpt"
        model.generate.return_value = [chunk1, chunk2]

        results = list(generate_speech("test", "af_heart", model, lang_code="a"))

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


class TestWavBytes:
    def test_returns_bytes(self) -> None:
        audio = np.ones(24000, dtype=np.float32)
        result = _wav_bytes(audio)
        assert isinstance(result, bytes)

    def test_returns_valid_wav_header(self) -> None:
        audio = np.ones(24000, dtype=np.float32)
        result = _wav_bytes(audio)
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_nonempty_output(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = _wav_bytes(audio)
        assert len(result) > 44  # WAV header is 44 bytes

    def test_sample_rate_in_header(self) -> None:
        import struct

        audio = np.ones(24000, dtype=np.float32)
        result = _wav_bytes(audio)
        rate = struct.unpack_from("<I", result, 24)[0]
        assert rate == SAMPLE_RATE


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


class TestPronunciationTips:
    def test_is_nonempty_string(self) -> None:
        assert isinstance(PRONUNCIATION_TIPS, str) and len(PRONUNCIATION_TIPS) > 0

    def test_contains_custom_pronunciation_syntax(self) -> None:
        assert "[word](/phonemes/)" in PRONUNCIATION_TIPS

    def test_contains_intonation_info(self) -> None:
        assert "Intonation" in PRONUNCIATION_TIPS

    def test_contains_stress_adjustment(self) -> None:
        assert "[word](-1)" in PRONUNCIATION_TIPS
        assert "[word](+1)" in PRONUNCIATION_TIPS

    def test_no_leading_trailing_whitespace(self) -> None:
        assert PRONUNCIATION_TIPS == PRONUNCIATION_TIPS.strip()


class TestLongSamples:
    def test_has_all_language_codes(self) -> None:
        assert set(LONG_SAMPLES.keys()) == set(LANGUAGES.values())

    def test_each_value_is_nonempty_string(self) -> None:
        for code, sample in LONG_SAMPLES.items():
            assert isinstance(sample, str) and len(sample) > 0, (
                f"Empty or non-string sample in '{code}'"
            )

    def test_values_are_strings_not_lists(self) -> None:
        for code, sample in LONG_SAMPLES.items():
            assert isinstance(sample, str), (
                f"LONG_SAMPLES['{code}'] should be a string, not {type(sample)}"
            )

    def test_within_char_limit(self) -> None:
        for code, sample in LONG_SAMPLES.items():
            assert len(sample) <= CHAR_LIMIT, (
                f"Long sample in '{code}' exceeds {CHAR_LIMIT} chars"
            )

    def test_minimum_length(self) -> None:
        for code, sample in LONG_SAMPLES.items():
            assert len(sample) >= 300, (
                f"Long sample in '{code}' is under 300 chars ({len(sample)})"
            )

    def test_no_leading_trailing_whitespace(self) -> None:
        for code, sample in LONG_SAMPLES.items():
            assert sample == sample.strip(), (
                f"Long sample in '{code}' has leading/trailing whitespace"
            )
