import sys
from unittest.mock import MagicMock

# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = lambda f: f
_st.cache_data = lambda **_kw: lambda f: f
_st.selectbox.side_effect = lambda label, **_kw: {
    "Language": "American English",
    "Gender": "All",
}.get(label, MagicMock())
_st.slider.side_effect = lambda label, **_kw: {
    "Speed": 1.0,
}.get(label, MagicMock())
_st.button.return_value = False
_st.text_area.return_value = ""
_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
_st.multiselect.return_value = []
_st.session_state = {}
sys.modules["streamlit"] = _st

# Mock mlx_audio to prevent model downloads on import
_mlx_audio = MagicMock()
_mlx_audio_tts = MagicMock()
_mlx_audio_tts_utils = MagicMock()
sys.modules["mlx_audio"] = _mlx_audio
sys.modules["mlx_audio.tts"] = _mlx_audio_tts
sys.modules["mlx_audio.tts.utils"] = _mlx_audio_tts_utils

# Mock misaki to prevent espeak-ng dependency in tests
_misaki = MagicMock()
_misaki_en = MagicMock()
_misaki_ja = MagicMock()
_misaki_zh = MagicMock()
_misaki_espeak = MagicMock()
_misaki.en = _misaki_en
_misaki.ja = _misaki_ja
_misaki.zh = _misaki_zh
_misaki.espeak = _misaki_espeak
sys.modules["misaki"] = _misaki
sys.modules["misaki.en"] = _misaki_en
sys.modules["misaki.ja"] = _misaki_ja
sys.modules["misaki.zh"] = _misaki_zh
sys.modules["misaki.espeak"] = _misaki_espeak

# Mock huggingface_hub to prevent network calls on import
_hf_hub = MagicMock()
_hf_hub.list_repo_tree.return_value = [
    MagicMock(rfilename="voices/af_heart.safetensors"),
    MagicMock(rfilename="voices/af_bella.safetensors"),
    MagicMock(rfilename="voices/am_adam.safetensors"),
    MagicMock(rfilename="voices/bf_alice.safetensors"),
    MagicMock(rfilename="voices/bm_daniel.safetensors"),
    MagicMock(rfilename="voices/jf_alpha.safetensors"),
    MagicMock(rfilename="voices/zf_xiaobei.safetensors"),
    MagicMock(rfilename="voices/ef_dora.safetensors"),
    MagicMock(rfilename="voices/ff_siwis.safetensors"),
    MagicMock(rfilename="voices/hf_alpha.safetensors"),
    MagicMock(rfilename="voices/if_sara.safetensors"),
    MagicMock(rfilename="voices/pf_dora.safetensors"),
]
sys.modules["huggingface_hub"] = _hf_hub
