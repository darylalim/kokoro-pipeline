import sys
from unittest.mock import MagicMock

# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = lambda f: f
_st.cache_data = lambda **_kw: lambda f: f
_st.selectbox.side_effect = lambda label, **_kw: {
    "Language": "American English",
    "Voice": "af_heart",
}.get(label, MagicMock())
_st.slider.side_effect = lambda label, **_kw: {
    "Speed": 1.0,
}.get(label, MagicMock())
_st.button.return_value = False
_st.text_area.return_value = ""
_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
_st.toggle.return_value = False
_st.multiselect.return_value = []
_st.session_state = {}
sys.modules["streamlit"] = _st

# Mock kokoro to prevent model downloads on import
_kokoro = MagicMock()
sys.modules["kokoro"] = _kokoro

# Mock huggingface_hub to prevent network calls on import
_hf_hub = MagicMock()
_hf_hub.list_repo_tree.return_value = [
    MagicMock(rfilename="voices/af_heart.pt"),
    MagicMock(rfilename="voices/af_bella.pt"),
    MagicMock(rfilename="voices/am_adam.pt"),
    MagicMock(rfilename="voices/bf_alice.pt"),
    MagicMock(rfilename="voices/bm_daniel.pt"),
    MagicMock(rfilename="voices/jf_alpha.pt"),
    MagicMock(rfilename="voices/zf_xiaobei.pt"),
    MagicMock(rfilename="voices/ef_dora.pt"),
    MagicMock(rfilename="voices/ff_siwis.pt"),
    MagicMock(rfilename="voices/hf_alpha.pt"),
    MagicMock(rfilename="voices/if_sara.pt"),
    MagicMock(rfilename="voices/pf_dora.pt"),
]
sys.modules["huggingface_hub"] = _hf_hub
