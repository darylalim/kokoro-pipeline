import io
import random
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

SAMPLES: dict[str, list[str]] = {
    "a": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore, and the shells she sells are seashells, I'm sure.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, but a comfortable hobbit-hole.",
    ],
    "b": [
        "The rain in Spain stays mainly in the plain.",
        "To be or not to be, that is the question.",
        "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
    ],
    "e": [
        "El que no arriesga, no gana.",
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, vivía un hidalgo.",
        "La vida es sueño, y los sueños, sueños son.",
    ],
    "f": [
        "La vie est belle quand on prend le temps de la savourer.",
        "Tout ce qui brille n'est pas or.",
        "Il faut imaginer Sisyphe heureux.",
    ],
    "h": [
        "जहाँ चाह, वहाँ राह।",
        "अच्छी सेहत सबसे बड़ा धन है।",
        "बूँद बूँद से सागर भरता है।",
    ],
    "i": [
        "Chi dorme non piglia pesci.",
        "La semplicità è la sofisticazione suprema.",
        "Tutte le strade portano a Roma.",
    ],
    "j": [
        "七転び八起き。何度失敗しても、また立ち上がればいい。",
        "花は桜木、人は武士。美しさには気高さが宿る。",
        "猿も木から落ちる。誰にでも失敗はあるものだ。",
    ],
    "p": [
        "Água mole em pedra dura, tanto bate até que fura.",
        "A vida é feita de pequenos momentos que valem a pena ser vividos.",
        "Quem não tem cão, caça com gato.",
    ],
    "z": [
        "千里之行，始于足下。",
        "学而不思则罔，思而不学则殆。",
        "天下没有不散的宴席，珍惜每一次相聚。",
    ],
}

LONG_SAMPLES: dict[str, str] = {
    # Walden (Thoreau)
    "a": (
        "I went to the woods because I wished to live deliberately, to"
        " front only the essential facts of life, and see if I could not"
        " learn what it had to teach, and not, when I came to die,"
        " discover that I had not lived. I did not wish to live what was"
        " not life, living is so dear; nor did I wish to practise"
        " resignation, unless it was quite necessary. I wanted to live"
        " deep and suck out all the marrow of life, to live so sturdily"
        " and Spartan-like as to put to rout all that was not life, to"
        " cut a broad swath and shave close, to drive life into a"
        " corner, and reduce it to its lowest terms, and, if it proved"
        " to be mean, why then to get the whole and genuine meanness of"
        " it, and publish its meanness to the world; or if it were"
        " sublime, to know it by experience, and be able to give a true"
        " account of it in my next excursion."
    ),
    # The Wind in the Willows (Grahame)
    "b": (
        "The Mole had been working very hard all the morning,"
        " spring-cleaning his little home. First with brooms, then with"
        " dusters; then on ladders and steps and chairs, with a brush"
        " and a pail of whitewash; till he had dust in his throat and"
        " eyes, and splashes of whitewash all over his black fur, and an"
        " aching back and weary arms. Spring was moving in the air above"
        " and in the earth below and around him, penetrating even his"
        " dark and lowly little house with its spirit of divine"
        " discontent and longing. It was small wonder, then, that he"
        ' suddenly flung down his brush on the floor, said, "Bother!"'
        ' and "O blow!" and also "Hang spring-cleaning!" and bolted out'
        " of the house without even waiting to put on his coat. Something"
        " up above was calling him imperiously, and he made for the steep"
        " little tunnel which answered in his case to the gravelled"
        " carriage-drive owned by animals whose residences are nearer to"
        " the sun and air. So he scraped and scratched and scrabbled and"
        " scrooged, and then he scrooged again and scrabbled and"
        " scratched and scraped, working busily with his little paws and"
        ' muttering to himself, "Up we go! Up we go!" till at last, pop!'
        " his snout came out into the sunlight and he found himself"
        " rolling in the warm grass of a great meadow."
    ),
    # Platero y yo (Jiménez)
    "e": (
        "Platero es pequeño, peludo, suave; tan blando por fuera, que se"
        " diría todo de algodón, que no lleva huesos. Sólo los espejos"
        " de azabache de sus ojos son duros cual dos escarabajos de"
        " cristal negro. Lo dejo suelto, y se va al prado, y acaricia"
        " tibiamente con su hocico, rozándolas apenas, las florecillas"
        " rosas, celestes y gualdas... Lo llamo dulcemente: «¿Platero?»"
        " y viene a mí con un trotecillo alegre que parece que se ríe en"
        " no sé qué cascabeleo ideal... Come cuanto le doy. Le gustan"
        " las naranjas mandarinas, las uvas moscateles, todas de ámbar;"
        " los higos morados, con su cristalina gotita de miel... Es"
        " tierno y mimoso igual que un niño, que una niña...; pero"
        " fuerte y seco por dentro como de piedra. Cuando paso sobre él,"
        " los domingos, por las últimas callejas del pueblo, los hombres"
        " del campo, vestidos de limpio y despaciosos, se quedan"
        " mirándolo: —Tien' asero... Tiene acero. Acero y plata de"
        " luna, al mismo tiempo."
    ),
    # Vingt mille lieues sous les mers (Verne)
    "f": (
        "L'année 1866 fut marquée par un événement bizarre, un"
        " phénomène inexpliqué et inexplicable que personne n'a sans"
        " doute oublié. Sans parler des rumeurs qui agitaient les"
        " populations des ports et surexcitaient l'esprit public à"
        " l'intérieur des continents, les gens de mer furent"
        " particulièrement émus. Les négociants, armateurs, capitaines"
        " de navires, skippers et masters de l'Europe et de l'Amérique,"
        " officiers des marines militaires de tous pays, et, après eux,"
        " les gouvernements des divers États des deux continents, se"
        " préoccupèrent de ce fait au plus haut point. En effet, depuis"
        " quelque temps, plusieurs navires s'étaient rencontrés sur mer"
        " avec « une chose énorme » un objet long, fusiforme, parfois"
        " phosphorescent, infiniment plus vaste et plus rapide qu'une"
        " baleine. Les faits relatifs à cette apparition, consignés aux"
        " divers livres de bord, s'accordaient assez exactement sur la"
        " structure de l'objet ou de l'être en question, la vitesse"
        " inouïe de ses mouvements, la puissance surprenante de sa"
        " locomotion, la vie particulière dont il semblait doué. Si"
        " c'était un cétacé, il surpassait en volume tous ceux que la"
        " science avait classés jusqu'alors."
    ),
    # Idgah (Premchand)
    "h": (
        "रमज़ान के पूरे तीस रोज़ के बाद आज ईद आई है। कितना मनोहर;"
        " कितना सुहावना प्रभात है। वृक्षों पर कुछ अजीब हरियाली है,"
        " खेतों में कुछ अजीब रौनक है, आसमान पर कुछ अजीब लालिमा है।"
        " आज का सूर्य देखो, कितना प्यारा, कितना शीतल है, मानों"
        " संसार को ईद की बधाई दे रहा है। गाँव में कितनी हलचल है।"
        " ईदगाह जाने की तैयारियाँ हो रही हैं। किसी के कुरते में बटन"
        " नहीं है। पड़ोस के घर से सुई-तागा लेने दौड़ा जा रहा है।"
        " किसी के जूते कड़े हो गये हैं, उनमें तेल डालने के लिए तेली"
        " के घर भागा जाता है। जल्दी-जल्दी बैलों की सानी-पानी दे दें।"
        " ईदगाह से लौटते-लौटते दोपहर हो जायेगा। तीन कोस का पैदल"
        " रास्ता, फिर सैकड़ों आदमियों से मिलना-भेंटना। दोपहर के पहले"
        " लौटना असम्भव है। लड़के सबसे यादा प्रसन्न हैं। किसी ने एक"
        " रोज़ा रखा है, वह भी दोपहर तक, किसी ने वह भी नहीं है लेकिन"
        " ईदगाह जाने की खुशी उनके हिस्से की चीज़ हैं। रोज़े बड़े-बूढ़े"
        " के लिए होंगे। इनके लिए तो ईद है। रोज़ ईद का नाम रटते थे।"
        " आज वह आ गई। अब जल्दी पड़ी है कि लोग ईदगाह क्यों नहीं"
        " चलते। इन्हें गृहस्थी को चिन्ताओं से क्या प्रयोजन!"
    ),
    # Le avventure di Pinocchio (Collodi)
    "i": (
        "C'era una volta... — Un re! — diranno subito i miei piccoli"
        " lettori. No, ragazzi, avete sbagliato. C'era una volta un"
        " pezzo di legno. Non era un legno di lusso, ma un semplice"
        " pezzo da catasta, di quelli che d'inverno si mettono nelle"
        " stufe e nei caminetti per accendere il fuoco e per riscaldare"
        " le stanze. Non so come andasse, ma il fatto gli è che un bel"
        " giorno questo pezzo di legno capitò nella bottega di un vecchio"
        " falegname, il quale aveva nome mastr'Antonio, se non che tutti"
        " lo chiamavano maestro Ciliegia, per via della punta del suo"
        " naso, che era sempre lustra e paonazza, come una ciliegia"
        " matura. Appena maestro Ciliegia ebbe visto quel pezzo di"
        " legno, si rallegrò tutto; e dandosi una fregatina di mani per"
        " la contentezza, borbottò a mezza voce: — Questo legno è"
        " capitato a tempo; voglio servirmene per fare una gamba di"
        " tavolino. — Detto fatto, prese subito l'ascia arrotata per"
        " cominciare a levargli la scorza e a digrossarlo; ma quando fu"
        " lì per lasciare andare la prima asciata, rimase col braccio"
        " sospeso in aria, perchè sentì una vocina sottile sottile, che"
        " disse raccomandandosi: — Non mi picchiar tanto forte!"
    ),
    # Botchan (Sōseki)
    "j": (
        "親譲りの無鉄砲で小供の時から損ばかりしている。小学校に居る時分学校の"
        "二階から飛び降りて一週間ほど腰を抜かした事がある。なぜそんな無闇をした"
        "と聞く人があるかも知れぬ。別段深い理由でもない。新築の二階から首を出し"
        "ていたら、同級生の一人が冗談に、いくら威張っても、そこから飛び降りる事"
        "は出来まい。弱虫やーい。と囃したからである。小使に負ぶさって帰って来た"
        "時、おやじが大きな眼をして二階ぐらいから飛び降りて腰を抜かす奴があるか"
        "と云ったから、この次は抜かさずに飛んで見せますと答えた。親類のものから"
        "西洋製のナイフを貰って奇麗な刃を日に翳して、友達に見せていたら、一人が"
        "光る事は光るが切れそうもないと云った。切れぬ事があるか、何でも切ってみ"
        "せると受け合った。"
    ),
    # O Guarani (José de Alencar)
    "p": (
        "A habitação, que acabamos de descrever, pertencia a D. Antonio"
        " de Mariz, fidalgo portuguez cota d'armas e um dos fundadores"
        " da cidade do Rio de Janeiro. Era um dos cavalheiros que mais"
        " se havião distinguido nas guerras da conquista, contra a"
        " invasão dos francezes e os ataques dos selvagens. Em 1567"
        " acompanhou Mem de Sá ao Rio de Janeiro, e depois da victoria"
        " alcançada pelos portuguezes, auxiliou o governador nos"
        " trabalhos da fundação da cidade e consolidação do domínio de"
        " Portugal nessa capitania. Fez parte em 1578 da celebre"
        " expedição do Dr. Antonio de Salema contra os francezes, que"
        " havião estabeleoido uma feitoria em Cabo Frio para fazerem o"
        " contrabando de páo-brasil. Servio por este mesmo tempo de"
        " provedor da real fazenda, e depois da alfandega do Rio de"
        " Janeiro; e mostrou sempre nesses empregos o seu zelo pela"
        " fazenda, e a sua dedicação ao rei. Homem de valor,"
        " experimentado na guerra, activo, affeito a combater os indios,"
        " prestou grandes serviços nas descobertas e explorações do"
        " interior de Minas e Espirito Santo."
    ),
    # 西遊記 Journey to the West (Wu Cheng'en)
    "z": (
        "蓋聞天地之數，有十二萬九千六百歲為一元。將一元分為十二會，乃子、丑、"
        "寅、卯、辰、巳、午、未、申、酉、戌、亥之十二支也。每會該一萬八百歲。"
        "且就一日而論：子時得陽氣，而丑則雞鳴；寅不通光，而卯則日出；辰時食後，"
        "而巳則挨排；日午天中，而未則西蹉；申時晡，而日落酉，戌黃昏，而人定亥。"
        "譬於大數，若到戌會之終，則天地昏曚而萬物否矣。再去五千四百歲，交亥會之"
        "初，則當黑暗，而兩間人物俱無矣，故曰混沌。又五千四百歲，亥會將終，貞下"
        "起元，近子之會，而復逐漸開明。邵康節曰：「冬至子之半，天心無改移。一陽"
        "初動處，萬物未生時。」到此，天始有根。再五千四百歲，正當子會，輕清上騰，"
        "有日，有月，有星，有辰。日、月、星、辰，謂之四象。故曰，天開於子。又經"
        "五千四百歲，子會將終，近丑之會，而逐漸堅實。"
    ),
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
) -> Generator[tuple[np.ndarray, str], None, None]:
    generated = False
    for result in model.generate(
        text=text, voice=voice, speed=speed, lang_code=lang_code
    ):
        if result.audio is not None:
            generated = True
            yield np.array(result.audio, dtype=np.float32), result.phonemes or ""
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
if "sample_text" in st.session_state:
    st.session_state["text_input"] = st.session_state.pop("sample_text")
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

btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
with btn_col1:
    generate_clicked = st.button("Generate", type="primary")
with btn_col2:
    tokenize_clicked = st.button("Tokenize")
with btn_col3:
    if st.button("Random Sample"):
        st.session_state["sample_text"] = random.choice(SAMPLES[lang_code])
        st.rerun()
with btn_col4:
    if st.button("Long Sample"):
        st.session_state["sample_text"] = LONG_SAMPLES[lang_code]
        st.rerun()

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
            for v in selected_voices:
                start = time.perf_counter()
                with st.status(f"Generating {v}...", expanded=True) as status:
                    audio_chunks = []
                    phoneme_chunks = []
                    for i, (audio_chunk, phonemes) in enumerate(
                        generate_speech(
                            text_input, v, pipeline, speed=speed, lang_code=lang_code
                        ),
                        1,
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
    elif len(text_input) > CHAR_LIMIT:
        st.warning(f"Text exceeds {CHAR_LIMIT} character limit.")
    else:
        phonemes = tokenize_text(text_input, lang_code)
        with st.expander("Phoneme Tokens", expanded=True):
            st.code(phonemes)

if st.session_state["current_output"] is not None:
    render_output(st.session_state["current_output"])
