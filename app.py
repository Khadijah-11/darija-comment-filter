import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import joblib
import re
import io
import random
import hashlib
from gtts import gTTS

st.set_page_config(page_title="Darija Comment Filter", page_icon="🌍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;600&family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'DM Sans',sans-serif!important; background:#0d1117!important; color:#c9d1d9!important; }
.stApp { background:#0d1117!important; }
.block-container { padding:2rem 2rem 4rem!important; max-width:1000px!important; }
#MainMenu, footer, header { visibility:hidden!important; }
.stTextArea textarea { background:#161b22!important; border:1px solid #30363d!important; color:#c9d1d9!important; border-radius:10px!important; font-size:14px!important; }
.stTextArea textarea:focus { border-color:#388bfd!important; }
[data-testid="stCheckbox"] label { color:#8b949e!important; font-size:13px!important; }
.stButton>button { background:#238636!important; color:white!important; border:none!important; border-radius:8px!important; font-weight:600!important; font-size:15px!important; padding:12px 28px!important; width:100%!important; }
.stButton>button:hover { background:#2ea043!important; }
[data-testid="stMetric"] { background:#161b22!important; border:1px solid #30363d!important; border-radius:12px!important; padding:16px!important; }
[data-testid="stMetricValue"] { color:#f0f6fc!important; font-size:28px!important; font-weight:700!important; }
[data-testid="stMetricLabel"] { color:#8b949e!important; font-size:11px!important; letter-spacing:1.5px!important; text-transform:uppercase!important; }
div[data-testid="stHorizontalBlock"] { gap: 8px!important; }
</style>
""", unsafe_allow_html=True)

# ── Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    return joblib.load("codeswitch_model.joblib")

@st.cache_resource
def load_translation_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    return tok, mdl

classifier = load_classifier()

# ── Patterns ──────────────────────────────────────────────────────────
ARABIC_RE    = re.compile(r'[\u0600-\u06FF]')
FRENCH_SIGNS = re.compile(r'[àâäçéèêëîïôöùûüÿœæ]', re.IGNORECASE)
FRENCH_STOP  = set("je tu il elle nous vous les des une pour dans sur avec qui que est sont merci bonjour ça ce cette".split())
LATIN_DARIJA = re.compile(
    r'\b(wach|labas|bzzaf|bghit|kayn|mzyan|safi|wakha|daba|hada|howa|hiya|'
    r'ntuma|ana|walakin|3lash|kifash|zwina|mashi|nta|nti|rani|khas|bayna|'
    r'tayban|makayn|fhmt|t3rf|wqe3|dyal|had|lli|fin|fash|bach|ach)\b',
    re.IGNORECASE
)

COLORS = ["#e74c3c","#e67e22","#2ecc71","#1abc9c","#3498db","#9b59b6","#e91e63","#00bcd4","#ff5722","#f39c12"]
NAMES  = ["أمين","سارة","يوسف","فاطمة","عمر","نور","خالد","ليلى","رشيد","هند","كريم","سلمى"]
TIMES  = ["2m","5m","12m","27m","1h","2h","3h","5h","8h","1d","2d","3d"]

def av_color(s): return COLORS[int(hashlib.md5(s.encode()).hexdigest()[:4],16) % len(COLORS)]
def initials(n): p=n.split(); return (p[0][0]+p[1][0]).upper() if len(p)>=2 else n[:2].upper()

# ── Detection ─────────────────────────────────────────────────────────
def detect(text):
    lbl = classifier.predict([text])[0]
    ar  = bool(ARABIC_RE.search(text))
    fr  = bool(FRENCH_SIGNS.search(text)) or sum(1 for w in text.lower().split() if w in FRENCH_STOP) >= 1
    dj  = bool(LATIN_DARIJA.search(text))
    if lbl == "AR" and fr:         return "MIXED"
    if lbl == "FR" and (ar or dj): return "MIXED"
    return lbl

def looks_french(t):
    if FRENCH_SIGNS.search(t): return True
    return sum(1 for w in t.lower().split() if w in FRENCH_STOP) >= 2

# ── Translation ───────────────────────────────────────────────────────
def translate(text, src="fra_Latn"):
    tok, mdl = load_translation_model()
    inputs    = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    target_id = tok.convert_tokens_to_ids("arb_Arab")
    out       = mdl.generate(**inputs, forced_bos_token_id=target_id, max_length=256)
    return tok.decode(out[0], skip_special_tokens=True)

def full_translate(text, label):
    if label == "AR":
        return text, "ar-kept"
    if label == "FR":
        return translate(text, "fra_Latn"), "fr-ar"
    if label == "MIXED":
        parts = re.split(r'([\u0600-\u06FF][^\u0000-\u0040]*)', text)
        out = []
        for p in parts:
            p = p.strip()
            if not p: continue
            if ARABIC_RE.search(p):  out.append(p)
            elif looks_french(p):    out.append(translate(p, "fra_Latn"))
            else:                    out.append(p)
        result = ' '.join(out)
        if result.strip() == text.strip():
            return translate(text, "fra_Latn"), "mixed-ar"
        return result, "mixed-ar"
    return text, "unknown"

# ── Audio ─────────────────────────────────────────────────────────────
def make_audio(text, label):
    """
    AR  → read original text in Arabic script with ar TTS
          but also offer latin Darija read in French TTS (sounds more natural)
    FR/MIXED → read translated Arabic text with ar TTS
    """
    try:
        # For Darija (AR): use French TTS on the latin text — sounds more natural than Arabic TTS
        # We detect if text has Arabic script or is latin Darija
        has_arabic_script = bool(ARABIC_RE.search(text))
        if label == "AR" and not has_arabic_script:
            # Latin Darija — read with French TTS (closest natural sound)
            lang = "fr"
        elif label == "AR" and has_arabic_script:
            # Arabic script Darija — read with Arabic TTS
            lang = "ar"
        else:
            # Translated Arabic text
            lang = "ar"

        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except:
        return None

# ── Badge ─────────────────────────────────────────────────────────────
def badge(label):
    d = {
        "AR":    ("#0a2e1a","#22c55e","#166534","🇲🇦 Darija"),
        "FR":    ("#0a1929","#60a5fa","#1e3a5f","🇫🇷 Français"),
        "MIXED": ("#1a0f00","#fb923c","#7c2d12","🔀 Mixed"),
    }.get(label, ("#111","#888","#444", label))
    bg, color, border, text = d
    return (f'<span style="background:{bg};color:{color};border:1px solid {border};'
            f'padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700">{text}</span>')

# ── Render one comment card ───────────────────────────────────────────
def render_card(item):
    text, label, translated, method, uname, ts, likes, reps = (
        item["text"], item["label"], item["translated"], item["method"],
        item["uname"], item["ts"], item["likes"], item["reps"]
    )
    col  = av_color(uname)
    ini  = initials(uname)

    if method == "ar-kept":
        tblock = '<div style="margin-top:10px;padding:8px 12px;background:#0a2e1a;border-radius:8px;font-size:12px;color:#22c55e">Darija — displayed as-is</div>'
    else:
        mlabel = {"fr-ar": "Translated from French", "mixed-ar": "Translated from Mixed"}.get(method, "Translated")
        tblock = f'''<div style="margin-top:12px;padding:12px 16px;background:#0d1117;border-radius:10px;border-left:3px solid {col}">
            <div style="font-size:10px;color:#6b7280;margin-bottom:8px;letter-spacing:0.5px">{mlabel}</div>
            <div style="font-family:Noto Sans Arabic,sans-serif;font-size:16px;color:#f0f6fc;direction:rtl;text-align:right;line-height:2">{translated}</div>
        </div>'''

    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:14px;padding:18px 20px;margin:8px 0">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
        <div style="width:42px;height:42px;border-radius:50%;background:{col};display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;color:#fff;flex-shrink:0">{ini}</div>
        <div style="flex:1">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <span style="font-weight:600;font-size:14px;color:#e6edf3">{uname}</span>{badge(label)}
          </div>
          <div style="font-size:11px;color:#6b7280;margin-top:2px">{ts} ago</div>
        </div>
      </div>
      <div style="font-size:15px;color:#c9d1d9;line-height:1.75">{text}</div>
      {tblock}
      <div style="display:flex;gap:20px;margin-top:14px;padding-top:12px;border-top:1px solid #21262d">
        <span style="font-size:12px;color:#6b7280">👍 {likes}</span>
        <span style="font-size:12px;color:#6b7280">💬 {reps}</span>
        <span style="font-size:12px;color:#6b7280">↩️ Reply</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Audio — for AR read original, for others read translated Arabic
    audio_text = text if method == "ar-kept" else translated
    ab = make_audio(audio_text, label)
    if ab:
        st.audio(ab, format="audio/mp3")

# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding-bottom:20px;border-bottom:1px solid #21262d;margin-bottom:28px">
  <div style="width:44px;height:44px;background:#238636;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:24px">🌍</div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#f0f6fc">Darija Comment Filter</div>
    <div style="font-size:13px;color:#6b7280;margin-top:3px">Language detection · Arabic translation · Audio — built for Moroccan social media</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Input section (only shown before analysis) ────────────────────────
if "results" not in st.session_state:

    col_in, col_opt = st.columns([3,1], gap="large")

    with col_in:
        st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px">Paste comments — one per line</p>', unsafe_allow_html=True)
        text_in = st.text_area("", placeholder="C'est vraiment incroyable !\nكيف داير اليوم؟\nwach labas, Ça va bien ?", height=180, label_visibility="collapsed")
        st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin:12px 0 4px">Or upload a .txt file</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["txt"], label_visibility="collapsed")

    with col_opt:
        st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px">Options</p>', unsafe_allow_html=True)
        show_trans = st.checkbox("Show Arabic translation", value=True)
        gen_audio  = st.checkbox("Generate audio for all",  value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Analyze Comments")

    if run:
        comments = []
        if uploaded:
            comments = [l.strip() for l in uploaded.read().decode("utf-8").splitlines() if l.strip()]
        elif text_in.strip():
            comments = [l.strip() for l in text_in.strip().splitlines() if l.strip()]

        if not comments:
            st.warning("Please enter at least one comment.")
            st.stop()

        with st.spinner("Detecting languages..."):
            detected = [(c, detect(c)) for c in comments]

        if show_trans:
            with st.spinner("Loading translation model — first time takes ~1 min..."):
                load_translation_model()

        results = []
        random.seed(42)
        for i, (text, label) in enumerate(detected):
            uname = random.choice(NAMES) + str(random.randint(10,99))
            ts    = random.choice(TIMES)
            likes = random.randint(3, 240)
            reps  = random.randint(0, 40)

            translated, method = text, "ar-kept"
            if show_trans:
                with st.spinner(f"Translating {i+1}/{len(detected)}..."):
                    translated, method = full_translate(text, label)

            results.append({
                "text": text, "label": label,
                "translated": translated, "method": method,
                "uname": uname, "ts": ts, "likes": likes, "reps": reps,
                "gen_audio": gen_audio
            })

        st.session_state["results"] = results
        st.rerun()

# ── Results section ───────────────────────────────────────────────────
else:
    results = st.session_state["results"]
    counts  = {"AR": 0, "FR": 0, "MIXED": 0}
    for r in results:
        if r["label"] in counts: counts[r["label"]] += 1

    # Stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total",  len(results))
    m2.metric("Darija", counts["AR"])
    m3.metric("French", counts["FR"])
    m4.metric("Mixed",  counts["MIXED"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter bar — live, no re-analysis needed
    st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px">Filter by language</p>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns([1,1,1,2])
    with fc1: f_ar  = st.checkbox("🇲🇦 Darija",  value=True, key="f_ar")
    with fc2: f_fr  = st.checkbox("🇫🇷 French",  value=True, key="f_fr")
    with fc3: f_mix = st.checkbox("🔀 Mixed",    value=True, key="f_mix")
    with fc4:
        if st.button("↩ Analyze new comments"):
            del st.session_state["results"]
            st.rerun()

    # Apply filter
    sel      = [l for l, on in [("AR", f_ar), ("FR", f_fr), ("MIXED", f_mix)] if on]
    filtered = [r for r in results if r["label"] in sel]

    st.markdown("<br>", unsafe_allow_html=True)

    # Feed header
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #21262d">
      <span style="font-size:17px;font-weight:700;color:#f0f6fc">Post Comments</span>
      <span style="margin-left:auto;font-size:11px;color:#6b7280;letter-spacing:0.5px">SHOWING {len(filtered)} OF {len(results)}</span>
    </div>""", unsafe_allow_html=True)

    if not filtered:
        st.info("No comments match the selected filters.")
    else:
        for item in filtered:
            render_card(item)


