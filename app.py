import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import joblib
import re
import io
import random
import hashlib
from gtts import gTTS
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Darija Comment Filter", page_icon="🇲🇦", layout="wide")

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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    clf = joblib.load("codeswitch_model.joblib")
    cli = InferenceClient()
    return clf, cli

classifier, client = load_models()

ARABIC_RE    = re.compile(r'[\u0600-\u06FF]')
FRENCH_SIGNS = re.compile(r'[àâäçéèêëîïôöùûüÿœæ]', re.IGNORECASE)
FRENCH_STOP  = set("je tu il elle nous vous les des une pour dans sur avec qui que est sont merci bonjour ça ce cette".split())
LATIN_DARIJA = re.compile(r'\b(wach|labas|bzzaf|bghit|kayn|mzyan|safi|wakha|daba|hada|howa|hiya|ntuma|ana|walakin|3lash|kifash|zwina|mashi|nta|nti|rani|khas|bayna|tayban|makayn|fhmt|t3rf|wqe3|dyal|had|lli|fin|fash|bach|ach)\b', re.IGNORECASE)

COLORS = ["#e74c3c","#e67e22","#2ecc71","#1abc9c","#3498db","#9b59b6","#e91e63","#00bcd4","#ff5722","#f39c12"]
NAMES  = ["أمين","سارة","يوسف","فاطمة","عمر","نور","خالد","ليلى","رشيد","هند","كريم","سلمى"]
TIMES  = ["2m","5m","12m","27m","1h","2h","3h","5h","8h","1d","2d","3d"]

def av_color(s): return COLORS[int(hashlib.md5(s.encode()).hexdigest()[:4],16) % len(COLORS)]
def initials(n): p=n.split(); return (p[0][0]+p[1][0]).upper() if len(p)>=2 else n[:2].upper()

def detect(text):
    lbl = classifier.predict([text])[0]
    ar  = bool(ARABIC_RE.search(text))
    fr  = bool(FRENCH_SIGNS.search(text)) or sum(1 for w in text.lower().split() if w in FRENCH_STOP)>=1
    dj  = bool(LATIN_DARIJA.search(text))
    if lbl=="AR" and fr:          return "MIXED"
    if lbl=="FR" and (ar or dj):  return "MIXED"
    return lbl

def looks_french(t):
    if FRENCH_SIGNS.search(t): return True
    return sum(1 for w in t.lower().split() if w in FRENCH_STOP)>=2

def translate(text, src="fra_Latn"):
    try:
        r = client.translation(text, model="facebook/nllb-200-distilled-600M", src_lang=src, tgt_lang="arb_Arab")
        return r if isinstance(r,str) else getattr(r,'translation_text', str(r))
    except: return "[Translation unavailable]"

def full_translate(text, label):
    if label=="AR": return text, "ar-kept"
    if label=="FR": return translate(text), "fr-ar"
    parts = re.split(r'([\u0600-\u06FF][^\u0000-\u0040]*)', text)
    out = []
    for p in parts:
        p=p.strip()
        if not p: continue
        if ARABIC_RE.search(p): out.append(p)
        elif looks_french(p):   out.append(translate(p))
        else:                   out.append(p)
    return ' '.join(out), "mixed-ar"

def audio(text):
    try:
        tts=gTTS(text=text,lang='ar',slow=False); buf=io.BytesIO(); tts.write_to_fp(buf); buf.seek(0); return buf.read()
    except: return None

def badge(label):
    d={"AR":("#0a2e1a","#22c55e","#166534","🇲🇦 Darija"),"FR":("#0a1929","#60a5fa","#1e3a5f","🇫🇷 Français"),"MIXED":("#1a0f00","#fb923c","#7c2d12","🔀 Mixed")}.get(label,("#111","#888","#444",label))
    return f'<span style="background:{d[0]};color:{d[1]};border:1px solid {d[2]};padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700">{d[3]}</span>'

# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding-bottom:20px;border-bottom:1px solid #21262d;margin-bottom:28px">
  <div style="width:44px;height:44px;background:#238636;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:24px">🇲🇦</div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#f0f6fc">Darija Comment Filter</div>
    <div style="font-size:13px;color:#6b7280;margin-top:3px">Language detection · Arabic translation · Audio — built for Moroccan social media</div>
  </div>
</div>""", unsafe_allow_html=True)

# Input layout
col_in, col_opt = st.columns([3,1], gap="large")
with col_in:
    st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px">Paste comments — one per line</p>', unsafe_allow_html=True)
    text_in = st.text_area("", placeholder="C'est vraiment incroyable !\nكيف داير اليوم؟\nwach labas, Ça va bien ?", height=180, label_visibility="collapsed")
    st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin:12px 0 4px">Or upload a .txt file</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["txt"], label_visibility="collapsed")

with col_opt:
    st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px">Filter by language</p>', unsafe_allow_html=True)
    f_ar  = st.checkbox("🇲🇦  Darija / Arabic", value=True)
    f_fr  = st.checkbox("🇫🇷  French",           value=True)
    f_mix = st.checkbox("🔀  Mixed",             value=True)
    st.markdown('<p style="font-size:11px;color:#8b949e;letter-spacing:1.2px;text-transform:uppercase;margin:18px 0 10px">Options</p>', unsafe_allow_html=True)
    show_trans = st.checkbox("Show Arabic translation", value=True)
    gen_audio  = st.checkbox("Generate audio",          value=True)

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

    counts = {"AR":0,"FR":0,"MIXED":0}
    for _,l in detected:
        if l in counts: counts[l]+=1

    st.markdown("<br>", unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total",  len(comments))
    m2.metric("Darija", counts["AR"])
    m3.metric("French", counts["FR"])
    m4.metric("Mixed",  counts["MIXED"])
    st.markdown("<br>", unsafe_allow_html=True)

    sel = [l for l,on in [("AR",f_ar),("FR",f_fr),("MIXED",f_mix)] if on]
    filtered = [(c,l) for c,l in detected if l in sel]

    if not filtered:
        st.warning("No comments match the selected filters.")
        st.stop()

    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #21262d">
      <span style="font-size:17px;font-weight:700;color:#f0f6fc">Post Comments</span>
      <span style="margin-left:auto;font-size:11px;color:#6b7280;letter-spacing:0.5px">SHOWING {len(filtered)} OF {len(comments)}</span>
    </div>""", unsafe_allow_html=True)

    random.seed(42)
    for i,(text,label) in enumerate(filtered):
        uname = random.choice(NAMES)+str(random.randint(10,99))
        ts    = random.choice(TIMES)
        col   = av_color(uname)
        ini   = initials(uname)
        likes = random.randint(3,240)
        reps  = random.randint(0,40)

        translated, method, ab = "", "", None
        if show_trans:
            with st.spinner(f"Translating {i+1}/{len(filtered)}..."):
                translated, method = full_translate(text, label)
            if gen_audio and method!="ar-kept":
                ab = audio(translated)

        if show_trans and method=="ar-kept":
            tblock = '<div style="margin-top:10px;padding:8px 12px;background:#0a2e1a;border-radius:8px;font-size:12px;color:#22c55e">Arabic / Darija — no translation needed</div>'
        elif show_trans:
            mlabel = "Translated from French" if method=="fr-ar" else "Translated from Mixed"
            tblock = f'<div style="margin-top:12px;padding:12px 16px;background:#0d1117;border-radius:10px;border-left:3px solid {col}"><div style="font-size:10px;color:#6b7280;margin-bottom:8px;letter-spacing:0.5px">{mlabel}</div><div style="font-family:Noto Sans Arabic,sans-serif;font-size:16px;color:#f0f6fc;direction:rtl;text-align:right;line-height:2">{translated}</div></div>'
        else:
            tblock = ""

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:14px;padding:18px 20px;margin:10px 0">
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

        if ab:
            st.audio(ab, format="audio/mp3")

st.markdown("""
<div style="margin-top:48px;padding-top:16px;border-top:1px solid #21262d;font-size:11px;color:#484f58;text-align:center">
  Classifier: Char TF-IDF + LinearSVC + Hybrid Rules &nbsp;·&nbsp; 96.1% accuracy &nbsp;·&nbsp; κ = 0.94
  &nbsp;&nbsp;|&nbsp;&nbsp; Translation: NLLB-200 (Meta AI) via HF Inference API &nbsp;·&nbsp; BLEU 0.59
</div>""", unsafe_allow_html=True)
