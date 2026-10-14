import html
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Text preprocessing
# The saved pipeline references a custom `wordopt` function in __main__, so it
# MUST be defined at top level before the model is loaded.
# ---------------------------------------------------------------------------
try:
    from nltk.stem import PorterStemmer

    _stem = PorterStemmer().stem
    NLTK_MISSING = False
except ImportError:
    _stem = lambda w: w  # noqa: E731
    NLTK_MISSING = True


def wordopt(text):
    """Clean + stem a review (matches the vocabulary the model was trained on)."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(_stem(w) for w in text.split())


MODEL_PATH = Path(__file__).parent / "Review.joblib"
LABELS = {0: "Negative", 1: "Positive"}

EXAMPLES = {
    "😍 Glowing": "An absolutely brilliant film. Great acting, a beautiful story and I loved every minute of it. Highly recommend!",
    "😡 Scathing": "Terrible movie. A complete waste of time with awful acting, a boring plot and a ridiculous ending.",
    "😐 Mixed": "The visuals were gorgeous but the story was dull and the pacing dragged. Some good moments, though.",
}


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    model = joblib.load(path)
    # Compatibility: the model was saved with a newer scikit-learn, which dropped
    # `multi_class`. Older installed versions still look for it during predict.
    clf = model.steps[-1][1]
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    return model


def predict(model, texts):
    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    return preds, probs


def pos_index(model):
    return list(model.classes_).index(1)


def highlight_words(model, text, top=None):
    """Return HTML of the review with words coloured by their influence."""
    vec, clf = model.named_steps["Tfidf"], model.named_steps["classifier"]
    vocab, coef = vec.vocabulary_, clf.coef_[0]
    scale = np.percentile(np.abs(coef), 99) or 1.0
    out = []
    for tok in re.findall(r"\S+", text):
        stem = wordopt(re.sub(r"[^A-Za-z]", "", tok))
        idx = vocab.get(stem)
        safe = html.escape(tok)
        if idx is None or abs(coef[idx]) < 0.15 * scale:
            out.append(safe)
            continue
        w = max(-1, min(1, coef[idx] / scale))
        rgb = "34,197,94" if w > 0 else "239,68,68"
        out.append(
            f'<span class="hl" style="background:rgba({rgb},{0.15 + 0.45 * abs(w):.2f})" '
            f'title="{"positive" if w > 0 else "negative"} signal">{safe}</span>'
        )
    return " ".join(out)


# ---------------------------------------------------------------------------
# Page + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Review Critic AI", page_icon="🎬", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Playfair+Display:ital,wght@0,600;0,800;1,500&family=Inter:wght@400;500;600&display=swap');

:root {--gold:#f5c518; --red:#e50914; --ink:#0b0b10; --card:rgba(255,255,255,.06);}

/* ---------- cinema backdrop ---------- */
.stApp {
  background:
    radial-gradient(ellipse 70% 40% at 50% -5%, rgba(255,214,102,.28), transparent 70%),
    radial-gradient(ellipse 50% 35% at 10% 90%, rgba(229,9,20,.18), transparent 70%),
    radial-gradient(ellipse 50% 35% at 95% 70%, rgba(99,102,241,.16), transparent 70%),
    linear-gradient(180deg, #14060a 0%, #0b0b10 45%, #07070b 100%);
  background-attachment: fixed;
  color: #f3efe6;
  font-family: 'Inter', sans-serif;
}
/* film-strip edges */
.stApp::before, .stApp::after {
  content:""; position: fixed; top:0; bottom:0; width: 26px; z-index: 0; opacity:.55;
  background:
    repeating-linear-gradient(180deg, transparent 0 14px, #000 14px 34px) ,
    linear-gradient(90deg, #1a1a1f, #26262d);
  box-shadow: inset 0 0 0 3px #1a1a1f;
  pointer-events:none;
}
.stApp::before {left:0; border-right:2px solid #333;}
.stApp::after  {right:0; border-left:2px solid #333;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.5rem; max-width: 860px; position: relative; z-index: 1;}
footer {visibility: hidden;}

h1,h2,h3,h4 {font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 1.5px; color: var(--gold);}

/* ---------- hero ---------- */
.hero {
  text-align:center; padding: 2.2rem 1rem 1.8rem; margin-bottom: 1.4rem; position:relative;
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(0,0,0,.55), rgba(0,0,0,.2));
  border: 1px solid rgba(245,197,24,.25);
  box-shadow: 0 0 60px rgba(245,197,24,.08), inset 0 0 40px rgba(0,0,0,.6);
  overflow:hidden;
}
.hero::before { /* projector beam */
  content:""; position:absolute; top:-40%; left:50%; width:140%; height:200%; transform:translateX(-50%);
  background: conic-gradient(from 160deg at 50% 0%, transparent 0 8deg, rgba(255,230,160,.10) 20deg 40deg, transparent 52deg);
  pointer-events:none;
}
.hero .kicker {font-family:'Bebas Neue'; letter-spacing:6px; color:var(--red); font-size:1rem;}
.hero h1 {
  font-family:'Bebas Neue',sans-serif !important; font-size: 3.6rem; margin: .1rem 0; padding:0;
  background: linear-gradient(180deg,#fff3bf,#f5c518 60%,#b8860b); -webkit-background-clip:text;
  -webkit-text-fill-color: transparent; text-shadow: none; letter-spacing: 4px;
}
.hero p {font-family:'Playfair Display',serif; font-style: italic; opacity:.85; font-size:1.1rem; margin:0;}
.hero .stars {color: var(--gold); letter-spacing: 8px; margin-top:.6rem; font-size:1.1rem;}

/* ---------- inputs ---------- */
.stTextArea textarea {
  background: rgba(255,255,255,.05) !important; color:#f3efe6 !important;
  border:1px solid rgba(245,197,24,.35) !important; border-radius: 14px !important;
  font-family:'Playfair Display',serif; font-size:1.05rem;
}
.stTextArea textarea:focus {border-color: var(--gold) !important; box-shadow: 0 0 0 2px rgba(245,197,24,.25) !important;}
.stTextArea label p, .stSelectbox label p, .stFileUploader label p {color: var(--gold); font-weight:600;}

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px; font-weight: 600; border:1px solid rgba(245,197,24,.4);
  background: rgba(255,255,255,.05); color:#f3efe6; transition: all .2s;
}
.stButton>button:hover {border-color: var(--gold); color: var(--gold); transform: translateY(-1px);}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {
  background: linear-gradient(90deg,#e50914,#b20710); color:#fff; border:none;
  font-family:'Bebas Neue'; letter-spacing:2px; font-size:1.25rem; padding:.6rem;
  box-shadow: 0 6px 20px rgba(229,9,20,.4);
}
.stButton>button[kind="primary"]:hover {color:#fff; box-shadow: 0 8px 26px rgba(229,9,20,.6);}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid rgba(245,197,24,.25);}
.stTabs [data-baseweb="tab"] {font-family:'Bebas Neue'; letter-spacing:1.5px; font-size:1.15rem; color:#cfc8b8;}
.stTabs [aria-selected="true"] {color: var(--gold) !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color: var(--gold) !important;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#120508,#0b0b10); border-right: 1px solid rgba(245,197,24,.2);
}

/* ---------- verdict ticket ---------- */
.ticket {
  position:relative; display:flex; margin: 1.2rem 0 .8rem; border-radius: 16px; overflow:hidden;
  border: 1px solid rgba(255,255,255,.15); box-shadow: 0 14px 40px rgba(0,0,0,.55);
}
.ticket .main {flex: 1; padding: 1.4rem 1.6rem;}
.ticket .stub {
  width: 150px; padding: 1.2rem .6rem; text-align:center; display:flex; flex-direction:column;
  justify-content:center; border-left: 3px dashed rgba(255,255,255,.35); background: rgba(0,0,0,.28);
}
.ticket.pos {background: linear-gradient(120deg,#0f3d26,#14633c 60%,#1a7f4b);}
.ticket.neg {background: linear-gradient(120deg,#4a0b10,#7a0f18 60%,#a3141f);}
.ticket .tag   {font-family:'Bebas Neue'; letter-spacing:4px; opacity:.75; font-size:.95rem;}
.ticket .verdict {font-family:'Bebas Neue'; font-size:3.4rem; letter-spacing:3px; line-height:1; margin:.15rem 0;}
.ticket .stars-big {color: var(--gold); font-size: 1.6rem; letter-spacing: 4px; text-shadow:0 0 12px rgba(245,197,24,.5);}
.ticket .stars-big .off {color: rgba(255,255,255,.25); text-shadow:none;}
.ticket .quote {font-family:'Playfair Display',serif; font-style:italic; opacity:.9; margin-top:.4rem;}
.ticket .stub .big {font-family:'Bebas Neue'; font-size:2.6rem; color:#fff; line-height:1;}
.ticket .stub .small {font-size:.75rem; letter-spacing:2px; opacity:.75; text-transform:uppercase;}
.ticket .stub .emoji {font-size:2rem;}

.meter {height: 12px; border-radius: 999px; background: linear-gradient(90deg,#e50914,#f5c518,#22c55e);
        position: relative; margin: 1.2rem 0 .3rem;}
.meter .dot {position:absolute; top:-6px; width:24px; height:24px; border-radius:50%;
             background:#fff; border:4px solid #0b0b10; transform:translateX(-50%);
             box-shadow:0 0 0 2px var(--gold), 0 2px 10px rgba(0,0,0,.6);}
.meter-labels {display:flex; justify-content:space-between; font-size:.8rem; opacity:.7;}

.hl {padding: 1px 5px; border-radius: 6px;}
.review-box {
  border:1px solid rgba(245,197,24,.25); background: rgba(255,255,255,.04);
  border-radius:14px; padding:1rem 1.2rem; line-height:2; font-family:'Playfair Display',serif; font-size:1.05rem;
}
[data-testid="stMetric"] {
  background: var(--card); border:1px solid rgba(245,197,24,.2); border-radius: 14px; padding: .8rem 1rem;
}
[data-testid="stMetricValue"] {color: var(--gold); font-family:'Bebas Neue'; letter-spacing:1px;}
[data-testid="stFileUploader"] section {background: rgba(255,255,255,.04); border:1px dashed rgba(245,197,24,.4); border-radius:14px;}

@media (max-width: 640px) {
  .stApp::before, .stApp::after {display:none;}
  .hero h1 {font-size: 2.4rem;}
  .ticket {flex-direction: column;}
  .ticket .stub {width:auto; border-left:none; border-top:3px dashed rgba(255,255,255,.35);}
}
</style>
<div class="hero">
  <div class="kicker">NOW SHOWING</div>
  <h1>REVIEW CRITIC AI</h1>
  <p>Drop in a movie review &mdash; our critic tells you if it&rsquo;s a standing ovation or a walk-out.</p>
  <div class="stars">&#9733; &#9733; &#9733; &#9733; &#9733;</div>
</div>
""",
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file **{MODEL_PATH.name}** not found. Put it in the same folder as app.py.")
    st.stop()

model = load_model(MODEL_PATH)

if "history" not in st.session_state:
    st.session_state.history = []
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# Sidebar -------------------------------------------------------------------
with st.sidebar:
    threshold = st.slider("Positive threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Reviews with a positive probability above this are labelled Positive.")
    show_hl = st.toggle("Highlight key words", value=True)
    if st.session_state.history and st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()
    if NLTK_MISSING:
        st.warning("`nltk` isn't installed — stemming is skipped and accuracy will drop. Run `pip install nltk`.")

# Tabs ----------------------------------------------------------------------
single_tab, batch_tab, history_tab = st.tabs(["🎬 Single review", "🎞️ Batch (CSV)", "🕘 History"])

with single_tab:
    st.markdown("**Try an example**")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state.review_text = text

    review = st.text_area(
        "Your review",
        key="review_text",
        height=150,
        placeholder="e.g. The movie was fantastic — great acting and a gripping story!",
    )
    st.caption(f"{len(review.split())} words")

    analyze = st.button("🎬 ROLL THE VERDICT", type="primary", use_container_width=True)

    if analyze:
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing…"):
                _, probs = predict(model, [review])
            p_pos = float(probs[0][pos_index(model)])
            is_pos = p_pos >= threshold
            confidence = p_pos if is_pos else 1 - p_pos

            stars = max(1, min(5, round(p_pos * 4) + 1))  # 1..5 from probability
            star_html = "".join("★" if i < stars else '<span class="off">★</span>' for i in range(5))
            quote = (
                "A triumph! The audience is on its feet." if p_pos >= 0.85 else
                "Thumbs up — a crowd-pleaser." if is_pos and p_pos >= 0.65 else
                "Leans positive, but not a clear-cut hit." if is_pos else
                "Mixed to poor — leaning thumbs down." if p_pos >= 0.35 else
                "A flop. Walk-outs reported."
            )

            st.markdown(
                f"""
<div class="ticket {'pos' if is_pos else 'neg'}">
  <div class="main">
    <div class="tag">CRITIC'S VERDICT</div>
    <div class="verdict">{'POSITIVE' if is_pos else 'NEGATIVE'}</div>
    <div class="stars-big">{star_html}</div>
    <div class="quote">&ldquo;{quote}&rdquo;</div>
  </div>
  <div class="stub">
    <div class="emoji">{'🍿' if is_pos else '🍅'}</div>
    <div class="big">{confidence:.0%}</div>
    <div class="small">confidence</div>
  </div>
</div>
<div class="meter"><div class="dot" style="left:{p_pos * 100:.1f}%"></div></div>
<div class="meter-labels"><span>Thumbs down</span><span>Mixed</span><span>Thumbs up</span></div>
""",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            c1.metric("Positive probability", f"{p_pos:.1%}")
            c2.metric("Negative probability", f"{1 - p_pos:.1%}")

            if show_hl:
                st.markdown("#### 🔎 What swayed the critic")
                st.markdown(
                    f'<div class="review-box">{highlight_words(model, review)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("🟩 pushes towards positive · 🟥 pushes towards negative · darker = stronger")

            st.session_state.history.insert(0, {
                "Review": review[:120] + ("…" if len(review) > 120 else ""),
                "Sentiment": "Positive" if is_pos else "Negative",
                "Positive prob.": round(p_pos, 3),
            })

with batch_tab:
    st.write("Upload a CSV with a column of reviews to classify them all at once.")
    file = st.file_uploader("Drop your CSV here", type="csv")
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read that file: {e}")
            st.stop()
        text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
        col = st.selectbox("Column with the review text", text_cols)
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Classify all reviews", type="primary", use_container_width=True):
            texts = df[col].fillna("").astype(str).tolist()
            with st.spinner(f"Classifying {len(texts):,} reviews…"):
                _, probs = predict(model, texts)
            p = probs[:, pos_index(model)]
            out = df.copy()
            out["sentiment"] = np.where(p >= threshold, "Positive", "Negative")
            out["positive_probability"] = p.round(4)

            n_pos = int((out["sentiment"] == "Positive").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", f"{len(out):,}")
            m2.metric("Positive", f"{n_pos:,}", f"{n_pos / max(len(out), 1):.0%}")
            m3.metric("Negative", f"{len(out) - n_pos:,}", f"{(len(out) - n_pos) / max(len(out), 1):.0%}", delta_color="inverse")

            st.bar_chart(out["sentiment"].value_counts(), color="#f5c518")
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv", use_container_width=True)

with history_tab:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    else:
        st.info("Your analyzed reviews will show up here.")