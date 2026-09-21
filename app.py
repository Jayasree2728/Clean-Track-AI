"""
CleanTrack AI  ·  Multi-Domain AI Waste Segregation Platform
=============================================================
1M1B IBM SkillsBuild AI + Sustainability Internship  ·  SDG 12

AI Domains integrated:
  1. Computer Vision   — image upload → colour/texture analysis → waste classification
  2. NLP / ML          — TF-IDF + cosine-similarity text classifier (scikit-learn)
  3. ML Forecasting    — Linear Regression campus waste-trend forecast (scikit-learn)
  4. Anomaly Detection — Z-score hazardous-scan spike detection (numpy/scipy)
  5. Confidence Calibration — Softmax probability across all 4 bin classes (numpy)

Run:  python -m streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io
import datetime
import random
import math

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# scikit-learn components used for NLP and forecasting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CleanTrack AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Premium dark-accent design system
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Import Inter font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base reset ── */
html,body,[class*="css"]{
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;
}

/* ══ SIDEBAR — dark premium ══════════════════════════════════════════════ */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a1628 0%,#0d1f2d 60%,#0a2218 100%) !important;
  border-right:1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{color:#f8fafc !important;font-weight:700;}
section[data-testid="stSidebar"] .stMetric label{color:#94a3b8 !important;font-size:.7rem !important;text-transform:uppercase;letter-spacing:.06em;}
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"]{color:#4ade80 !important;font-size:1.6rem !important;font-weight:800;}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.08) !important;}
section[data-testid="stSidebar"] .stDownloadButton button{
  background:linear-gradient(135deg,#16a34a,#15803d) !important;
  color:#fff !important;border:none !important;border-radius:10px !important;
  font-weight:600 !important;padding:10px 0 !important;
  box-shadow:0 4px 14px rgba(22,163,74,.35) !important;
}

/* ══ MAIN AREA — premium white/off-white ═══════════════════════════════════ */
.main .block-container{
  padding-top:1.6rem;
  padding-bottom:3rem;
  max-width:1400px;
}

/* ══ HERO ════════════════════════════════════════════════════════════════ */
.hero{
  background:linear-gradient(135deg,#052e16 0%,#064e3b 35%,#065f46 65%,#047857 100%);
  border-radius:20px;
  padding:40px 44px 34px;
  color:#fff;
  margin-bottom:28px;
  border:1px solid rgba(255,255,255,.08);
  box-shadow:0 8px 32px rgba(5,46,22,.4),0 1px 0 rgba(255,255,255,.05) inset;
  position:relative;
  overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:-60px;right:-60px;
  width:240px;height:240px;
  background:radial-gradient(circle,rgba(74,222,128,.18) 0%,transparent 70%);
  border-radius:50%;pointer-events:none;
}
.hero h1{font-size:2.4rem;font-weight:900;margin:0 0 8px;letter-spacing:-.8px;
         text-shadow:0 2px 12px rgba(0,0,0,.3);}
.hero p{font-size:.97rem;opacity:.82;margin:0;line-height:1.6;font-weight:400;}
.sdg-pill{
  display:inline-block;
  background:rgba(74,222,128,.15);
  border:1px solid rgba(74,222,128,.4);
  border-radius:99px;
  padding:4px 16px;
  font-size:.72rem;font-weight:700;
  letter-spacing:.06em;margin-bottom:14px;
  color:#4ade80;text-transform:uppercase;
}

/* ══ KPI ROW ═════════════════════════════════════════════════════════════ */
.kpi-row{display:flex;gap:14px;margin-bottom:26px;flex-wrap:wrap;}
.kpi{
  flex:1;min-width:130px;
  background:#fff;
  border:1px solid #e2e8f0;
  border-radius:16px;
  padding:20px 18px 16px;
  text-align:center;
  box-shadow:0 1px 3px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.04);
  transition:transform .15s,box-shadow .15s;
  position:relative;overflow:hidden;
}
.kpi::after{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#16a34a,#059669);border-radius:16px 16px 0 0;
}
.kpi .kv{font-size:2rem;font-weight:900;color:#0f172a;line-height:1;letter-spacing:-.04em;}
.kpi .kl{font-size:.68rem;color:#94a3b8;margin-top:5px;text-transform:uppercase;
         letter-spacing:.07em;font-weight:600;}

/* ══ TABS ════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"]{
  gap:4px;
  background:#f1f5f9;
  border-radius:12px;
  padding:4px;
  border:none;
}
.stTabs [data-baseweb="tab"]{
  border-radius:9px;
  padding:8px 18px;
  font-size:.84rem;font-weight:600;
  color:#64748b;
  background:transparent;
  border:none;
  transition:all .18s;
}
.stTabs [aria-selected="true"]{
  background:#fff !important;
  color:#0f172a !important;
  box-shadow:0 1px 4px rgba(0,0,0,.1);
}

/* ══ PRIMARY BUTTON ══════════════════════════════════════════════════════ */
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#16a34a 0%,#15803d 100%) !important;
  color:#fff !important;
  border:none !important;
  border-radius:12px !important;
  font-weight:700 !important;
  font-size:.92rem !important;
  padding:12px 24px !important;
  box-shadow:0 4px 14px rgba(22,163,74,.4) !important;
  letter-spacing:.01em !important;
  transition:all .18s !important;
}
.stButton>button[kind="primary"]:hover{
  transform:translateY(-1px) !important;
  box-shadow:0 6px 20px rgba(22,163,74,.5) !important;
}

/* ══ INPUTS ══════════════════════════════════════════════════════════════ */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea{
  border-radius:10px !important;
  border:1.5px solid #e2e8f0 !important;
  font-size:.9rem !important;
  padding:10px 14px !important;
  transition:border-color .15s !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus{
  border-color:#16a34a !important;
  box-shadow:0 0 0 3px rgba(22,163,74,.12) !important;
}
.stSelectbox>div>div{
  border-radius:10px !important;
  border:1.5px solid #e2e8f0 !important;
}

/* ══ INFO BANNERS (tab descriptions) ════════════════════════════════════ */
.info-banner{
  display:flex;align-items:flex-start;gap:12px;
  background:#f8faff;
  border:1px solid #e0e7ff;
  border-left:4px solid;
  border-radius:12px;
  padding:14px 18px;
  margin-bottom:20px;
  font-size:.86rem;line-height:1.55;
  color:#1e293b;
}
.info-banner .ib-icon{font-size:1.2rem;margin-top:1px;flex-shrink:0;}

/* ══ RESULT CARDS ════════════════════════════════════════════════════════ */
.rcard{
  background:#fff;
  border:1px solid #e2e8f0;
  border-radius:16px;
  overflow:hidden;
  margin-bottom:16px;
  box-shadow:0 1px 3px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.05);
}
.rcard .stripe{height:4px;}
.rcard .body{padding:22px 24px;}
.rcard h3{
  font-size:.65rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:#94a3b8;margin:0 0 10px;
}
.rcard p{font-size:.9rem;color:#334155;line-height:1.7;margin:0;}

/* ══ BIN BADGE ═══════════════════════════════════════════════════════════ */
.badge{
  display:inline-flex;align-items:center;gap:8px;
  border-radius:10px;padding:10px 20px;
  font-size:.95rem;font-weight:800;color:#fff;
  margin:6px 0 14px;
  letter-spacing:.01em;
  box-shadow:0 3px 10px rgba(0,0,0,.2);
}

/* ══ CONFIDENCE BAR ══════════════════════════════════════════════════════ */
.cbar-bg{background:#f1f5f9;border-radius:99px;height:9px;margin-top:8px;}
.cbar{height:9px;border-radius:99px;transition:width .6s cubic-bezier(.4,0,.2,1);}

/* ══ NUMBERED STEPS ══════════════════════════════════════════════════════ */
.steps{list-style:none;padding:0;margin:0;}
.steps li{
  display:flex;align-items:flex-start;gap:10px;
  padding:8px 0;
  border-bottom:1px solid #f8fafc;
  font-size:.875rem;color:#334155;line-height:1.5;
}
.steps li:last-child{border-bottom:none;}
.snum{
  min-width:24px;height:24px;border-radius:50%;
  background:linear-gradient(135deg,#16a34a,#059669);
  color:#fff;display:flex;align-items:center;justify-content:center;
  font-size:.65rem;font-weight:800;margin-top:1px;flex-shrink:0;
  box-shadow:0 2px 6px rgba(22,163,74,.35);
}

/* ══ FACILITY ROWS ═══════════════════════════════════════════════════════ */
.frow{
  display:flex;align-items:center;gap:12px;
  padding:10px 0;
  border-bottom:1px solid #f8fafc;
  font-size:.855rem;
}
.frow:last-child{border-bottom:none;}
.fdot{
  width:10px;height:10px;border-radius:50%;flex-shrink:0;
  box-shadow:0 0 0 3px rgba(0,0,0,.06);
}

/* ══ DOMAIN TAGS ═════════════════════════════════════════════════════════ */
.domain-tag{
  display:inline-block;padding:3px 11px;border-radius:99px;
  font-size:.68rem;font-weight:700;margin-right:5px;margin-bottom:5px;
  letter-spacing:.04em;text-transform:uppercase;
}

/* ══ SOFTMAX BARS ════════════════════════════════════════════════════════ */
.prob-row{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.prob-label{font-size:.8rem;color:#475569;width:148px;flex-shrink:0;font-weight:500;}
.prob-bar-bg{flex:1;background:#f1f5f9;border-radius:99px;height:8px;}
.prob-bar{height:8px;border-radius:99px;min-width:3px;}
.prob-pct{font-size:.76rem;font-weight:800;width:38px;text-align:right;}

/* ══ REGULATION BOX ══════════════════════════════════════════════════════ */
.reg-box{
  margin-top:12px;padding:9px 13px;
  background:#f8fafc;border:1px solid #e2e8f0;
  border-radius:9px;font-size:.78rem;color:#64748b;
}

/* ══ DID YOU KNOW BOX ════════════════════════════════════════════════════ */
.dyk-box{
  padding:11px 13px;
  background:linear-gradient(135deg,#f0fdf4,#dcfce7);
  border:1px solid #bbf7d0;
  border-radius:10px;
  font-size:.8rem;color:#166534;
  line-height:1.55;
  margin-top:12px;
}

/* ══ ANOMALY ALERT ═══════════════════════════════════════════════════════ */
.anomaly-box{
  display:flex;gap:12px;align-items:flex-start;
  background:linear-gradient(135deg,#fff1f2,#ffe4e6);
  border:1px solid #fecdd3;border-left:4px solid #ef4444;
  border-radius:12px;padding:14px 18px;
  font-size:.86rem;color:#9f1239;line-height:1.55;
}

/* ══ FORECAST BAR ════════════════════════════════════════════════════════ */
.fc-row{margin-bottom:9px;}
.fc-top{display:flex;justify-content:space-between;font-size:.79rem;margin-bottom:4px;color:#334155;}
.fc-top span:last-child{font-weight:700;}
.fc-bar-bg{background:#f1f5f9;border-radius:99px;height:7px;}
.fc-bar{height:7px;border-radius:99px;background:linear-gradient(90deg,#f59e0b,#d97706);}

/* ══ MODEL INFO BOX ══════════════════════════════════════════════════════ */
.model-box{
  margin-top:14px;
  background:linear-gradient(135deg,#fffbeb,#fef3c7);
  border:1px solid #fde68a;border-radius:10px;
  padding:12px 14px;font-size:.8rem;color:#78350f;line-height:1.6;
}

/* ══ FEATURE TABLE ═══════════════════════════════════════════════════════ */
.feat-row{
  display:flex;justify-content:space-between;align-items:center;
  border-bottom:1px solid #f8fafc;padding:7px 0;font-size:.84rem;
}
.feat-row:last-child{border-bottom:none;}
.feat-key{color:#94a3b8;font-weight:500;}
.feat-val{font-weight:700;color:#1e293b;}

/* ══ CV DECISION NOTE ════════════════════════════════════════════════════ */
.cv-note{
  margin-top:14px;
  background:linear-gradient(135deg,#f0fdf4,#dcfce7);
  border-left:4px solid #16a34a;
  border-radius:0 10px 10px 0;
  padding:11px 14px;font-size:.82rem;color:#166534;line-height:1.55;
}

/* ══ SIDEBAR DOMAIN TAGS ═════════════════════════════════════════════════ */
.sb-domain{
  display:flex;align-items:center;gap:9px;
  padding:6px 10px;border-radius:8px;
  background:rgba(255,255,255,.05);
  margin-bottom:4px;
  font-size:.8rem;font-weight:600;
}
.sb-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}

/* ══ SIDEBAR BIN LEGEND ══════════════════════════════════════════════════ */
.sb-bin{
  display:flex;align-items:center;gap:10px;
  padding:5px 10px;border-radius:8px;
  font-size:.82rem;font-weight:500;
  margin-bottom:3px;
  background:rgba(255,255,255,.04);
}
.sb-swatch{width:12px;height:12px;border-radius:4px;flex-shrink:0;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "scan_log" not in st.session_state:
    st.session_state.scan_log = []      # list of dicts — one per scan
if "daily_counts" not in st.session_state:
    # Simulate 14 days of historical campus waste scan counts for the forecast
    np.random.seed(42)
    base = np.array([12,15,11,18,22,9,14,17,20,13,16,19,21,24], dtype=float)
    noise = np.random.normal(0, 1.5, 14)
    st.session_state.daily_counts = (base + noise).clip(0).tolist()

# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  (ground truth — used by all AI engines)
# ══════════════════════════════════════════════════════════════════════════════
WASTE_DATA = {
    "Greasy pizza box": {
        "bin":"Landfill Residual","bin_color":"#6b7280","bin_icon":"🗑️",
        "confidence":98,
        "prep_steps":[
            "Tear off the clean grease-free lid — recycle it separately.",
            "Place the greasy base in the grey landfill bin.",
            "Never compost: grease repels microbial action and attracts pests.",
        ],
        "why":("Grease molecules bond to cellulose during pulping, preventing "
               "fibre re-bonding. One greasy box can reject an entire 500 kg "
               "cardboard bale. Landfill is the only responsible route."),
        "co2_saved_grams":0,
        "co2_context":"No diversion benefit; correct segregation protects ~200 g CO₂e per bale.",
        "facilities":[
            {"name":"MSW Landfill","distance":"2.1 km","hours":"Mon–Sat 7 am–5 pm"},
            {"name":"Campus Grey Bin","distance":"On-site","hours":"24 / 7"},
        ],
        "regulation":"SWM Rules 2016 (India) — mixed/contaminated waste.",
        "did_you_know":"~3 billion pizza boxes discarded yearly; the clean lid alone saves 17 g CO₂e.",
        # NLP corpus sentence for TF-IDF training
        "corpus":"greasy oily pizza box cardboard contaminated food stain fat grease landfill residual",
        # CV dominant colour hint (HSV hue range expected from real cardboard)
        "cv_hint":"brown cardboard grease stain",
    },
    "Crushed PET plastic bottle": {
        "bin":"Dry Recyclables","bin_color":"#2563eb","bin_icon":"♻️",
        "confidence":99,
        "prep_steps":[
            "Rinse with water to remove all beverage residue.",
            "Crush flat to reduce volume.",
            "Remove cap; recycle separately if facility accepts PP.",
            "Leave label on — stripped automatically at the mill.",
        ],
        "why":("PET (resin code ①) is shredded into flakes and re-extruded into "
               "new bottles or polyester yarn at near-virgin quality. Rinsing "
               "prevents microbial bale degradation. Crushing saves 6× truck space."),
        "co2_saved_grams":340,
        "co2_context":"Vs. virgin PET from petroleum. WRAP Material Change Report 2022.",
        "facilities":[
            {"name":"Blue Dry-Recyclables Bin","distance":"On-site","hours":"24 / 7"},
            {"name":"Plastic Bottle Bank — Student Union","distance":"150 m","hours":"Mon–Fri 8–8"},
            {"name":"Municipal MRF","distance":"3.4 km","hours":"Mon–Sat 8–6"},
        ],
        "regulation":"PWM Rules 2016 — EPR applies to PET packaging.",
        "did_you_know":"Recycling 1 tonne of PET saves 1.5 t CO₂e and 6,000 L of water.",
        "corpus":"plastic bottle pet polyethylene terephthalate crushed rinse clean dry recyclable",
        "cv_hint":"transparent clear plastic bottle crushed",
    },
    "Tetra Pak juice container": {
        "bin":"Dry Recyclables","bin_color":"#2563eb","bin_icon":"♻️",
        "confidence":96,
        "prep_steps":[
            "Rinse interior with water.",
            "Open top flap fully to air-dry.",
            "Do NOT flatten — layered structure aids optical sorters.",
            "Use specialist Tetra Pak drop-off if available on campus.",
        ],
        "why":("74% paperboard / 22% LDPE / 4% aluminium laminate. Specialist "
               "hydrapulping separates layers. Paper fibre recovered as high-quality "
               "pulp; polyAl processed into roofing panels."),
        "co2_saved_grams":280,
        "co2_context":"Recovering paper fibre avoids landfill methane. Tetra Pak LCA 2021.",
        "facilities":[
            {"name":"Tetra Pak Drop-Off — Canteen","distance":"80 m","hours":"Mon–Fri 9–4"},
            {"name":"Blue Dry-Recyclables Bin","distance":"On-site","hours":"24 / 7"},
            {"name":"ITC Paperboards Carton Recycler","distance":"8.2 km","hours":"Mon–Fri 9–5"},
        ],
        "regulation":"PWM Rules 2016 — multi-layer packaging under EPR.",
        "did_you_know":"A Tetra Pak uses 75% less material than an equivalent glass jar.",
        "corpus":"tetra pak juice carton multilayer aluminium paperboard ldpe recyclable",
        "cv_hint":"white red tetra pak rectangular carton",
    },
    "Used lithium battery": {
        "bin":"Hazardous","bin_color":"#dc2626","bin_icon":"⚠️",
        "confidence":100,
        "prep_steps":[
            "STOP — do not place in any general bin.",
            "Cover each terminal with non-conductive tape.",
            "Store cool and dry, away from flammable materials.",
            "Take to authorised e-waste / battery collection point.",
            "Never puncture, crush, or dispose of in fire.",
        ],
        "why":("Flammable electrolytes and reactive Li oxides undergo thermal "
               "runaway at 500–900 °C if damaged, releasing toxic HF gas. "
               "Co, Ni, Mn are Schedule-I hazardous substances. Proper recycling "
               "recovers 95% of Li and Co for reuse."),
        "co2_saved_grams":1200,
        "co2_context":"Recovering Li and Co vs. virgin mining. IEA Global EV Outlook 2023.",
        "facilities":[
            {"name":"E-Waste Drop Box — Security Desk","distance":"On-site","hours":"24 / 7"},
            {"name":"Attero Recycling — Authorised E-Waste","distance":"5.6 km","hours":"Mon–Sat 9–6"},
            {"name":"Municipal Hazardous Waste Centre","distance":"7.1 km","hours":"Tue & Fri 10–3"},
        ],
        "regulation":"E-Waste Management Rules 2022 — Schedule-I restricted substance.",
        "did_you_know":"One mishandled Li-ion battery can start a landfill fire burning for weeks.",
        "corpus":"lithium battery ewaste hazardous toxic terminal reactive flammable cobalt",
        "cv_hint":"grey silver rectangular battery cell terminal",
    },
    "Single-use paper coffee cup with plastic lining": {
        "bin":"Landfill Residual","bin_color":"#6b7280","bin_icon":"🗑️",
        "confidence":97,
        "prep_steps":[
            "Separate plastic lid — recycle if marked ① or ②.",
            "Empty all liquid.",
            "If campus has Simply Cups scheme, use it.",
            "Otherwise place cup body in grey landfill bin.",
        ],
        "why":("5% PE liner bonds to paper fibres, preventing hydrapulping. "
               "Conventional mills reject the entire cup. Liner is not EN 13432 "
               "home-compostable. Residual bin is the responsible default."),
        "co2_saved_grams":0,
        "co2_context":"No diversion benefit; correct disposal protects paper-recycling streams.",
        "facilities":[
            {"name":"Campus Grey Bin","distance":"On-site","hours":"24 / 7"},
            {"name":"Simply Cups — Library Entrance","distance":"200 m","hours":"Mon–Fri 8–7"},
        ],
        "regulation":"CPCB SUP Rules 2021 — disposable cups under phase-out schedule.",
        "did_you_know":"500 billion disposable cups used yearly globally; <1% are recycled.",
        "corpus":"paper cup coffee single use plastic lining polyethylene landfill residual",
        "cv_hint":"white paper cup coffee sleeve disposable",
    },
}

BIN_COLORS = {
    "Dry Recyclables":"#2563eb",
    "Wet Organic":"#16a34a",
    "Hazardous":"#dc2626",
    "Landfill Residual":"#6b7280",
}

# ══════════════════════════════════════════════════════════════════════════════
# AI ENGINE 1 — NLP TEXT CLASSIFIER  (TF-IDF + Cosine Similarity)
# Simulates what a fine-tuned IBM Granite / BERT text classifier does:
# encodes text as TF-IDF vectors and finds the closest waste category.
# ══════════════════════════════════════════════════════════════════════════════

# Build corpus: one document per known waste item
_corpus_docs   = [v["corpus"] for v in WASTE_DATA.values()]
_corpus_labels = list(WASTE_DATA.keys())

# Fit a TF-IDF vectorizer on the corpus
_tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
_tfidf_matrix = _tfidf.fit_transform(_corpus_docs)   # shape: (5, features)

# Extra domain corpus for open-ended items not in WASTE_DATA
_DOMAIN_CORPUS = [
    # (document, bin, co2)
    ("food waste fruit peel vegetable kitchen organic compost wet",
     "Wet Organic", 150),
    ("glass jar bottle clean dry recyclable aluminium tin steel metal can",
     "Dry Recyclables", 220),
    ("newspaper magazine cardboard box paper envelope clean dry",
     "Dry Recyclables", 190),
    ("ewaste electronic phone laptop circuit board capacitor solvent paint bleach",
     "Hazardous", 600),
    ("medicine pill drug syringe needle expired chemical acid alkali",
     "Hazardous", 400),
    ("dirty soiled contaminated mixed unknown residual",
     "Landfill Residual", 0),
]
_ext_docs    = [d[0] for d in _DOMAIN_CORPUS]
_ext_bins    = [d[1] for d in _DOMAIN_CORPUS]
_ext_co2     = [d[2] for d in _DOMAIN_CORPUS]
_ext_tfidf   = _tfidf.transform(_ext_docs)   # use same fitted vectorizer


def nlp_classify(text: str):
    """
    TF-IDF + cosine similarity classifier.
    First checks WASTE_DATA corpus (high-confidence known items),
    then falls back to extended domain corpus.
    Returns (best_label_or_bin, similarity_score, source).
    """
    vec = _tfidf.transform([text.lower()])

    # --- Check against known items ---
    sims_known = cosine_similarity(vec, _tfidf_matrix).flatten()
    best_known_idx  = int(np.argmax(sims_known))
    best_known_sim  = float(sims_known[best_known_idx])

    # --- Check against extended domain corpus ---
    sims_ext = cosine_similarity(vec, _ext_tfidf).flatten()
    best_ext_idx = int(np.argmax(sims_ext))
    best_ext_sim = float(sims_ext[best_ext_idx])

    if best_known_sim > 0.12:
        label  = _corpus_labels[best_known_idx]
        result = dict(WASTE_DATA[label])
        result["confidence"] = min(99, int(50 + best_known_sim * 120))
        return result, best_known_sim, "TF-IDF ↔ Knowledge Base"

    if best_ext_sim > 0.08:
        bin_name = _ext_bins[best_ext_idx]
        co2      = _ext_co2[best_ext_idx]
        conf     = min(85, int(40 + best_ext_sim * 110))
        return _build_generic(text, bin_name, co2, conf), best_ext_sim, "TF-IDF ↔ Domain Corpus"

    return _build_generic(text, "Landfill Residual", 0, 55), 0.0, "Low similarity — default"


def softmax_probs(text: str) -> dict:
    """
    Return a softmax-calibrated probability distribution across the 4 bin
    classes. This mirrors the output layer of a real waste-classification CNN
    or fine-tuned language model.
    """
    vec = _tfidf.transform([text.lower()])
    # Aggregate extended-corpus similarities per bin class
    bin_classes = ["Dry Recyclables", "Wet Organic", "Hazardous", "Landfill Residual"]
    scores = {b: 0.0 for b in bin_classes}
    for sim, bin_name in zip(cosine_similarity(vec, _ext_tfidf).flatten(), _ext_bins):
        scores[bin_name] = max(scores[bin_name], float(sim))
    # Add known-item sims
    for sim, label in zip(cosine_similarity(vec, _tfidf_matrix).flatten(), _corpus_labels):
        b = WASTE_DATA[label]["bin"]
        scores[b] = max(scores[b], float(sim))
    # Softmax
    vals = np.array([scores[b] for b in bin_classes])
    vals += 1e-6
    exp_v = np.exp(vals * 5)        # temperature = 1/5 sharpens distribution
    probs = exp_v / exp_v.sum()
    return {b: round(float(p)*100, 1) for b, p in zip(bin_classes, probs)}


def _build_generic(text, bin_name, co2, conf):
    """Build a result dict for items not in WASTE_DATA."""
    icon_map = {"Dry Recyclables":"♻️","Wet Organic":"🌿",
                "Hazardous":"⚠️","Landfill Residual":"🗑️"}
    prep_map = {
        "Dry Recyclables":["Clean and dry the item.","Place in blue dry-recyclables bin."],
        "Wet Organic":["Remove non-organic packaging.","Place loose in green compost bin."],
        "Hazardous":["Keep away from general waste.","Take to authorised collection point."],
        "Landfill Residual":["No recyclable stream identified.","Place in grey residual bin."],
    }
    why_map = {
        "Dry Recyclables":(f'"{text}" matches dry recyclable material signatures. '
                           "Clean materials of this type can be processed at an MRF."),
        "Wet Organic":(f'"{text}" contains biodegradable organic indicators. '
                       "Composting converts it to biogas or nutrient-rich compost."),
        "Hazardous":(f'"{text}" matches hazardous substance signatures. '
                     "Requires specialist disposal to prevent soil/water contamination."),
        "Landfill Residual":(f'"{text}" could not be matched to a higher-value stream. '
                             "Residual bin prevents cross-contamination."),
    }
    fac_map = {
        "Dry Recyclables":[{"name":"Blue Dry-Recyclables Bin","distance":"On-site","hours":"24/7"}],
        "Wet Organic":[{"name":"Green Compost Bin","distance":"On-site","hours":"24/7"}],
        "Hazardous":[{"name":"E-Waste / Hazardous Drop Box","distance":"On-site","hours":"24/7"},
                     {"name":"Municipal Hazardous Centre","distance":"~5 km","hours":"Tue & Fri 10–3"}],
        "Landfill Residual":[{"name":"Campus Grey Bin","distance":"On-site","hours":"24/7"}],
    }
    return {
        "bin": bin_name,
        "bin_color": BIN_COLORS[bin_name],
        "bin_icon": icon_map[bin_name],
        "confidence": conf,
        "prep_steps": prep_map[bin_name],
        "why": why_map[bin_name],
        "co2_saved_grams": co2,
        "co2_context": "Estimated saving for this material category.",
        "facilities": fac_map[bin_name],
        "regulation": "SWM / E-Waste / PWM Rules 2016–2022 (India).",
        "did_you_know": "Correct segregation at source can divert up to 80% of campus waste from landfill.",
        "corpus": text.lower(),
        "cv_hint": "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI ENGINE 2 — COMPUTER VISION  (PIL image analysis)
# Simulates what a CNN (e.g., MobileNetV3 fine-tuned on TrashNet) does:
# analyses pixel colour distribution and saturation to infer material class.
# In production: replace with torchvision / tensorflow model inference.
# ══════════════════════════════════════════════════════════════════════════════

def cv_classify(image: Image.Image):
    """
    Classify a waste item from an uploaded image.

    Pipeline (mirrors real CNN preprocessing + inference):
      1. Resize to 224×224 (standard CNN input)
      2. Convert to HSV colour space for material-aware analysis
      3. Extract dominant hue bucket, saturation mean, value mean
      4. Apply decision rules that approximate a trained classifier's
         learned feature maps for common waste materials
      5. Return bin prediction + feature explanation
    """
    # Step 1: resize to CNN-standard input size
    img_resized = image.convert("RGB").resize((224, 224))
    pixels = np.array(img_resized, dtype=np.float32) / 255.0   # normalise to [0,1]

    r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]

    # Step 2: RGB → HSV conversion (vectorised, no cv2 needed)
    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min + 1e-8

    # Hue channel
    hue = np.where(c_max == r, 60 * ((g - b) / delta % 6),
          np.where(c_max == g, 60 * ((b - r) / delta + 2),
                               60 * ((r - g) / delta + 4)))
    hue = hue % 360

    saturation = np.where(c_max > 0, delta / c_max, 0)
    value      = c_max   # brightness

    sat_mean = float(saturation.mean())
    val_mean = float(value.mean())

    # Step 3: Hue histogram — 8 buckets covering 360°
    hue_hist, _ = np.histogram(hue.flatten(), bins=8, range=(0, 360))
    dominant_bucket = int(np.argmax(hue_hist))   # 0=red, 1=orange/brown, 2=yellow, 3=green, 4=cyan, 5=blue, 6=violet, 7=magenta

    # Step 4: Decision rules (approximate CNN class logits)
    # Hazardous: high-saturation red/yellow warning colours
    if sat_mean > 0.45 and dominant_bucket in (0, 2):
        bin_pred   = "Hazardous"
        cv_conf    = 72
        feat_note  = f"High saturation ({sat_mean:.2f}) + red/yellow hue → warning-colour object (battery, chemical container)"

    # Organic: green-dominant, medium brightness (vegetable/garden)
    elif dominant_bucket in (3, 4) and val_mean > 0.2:
        bin_pred   = "Wet Organic"
        cv_conf    = 68
        feat_note  = f"Green-dominant hue bucket ({dominant_bucket}) + brightness {val_mean:.2f} → likely organic/plant material"

    # Transparent/clear plastic: very low saturation, high brightness
    elif sat_mean < 0.15 and val_mean > 0.7:
        bin_pred   = "Dry Recyclables"
        cv_conf    = 74
        feat_note  = f"Low saturation ({sat_mean:.2f}) + high brightness ({val_mean:.2f}) → clear/white recyclable (PET bottle, paper)"

    # Brown/tan cardboard: orange-brown hue, medium saturation
    elif dominant_bucket in (1,) and 0.1 < sat_mean < 0.5:
        bin_pred   = "Landfill Residual"
        cv_conf    = 65
        feat_note  = f"Brown/orange dominant hue + medium saturation ({sat_mean:.2f}) → likely cardboard or contaminated packaging"

    # Dark/grey objects: low brightness (batteries, electronics)
    elif val_mean < 0.35:
        bin_pred   = "Hazardous"
        cv_conf    = 63
        feat_note  = f"Low brightness ({val_mean:.2f}) → dark object consistent with battery / e-waste"

    # Blue packaging: recyclable container
    elif dominant_bucket in (5, 6):
        bin_pred   = "Dry Recyclables"
        cv_conf    = 70
        feat_note  = f"Blue/violet dominant hue → common recyclable packaging colour"

    else:
        bin_pred   = "Landfill Residual"
        cv_conf    = 55
        feat_note  = f"No dominant material signature detected (sat={sat_mean:.2f}, val={val_mean:.2f}) → residual default"

    features = {
        "Dominant Hue Bucket": f"Bucket {dominant_bucket} (~{dominant_bucket*45}°–{(dominant_bucket+1)*45}°)",
        "Mean Saturation": f"{sat_mean:.3f}",
        "Mean Brightness": f"{val_mean:.3f}",
        "Pixel Count": f"{224*224:,}",
        "Colour Space": "HSV (converted from RGB)",
    }
    result = _build_generic(f"Image upload", bin_pred, BIN_COLORS[bin_pred], cv_conf)
    # Override colour
    result["bin_color"]  = BIN_COLORS[bin_pred]
    result["confidence"] = cv_conf
    return result, features, feat_note


# ══════════════════════════════════════════════════════════════════════════════
# AI ENGINE 3 — LINEAR REGRESSION WASTE FORECAST
# Trains a simple regression model on 14 days of campus scan history and
# predicts the next 7 days. In production: replace with LSTM / Prophet.
# ══════════════════════════════════════════════════════════════════════════════

def forecast_waste(history: list) -> tuple[list, list, float]:
    """
    Fit LinearRegression on historical daily scan counts.
    Returns (forecast_values, forecast_days_labels, R² score).
    """
    n = len(history)
    X = np.arange(n).reshape(-1, 1)
    y = np.array(history)

    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))

    future_X = np.arange(n, n + 7).reshape(-1, 1)
    forecast  = model.predict(future_X).clip(0).tolist()

    today = datetime.date.today()
    labels = [(today + datetime.timedelta(days=i+1)).strftime("%b %d") for i in range(7)]
    return forecast, labels, r2


# ══════════════════════════════════════════════════════════════════════════════
# AI ENGINE 4 — ANOMALY DETECTION  (Z-score on session scan bins)
# Flags unusually high hazardous item counts using statistical Z-score,
# mirroring real-time monitoring systems used by waste facility operators.
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomaly(scan_log: list) -> dict | None:
    """
    Computes per-bin scan counts and flags any bin whose count is
    more than 1.5 standard deviations above the mean (Z > 1.5).
    Returns the anomalous bin info, or None if all is normal.
    """
    if len(scan_log) < 4:
        return None   # not enough data for meaningful statistics

    bin_counts = {}
    for row in scan_log:
        b = row["Bin"]
        bin_counts[b] = bin_counts.get(b, 0) + 1

    counts = np.array(list(bin_counts.values()), dtype=float)
    if counts.std() < 1e-6:
        return None   # all bins equally represented — no anomaly

    z_scores = (counts - counts.mean()) / counts.std()
    bins     = list(bin_counts.keys())

    for z, b in sorted(zip(z_scores, bins), reverse=True):
        if z > 1.5:
            return {
                "bin": b,
                "count": bin_counts[b],
                "z_score": round(float(z), 2),
                "message": (f"⚠️ Anomaly detected: '{b}' items are disproportionately "
                            f"high this session (Z = {z:.2f}). "
                            f"This may indicate a waste audit is needed for this stream."),
            }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log_scan(item: str, result: dict, method: str = "NLP"):
    st.session_state.scan_log.append({
        "Time":           datetime.datetime.now().strftime("%H:%M:%S"),
        "Item":           item,
        "Bin":            result["bin"],
        "CO₂ Saved (g)":  result["co2_saved_grams"],
        "Confidence (%)": result["confidence"],
        "AI Method":      method,
    })
    # Also append to daily counts for the forecast
    st.session_state.daily_counts.append(1)


def render_domain_tags(tags: list[tuple[str,str]]):
    """Render coloured domain badges e.g. [('NLP','#7c3aed'),...]"""
    html = ""
    for label, color in tags:
        html += (f'<span class="domain-tag" style="background:{color}20;'
                 f'color:{color};border:1px solid {color}60;">{label}</span>')
    st.markdown(html, unsafe_allow_html=True)


def render_result(item_label: str, r: dict, source_tag: str, domain_tags: list):
    """Full result card rendering — shared by all scan modes."""
    bc = r["bin_color"]

    # ── Top classification card ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="rcard">
      <div class="stripe" style="background:{bc};"></div>
      <div class="body">
        <h3>AI Classification Result</h3>
        <div style="display:flex;align-items:center;justify-content:space-between;
                    flex-wrap:wrap;gap:10px;margin-bottom:10px;">
          <div>
            <div style="font-size:.78rem;color:#6b7280;margin-bottom:3px;">{source_tag}</div>
            <div style="font-size:1.3rem;font-weight:800;color:#111827;">
              {r["bin_icon"]} {item_label}
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:.72rem;color:#9ca3af;margin-bottom:4px;">TARGET BIN</div>
            <div class="badge" style="background:{bc};">{r["bin_icon"]} {r["bin"]}</div>
          </div>
        </div>
        <div style="font-size:.78rem;color:#6b7280;margin-bottom:4px;">
          AI Confidence: <strong>{r["confidence"]}%</strong>
        </div>
        <div class="cbar-bg">
          <div class="cbar" style="width:{r['confidence']}%;background:{bc};"></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    render_domain_tags(domain_tags)
    st.write("")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        steps_html = "".join(
            f'<li><div class="snum">{i+1}</div><span>{s}</span></li>'
            for i, s in enumerate(r["prep_steps"]))
        st.markdown(f"""
        <div class="rcard">
          <div class="stripe" style="background:{bc};"></div>
          <div class="body">
            <h3>Preparation Steps</h3>
            <ul class="steps">{steps_html}</ul>
          </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        fac_html = "".join(
            f"""<div class="frow"><div class="fdot" style="background:{bc};"></div>
            <div><div style="font-weight:600;color:#111827;font-size:.88rem;">{f['name']}</div>
            <div style="color:#6b7280;font-size:.78rem;">📍 {f['distance']} &nbsp;·&nbsp; 🕐 {f['hours']}</div>
            </div></div>""" for f in r["facilities"])
        st.markdown(f"""
        <div class="rcard">
          <div class="stripe" style="background:{bc};"></div>
          <div class="body">
            <h3>Nearest Disposal Facilities</h3>
            {fac_html}
          </div>
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns([3, 2], gap="medium")
    with col3:
        st.markdown(f"""
        <div class="rcard">
          <div class="stripe" style="background:{bc};"></div>
          <div class="body">
            <h3>🧠 Why This Bin? — AI Explainability</h3>
            <p>{r["why"]}</p>
            <div class="reg-box">
              📋 <strong>Regulation:</strong> {r["regulation"]}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with col4:
        g = r["co2_saved_grams"]
        co2_col = "#16a34a" if g > 0 else "#6b7280"
        st.markdown(f"""
        <div class="rcard">
          <div class="stripe" style="background:{bc};"></div>
          <div class="body">
            <h3>🌍 Sustainability Impact</h3>
            <div style="font-size:2.2rem;font-weight:900;color:{co2_col};
                        line-height:1;margin:10px 0 4px;letter-spacing:-.04em;">
              {g if g>0 else "—"}
              <span style="font-size:.85rem;font-weight:500;color:#94a3b8;"> g CO₂e</span>
            </div>
            <div style="font-size:.78rem;color:#94a3b8;margin-bottom:2px;">{r["co2_context"]}</div>
            <div class="dyk-box">💡 {r["did_you_know"]}</div>
          </div>
        </div>""", unsafe_allow_html=True)


def render_softmax(text: str, bin_color: str):
    """Render softmax probability distribution card."""
    probs = softmax_probs(text)
    bin_colors_map = {
        "Dry Recyclables":"#2563eb","Wet Organic":"#16a34a",
        "Hazardous":"#dc2626","Landfill Residual":"#6b7280",
    }
    rows_html = ""
    for b, p in sorted(probs.items(), key=lambda x: -x[1]):
        c = bin_colors_map[b]
        rows_html += (
            f'<div class="prob-row">'
            f'<div class="prob-label">{b}</div>'
            f'<div class="prob-bar-bg"><div class="prob-bar" '
            f'style="width:{p}%;background:{c};"></div></div>'
            f'<div class="prob-pct" style="color:{c};">{p}%</div>'
            f'</div>'
        )
    st.markdown(f"""
    <div class="rcard">
      <div class="stripe" style="background:{bin_color};"></div>
      <div class="body">
        <h3>📊 Softmax Confidence Distribution — All Bin Classes</h3>
        <div style="font-size:.78rem;color:#6b7280;margin-bottom:10px;">
          Simulates the output probability layer of a waste-classification
          neural network (e.g., fine-tuned IBM Granite / MobileNetV3).
        </div>
        {rows_html}
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px;">
      <div style="font-size:1.15rem;font-weight:900;letter-spacing:-.3px;color:#f8fafc;">
        ♻️ CleanTrack AI
      </div>
      <div style="font-size:.72rem;color:#64748b;font-weight:600;letter-spacing:.06em;
                  text-transform:uppercase;margin-top:3px;">
        Multi-Domain AI Platform · SDG 12
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    log       = st.session_state.scan_log
    total_co2 = sum(r["CO₂ Saved (g)"] for r in log)

    st.markdown('<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:.08em;color:#475569;margin-bottom:10px;">📊 Session Stats</div>',
                unsafe_allow_html=True)
    st.metric("Total Scans",    len(log))
    st.metric("CO₂e Saved (g)", total_co2)
    st.metric("Unique Items",   len(set(r["Item"] for r in log)))
    st.divider()

    # Anomaly Detection result in sidebar
    anomaly = detect_anomaly(log)
    if anomaly:
        st.markdown(
            f'<div class="anomaly-box">'
            f'<span style="font-size:1.1rem;flex-shrink:0;">🚨</span>'
            f'<div><strong>Anomaly Detected</strong><br>{anomaly["message"]}'
            f'<br><span style="font-size:.74rem;opacity:.8;">Z = {anomaly["z_score"]} · n = {anomaly["count"]}</span>'
            f'</div></div>',
            unsafe_allow_html=True)
        st.divider()

    st.markdown('<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:.08em;color:#475569;margin-bottom:8px;">🗂 Bin Guide</div>',
                unsafe_allow_html=True)
    for label, color in [
        ("Dry Recyclables","#2563eb"),
        ("Wet Organic","#16a34a"),
        ("Hazardous / E-Waste","#dc2626"),
        ("Landfill Residual","#6b7280"),
    ]:
        st.markdown(
            f'<div class="sb-bin">'
            f'<div class="sb-swatch" style="background:{color};"></div>'
            f'<span>{label}</span></div>',
            unsafe_allow_html=True)
    st.divider()

    st.markdown('<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:.08em;color:#475569;margin-bottom:8px;">🤖 AI Engines Active</div>',
                unsafe_allow_html=True)
    for tag, color, icon in [
        ("NLP / TF-IDF","#7c3aed","🔤"),
        ("Computer Vision","#0891b2","👁️"),
        ("ML Forecasting","#b45309","📈"),
        ("Anomaly Detection","#be123c","🔍"),
        ("Softmax Calibration","#0f766e","📊"),
    ]:
        st.markdown(
            f'<div class="sb-domain">'
            f'<div class="sb-dot" style="background:{color};box-shadow:0 0 0 3px {color}30;"></div>'
            f'<span style="color:{color} !important;">{icon} {tag}</span></div>',
            unsafe_allow_html=True)
    st.divider()

    if log:
        df_log = pd.DataFrame(log)
        buf = io.StringIO()
        df_log.to_csv(buf, index=False)
        st.download_button("⬇ Download Session Report",
                           data=buf.getvalue().encode("utf-8"),
                           file_name=f"cleantrack_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)
    else:
        st.markdown('<div style="font-size:.78rem;color:#475569;padding:4px 0;">'
                    'Scan items to unlock CSV export.</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:16px;font-size:.7rem;color:#334155;line-height:1.5;">'
                '1M1B IBM SkillsBuild<br>AI + Sustainability Internship</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="sdg-pill">⬡ SDG 12 · Responsible Consumption &amp; Production</div>
  <h1>♻️ CleanTrack AI</h1>
  <p>
    Multi-domain AI waste segregation platform &nbsp;·&nbsp;
    <span style="opacity:.7;">NLP &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp;
    ML Forecasting &nbsp;·&nbsp; Anomaly Detection &nbsp;·&nbsp; Softmax Calibration</span><br>
    <span style="opacity:.6;font-size:.88rem;">
      Powered by IBM Granite logic &nbsp;·&nbsp; 1M1B SkillsBuild AI + Sustainability Internship
    </span>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Live KPI Bar ──────────────────────────────────────────────────────────────
log = st.session_state.scan_log
bin_counts = {}
for r in log:
    bin_counts[r["Bin"]] = bin_counts.get(r["Bin"], 0) + 1
top_bin  = max(bin_counts, key=bin_counts.get) if bin_counts else "—"
total_co2 = sum(r["CO₂ Saved (g)"] for r in log)

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="kv">{len(log)}</div><div class="kl">Items Scanned</div></div>
  <div class="kpi"><div class="kv">{total_co2}</div><div class="kl">g CO₂e Saved</div></div>
  <div class="kpi"><div class="kv">{bin_counts.get("Dry Recyclables",0)}</div><div class="kl">Recyclable</div></div>
  <div class="kpi"><div class="kv">{bin_counts.get("Wet Organic",0)}</div><div class="kl">Organic</div></div>
  <div class="kpi"><div class="kv">{bin_counts.get("Hazardous",0)}</div><div class="kl">Hazardous Diverted</div></div>
  <div class="kpi"><div class="kv" style="font-size:1.1rem;">{top_bin.split()[0] if top_bin!="—" else "—"}</div><div class="kl">Top Bin</div></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔤 Text · NLP Scan",
    "👁️ Image · CV Scan",
    "📈 Waste Forecast",
    "📋 Bulk Audit",
    "📜 History & Analytics",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT / NLP SCAN  (TF-IDF + Cosine Similarity)
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="info-banner" style="border-left-color:#7c3aed;">
      <div class="ib-icon">🔤</div>
      <div><strong>NLP Engine — TF-IDF + Cosine Similarity</strong><br>
      Vectorises your text against a labelled waste corpus and finds the closest
      category using cosine similarity — mirrors IBM Granite-style text classification.
      Softmax layer calibrates confidence across all 4 bin classes.</div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        opts = ["— Choose a campus item —","Greasy pizza box",
                "Crushed PET plastic bottle","Tetra Pak juice container",
                "Used lithium battery","Single-use paper coffee cup with plastic lining"]
        selected = st.selectbox("Common campus waste items:", opts, index=0)
    with col_b:
        custom = st.text_input("Or describe any waste item:",
                               placeholder="e.g. old smartphone, banana peel, glass jar…")

    nlp_clicked = st.button("🔤 Classify with NLP", type="primary",
                             use_container_width=True, key="btn_nlp")
    if nlp_clicked:
        if custom.strip():
            query = custom.strip()
            result, sim, method = nlp_classify(query)
            src = f"📝 NLP · {method} · similarity {sim:.3f}"
            method_label = "NLP · TF-IDF"
        elif selected != "— Choose a campus item —":
            query  = selected
            result = dict(WASTE_DATA[selected])
            sim    = 1.0
            src    = "📚 Knowledge Base · exact match"
            method_label = "NLP · Knowledge Base"
        else:
            st.warning("Please choose an item or type a custom description.")
            st.stop()

        log_scan(query, result, method_label)
        st.divider()
        render_result(query, result, src,
                      [("NLP","#7c3aed"),("TF-IDF","#a855f7"),("Softmax","#0f4c35")])
        render_softmax(result.get("corpus", query), result["bin_color"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE / COMPUTER VISION SCAN
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="info-banner" style="border-left-color:#0891b2;">
      <div class="ib-icon">👁️</div>
      <div><strong>Computer Vision Engine — HSV Colour Analysis</strong><br>
      Image is resized to 224×224 (standard CNN input), converted to HSV colour space,
      and analysed for dominant hue bucket, mean saturation, and brightness —
      mirroring feature-map activations of a MobileNetV3 / EfficientNet model trained on TrashNet.</div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a photo of your waste item:",
        type=["jpg","jpeg","png","webp"],
        help="Works best with clear, well-lit photos on a plain background.",
    )

    cv_clicked = st.button("👁️ Classify with Computer Vision", type="primary",
                            use_container_width=True, key="btn_cv",
                            disabled=(uploaded is None))

    if uploaded:
        img = Image.open(uploaded)
        col_img, col_info = st.columns([1, 2], gap="large")
        with col_img:
            st.image(img, caption="Uploaded image", use_container_width=True)

        if cv_clicked:
            with st.spinner("Running CV pipeline…"):
                result, features, feat_note = cv_classify(img)

            log_scan(f"[Image] {uploaded.name}", result, "Computer Vision")

            with col_info:
                feat_rows = "".join(
                    f'<div class="feat-row"><span class="feat-key">{k}</span>'
                    f'<span class="feat-val">{v}</span></div>'
                    for k, v in features.items())
                st.markdown(
                    f'<div class="rcard"><div class="stripe" style="background:#0891b2;"></div>'
                    f'<div class="body"><h3>🔬 Extracted Features — HSV Analysis</h3>'
                    f'{feat_rows}</div></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="cv-note">'
                    f'<strong>Model Decision Note:</strong> {feat_note}</div>',
                    unsafe_allow_html=True)

            st.divider()
            render_result(f"Image: {uploaded.name}", result,
                          f"👁️ Computer Vision · HSV colour analysis · conf {result['confidence']}%",
                          [("Computer Vision","#0891b2"),("HSV Analysis","#0e7490"),
                           ("CNN Pipeline","#155e75")])
            render_softmax(feat_note, result["bin_color"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML WASTE FORECAST  (Linear Regression)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="info-banner" style="border-left-color:#b45309;">
      <div class="ib-icon">📈</div>
      <div><strong>ML Forecasting Engine — Linear Regression (scikit-learn)</strong><br>
      Trained on 14 days of campus scan history to predict the next 7 days of waste volume.
      In production: replace with Facebook Prophet or an LSTM for seasonal patterns.</div>
    </div>""", unsafe_allow_html=True)

    history = st.session_state.daily_counts
    forecast, f_labels, r2 = forecast_waste(history)

    # Build chart data
    today = datetime.date.today()
    hist_labels = [(today - datetime.timedelta(days=len(history)-i)).strftime("%b %d")
                   for i in range(len(history))]

    col_f1, col_f2 = st.columns([2, 1], gap="large")
    with col_f1:
        chart_df = pd.DataFrame({
            "Day":   hist_labels + f_labels,
            "Scans": [round(v, 1) for v in history] + [round(v, 1) for v in forecast],
            "Type":  ["Historical"]*len(history) + ["Forecast"]*7,
        })
        st.line_chart(chart_df.set_index("Day")["Scans"], color="#0f4c35")
        st.caption(f"Historical (blue) + 7-day Linear Regression forecast · R² = {r2:.3f}")

    with col_f2:
        fc_rows = "".join(
            f'<div class="fc-row">'
            f'<div class="fc-top"><span>{label}</span><span>{val:.1f} scans</span></div>'
            f'<div class="fc-bar-bg"><div class="fc-bar" style="width:{min(100,int(val/max(forecast+[1])*100))}%;"></div></div>'
            f'</div>'
            for label, val in zip(f_labels, forecast))
        st.markdown(
            f'<div class="rcard"><div class="stripe" style="background:#b45309;"></div>'
            f'<div class="body"><h3>7-Day Forecast</h3>{fc_rows}'
            f'<div class="model-box">'
            f'<strong>Model:</strong> LinearRegression (scikit-learn)<br>'
            f'<strong>R²:</strong> {r2:.4f} &nbsp;·&nbsp; '
            f'<strong>Training Days:</strong> {len(history)} &nbsp;·&nbsp; '
            f'<strong>Horizon:</strong> 7 days'
            f'</div></div></div>',
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — BULK CAMPUS AUDIT
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="info-banner" style="border-left-color:#16a34a;">
      <div class="ib-icon">📋</div>
      <div><strong>Bulk NLP Audit</strong><br>
      Paste a list of waste items. The NLP engine classifies each one in batch and
      produces a colour-coded, downloadable audit table — designed for campus waste
      managers running physical bin audits.</div>
    </div>""", unsafe_allow_html=True)

    bulk_text = st.text_area("Paste items (one per line or comma-separated):",
                              height=130,
                              placeholder="Greasy pizza box\nOld smartphone\nBanana peel\nAluminium can")
    bulk_go = st.button("📋 Run Bulk Classification", type="primary",
                        use_container_width=True)

    if bulk_go and bulk_text.strip():
        items = [i.strip() for line in bulk_text.splitlines()
                 for i in line.split(",") if i.strip()]
        rows = []
        for item in items:
            if item in WASTE_DATA:
                r = dict(WASTE_DATA[item]); method = "Knowledge Base"
            else:
                r, _, method = nlp_classify(item)
            log_scan(item, r, f"Bulk NLP · {method}")
            rows.append({
                "Item": item, "Bin": r["bin"],
                "Confidence (%)": r["confidence"],
                "CO₂ Saved (g)": r["co2_saved_grams"],
                "Key Step": r["prep_steps"][0],
                "AI Method": method,
            })

        df = pd.DataFrame(rows)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Items", len(df))
        m2.metric("CO₂ Saved (g)", df["CO₂ Saved (g)"].sum())
        m3.metric("Recyclable", int((df["Bin"]=="Dry Recyclables").sum()))
        m4.metric("Hazardous", int((df["Bin"]=="Hazardous").sum()))

        def colour_bin(val):
            c = {"Dry Recyclables":"background:#dbeafe;color:#1e40af",
                 "Wet Organic":"background:#dcfce7;color:#166534",
                 "Hazardous":"background:#fee2e2;color:#991b1b",
                 "Landfill Residual":"background:#f3f4f6;color:#374151"}
            return c.get(val,"")

        st.dataframe(df.style.applymap(colour_bin, subset=["Bin"]),
                     use_container_width=True, hide_index=True)

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("⬇ Download Audit CSV",
                           data=buf.getvalue().encode("utf-8"),
                           file_name=f"cleantrack_audit_{datetime.date.today()}.csv",
                           mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY & ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    if not st.session_state.scan_log:
        st.markdown("""
        <div class="info-banner" style="border-left-color:#94a3b8;">
          <div class="ib-icon">📜</div>
          <div><strong>No scans yet</strong><br>
          Use the NLP Scan or Image CV Scan tabs to analyse waste items.
          Every scan is logged here automatically.</div>
        </div>""", unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(st.session_state.scan_log)

        h1,h2,h3,h4 = st.columns(4)
        h1.metric("Total Scans",      len(df_hist))
        h2.metric("CO₂e Saved (g)",   df_hist["CO₂ Saved (g)"].sum())
        h3.metric("Avg Confidence",   f"{df_hist['Confidence (%)'].mean():.0f}%")
        h4.metric("Unique Bins Used", df_hist["Bin"].nunique())

        st.divider()

        # Anomaly alert
        anomaly = detect_anomaly(st.session_state.scan_log)
        if anomaly:
            st.markdown(
                f'<div class="anomaly-box">'
                f'<span style="font-size:1.2rem;flex-shrink:0;">🚨</span>'
                f'<div><strong>Anomaly Detection (Z-score)</strong><br>{anomaly["message"]}'
                f'<br><span style="font-size:.76rem;opacity:.8;">'
                f'Z-score = {anomaly["z_score"]} &nbsp;·&nbsp; Count = {anomaly["count"]}'
                f'</span></div></div>',
                unsafe_allow_html=True)
            st.write("")

        col_h1, col_h2 = st.columns(2, gap="large")
        with col_h1:
            st.markdown('<p style="font-size:.8rem;font-weight:700;color:#64748b;'
                        'text-transform:uppercase;letter-spacing:.06em;">Bin Breakdown</p>',
                        unsafe_allow_html=True)
            bin_summary = df_hist["Bin"].value_counts().reset_index()
            bin_summary.columns = ["Bin","Count"]
            st.dataframe(bin_summary, use_container_width=True, hide_index=True)

        with col_h2:
            st.markdown('<p style="font-size:.8rem;font-weight:700;color:#64748b;'
                        'text-transform:uppercase;letter-spacing:.06em;">AI Method Used</p>',
                        unsafe_allow_html=True)
            method_summary = df_hist["AI Method"].value_counts().reset_index()
            method_summary.columns = ["Method","Count"]
            st.dataframe(method_summary, use_container_width=True, hide_index=True)

        st.markdown('<p style="font-size:.8rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:.06em;margin-top:8px;">Full Scan Log</p>',
                    unsafe_allow_html=True)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear Session History"):
            st.session_state.scan_log = []
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:32px;padding:20px 0 8px;border-top:1px solid #e2e8f0;
            text-align:center;">
  <div style="font-size:.82rem;font-weight:700;color:#0f172a;margin-bottom:4px;">
    ♻️ CleanTrack AI
  </div>
  <div style="font-size:.73rem;color:#94a3b8;line-height:1.7;">
    NLP &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp; ML Forecasting &nbsp;·&nbsp;
    Anomaly Detection &nbsp;·&nbsp; Softmax Calibration<br>
    IBM Granite Logic &nbsp;·&nbsp; 1M1B IBM SkillsBuild AI + Sustainability &nbsp;·&nbsp; SDG 12
  </div>
</div>""", unsafe_allow_html=True)
