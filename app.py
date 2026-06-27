"""
Streamlit front-end — inference only.
Models are trained offline with train.py and loaded at startup.
"""

import os, sys, json, base64
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_PATH = Path("reports/pipeline_results.json")
RISK_CSV     = Path("reports/risk_report.csv")
FIGURES      = Path("reports/figures")
MODEL_PATHS  = {
    "ensemble":     Path("models/stacking_model.joblib"),
    "lstm":         Path("models/lstm_weights.npz"),
    "preprocessor": Path("models/preprocessor.joblib"),
    "features":     Path("models/feature_names.joblib"),
}


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    missing = [k for k, p in MODEL_PATHS.items() if not p.exists()]
    if missing:
        return None, missing

    from models.ensemble_model import StackingDropoutModel
    from models.lstm_attendance import load_weights

    ensemble      = StackingDropoutModel.load(str(MODEL_PATHS["ensemble"]))
    lstm_weights  = load_weights(str(MODEL_PATHS["lstm"]))
    preprocessor  = joblib.load(MODEL_PATHS["preprocessor"])
    feature_names = joblib.load(MODEL_PATHS["features"])
    return {"ensemble": ensemble, "lstm": lstm_weights,
            "preprocessor": preprocessor, "feature_names": feature_names}, []


def load_training_results():
    if not RESULTS_PATH.exists():
        return None
    try:
        p = json.loads(RESULTS_PATH.read_text())
        risk_df = pd.read_csv(RISK_CSV) if RISK_CSV.exists() else None
        return {**p, "risk_df": risk_df}
    except Exception:
        return None


# ── Inference ─────────────────────────────────────────────────────────────────
def score_dataset(df: pd.DataFrame, models: dict, tier_thresholds: dict | None = None) -> dict:
    from data.synthetic_generator import generate_attendance_sequences
    from preprocessing.pipeline import transform_for_inference
    from models.lstm_attendance import generate_features
    from models.risk_stratification import stratify_students

    if "student_id" not in df.columns:
        df = df.copy()
        df.insert(0, "student_id", [f"STU{i:05d}" for i in range(len(df))])

    _STEPS = [
        (0,  "📡", "Reading attendance patterns…"),
        (25, "🧠", "Transforming features…"),
        (50, "⚡", "LSTM generating attendance signals…"),
        (70, "🔮", "Ensemble scoring students…"),
        (85, "📊", "Stratifying risk tiers…"),
    ]

    anim = st.empty()

    def _show(pct, icon, label):
        filled = int(pct / 5)
        bar_html = (
            "█" * filled + "░" * (20 - filled)
        )
        anim.markdown(f"""
<div style="
  background:linear-gradient(135deg,#0f1117 0%,#1a1f2e 100%);
  border:1px solid rgba(124,111,247,.35);
  border-radius:16px;padding:40px 32px;text-align:center;
  box-shadow:0 0 40px rgba(124,111,247,.15);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="font-size:3.2em;margin-bottom:16px;animation:none">{icon}</div>
  <div style="font-size:1.15em;font-weight:700;color:#e6edf3;margin-bottom:20px">{label}</div>
  <div style="
    font-family:monospace;font-size:1.05em;letter-spacing:3px;
    color:#7c6ff7;margin-bottom:14px">{bar_html}</div>
  <div style="font-size:.82em;color:rgba(255,255,255,.35)">{pct}% complete · {len(df):,} students</div>
  <style>
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}
  </style>
</div>""", unsafe_allow_html=True)

    _show(*_STEPS[0])
    attendance = generate_attendance_sequences(df)

    _show(*_STEPS[1])
    X = transform_for_inference(df, models["preprocessor"])

    _show(*_STEPS[2])
    trend, lstm_prob = generate_features(models["lstm"], attendance)
    X_aug = np.column_stack([X, trend, lstm_prob])

    _show(*_STEPS[3])
    dropout_probs = models["ensemble"].predict_proba(X_aug)[:, 1]

    _show(*_STEPS[4])
    risk_df = stratify_students(
        student_ids=df["student_id"].values,
        dropout_probs=dropout_probs,
        output_dir="reports",
        source_df=df,
        attendance_trend_scores=trend,
        thresholds=tier_thresholds,
    )

    anim.markdown(f"""
<div style="
  background:linear-gradient(135deg,#0f1117 0%,#1a1f2e 100%);
  border:1px solid rgba(63,185,80,.4);
  border-radius:16px;padding:40px 32px;text-align:center;
  box-shadow:0 0 40px rgba(63,185,80,.12);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="font-size:3.2em;margin-bottom:16px">✅</div>
  <div style="font-size:1.15em;font-weight:700;color:#3fb950;margin-bottom:8px">Analysis complete</div>
  <div style="font-size:.88em;color:rgba(255,255,255,.4)">{len(df):,} students scored · navigate to Risk Dashboard</div>
</div>""", unsafe_allow_html=True)
    import time; time.sleep(1.2)
    anim.empty()

    return {"risk_df": risk_df, "dropout_probs": dropout_probs}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"
if "score_results" not in st.session_state:
    st.session_state["score_results"] = None

DARK = st.session_state["theme"] == "dark"

# ── Colour tokens ─────────────────────────────────────────────────────────────
if DARK:
    BG, SB, CARD, BORDER = "#0d1117", "#161b22", "#21262d", "#30363d"
    TEXT, MUTED           = "#e6edf3", "#8b949e"
    A1, A2                = "#7c6ff7", "#3fb950"
    HIGH_C, MED_C, LOW_C  = "#f85149", "#e3b341", "#3fb950"
    HIGH_BG, MED_BG, LOW_BG = "rgba(248,81,73,.12)", "rgba(227,179,65,.12)", "rgba(63,185,80,.12)"
    PTMPL, SHADOW         = "plotly_dark", "0 0 0 1px #30363d,0 4px 24px rgba(0,0,0,.5)"
    HERO_A, HERO_B        = "#1a1f35", "#0d1b3e"
else:
    BG, SB, CARD, BORDER  = "#f6f8fa", "#1a3a5c", "#ffffff", "#d0d7de"
    TEXT, MUTED            = "#1f2328", "#656d76"
    A1, A2                 = "#5b4fcf", "#2da44e"
    HIGH_C, MED_C, LOW_C   = "#cf222e", "#9a6700", "#1a7f37"
    HIGH_BG, MED_BG, LOW_BG = "rgba(207,34,46,.08)", "rgba(154,103,0,.08)", "rgba(26,127,55,.08)"
    PTMPL, SHADOW          = "plotly_white", "0 1px 3px rgba(31,35,40,.12),0 8px 24px rgba(66,74,83,.12)"
    HERO_A, HERO_B         = "#1a3a5c", "#0f2a4a"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main>div{{
  background:{BG}!important;color:{TEXT}!important;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif!important;
}}
[data-testid="stSidebar"]>div:first-child{{background:{SB}!important;border-right:1px solid rgba(255,255,255,.08)!important;}}
[data-testid="stSidebar"] *{{color:#e6edf3!important;}}
[data-testid="stSidebar"] .stRadio>label,[data-testid="stSidebar"] .stSlider>label,
[data-testid="stSidebar"] .stFileUploader>label{{
  color:rgba(255,255,255,.5)!important;font-size:.72em!important;
  font-weight:600!important;text-transform:uppercase!important;letter-spacing:.08em!important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p{{color:#e6edf3!important;font-size:.9em!important;}}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapseButton"] button{{visibility:visible!important;opacity:1!important;}}
[data-testid="stSidebarCollapseButton"] button{{color:rgba(255,255,255,.7)!important;border-radius:6px!important;}}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] button{{
  position:fixed!important;left:0!important;top:14px!important;width:28px!important;
  height:36px!important;min-height:36px!important;background:{A1}!important;
  border-radius:0 8px 8px 0!important;box-shadow:2px 0 12px rgba(0,0,0,.5)!important;
  z-index:999999!important;padding:0 4px!important;color:white!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
}}
.card{{background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:24px;box-shadow:{SHADOW};}}
.kpi{{background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:20px 16px 16px;
  text-align:center;box-shadow:{SHADOW};position:relative;overflow:hidden;}}
.kpi::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,{A1},{A2});border-radius:12px 12px 0 0;}}
.kpi-icon{{font-size:1.4em;margin-bottom:6px;}}
.kpi-val{{font-size:2.1em;font-weight:800;
  background:linear-gradient(135deg,{A1} 0%,{A2} 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1;margin:4px 0 6px;}}
.kpi-lbl{{font-size:.7em;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:{MUTED};}}
.kpi-sub{{font-size:.75em;color:{MUTED};margin-top:4px;}}
.hero{{background:linear-gradient(135deg,{HERO_A} 0%,{HERO_B} 100%);
  border:1px solid rgba(124,111,247,.2);border-radius:14px;padding:32px 36px;
  margin-bottom:24px;position:relative;overflow:hidden;}}
.hero::after{{content:"";position:absolute;top:-60px;right:-60px;width:200px;height:200px;
  background:radial-gradient(circle,rgba(124,111,247,.15) 0%,transparent 70%);border-radius:50%;}}
.hero h1{{color:#fff!important;font-size:1.8em!important;font-weight:700!important;
  margin:0 0 6px!important;line-height:1.2!important;}}
.hero p{{color:rgba(255,255,255,.6)!important;font-size:.9em!important;margin:0!important;}}
.hero-badge{{display:inline-block;background:rgba(124,111,247,.2);color:#a69df5!important;
  border:1px solid rgba(124,111,247,.3);border-radius:20px;font-size:.72em;font-weight:600;
  padding:3px 10px;margin:0 4px 0 0;text-transform:uppercase;letter-spacing:.06em;}}
.risk-big{{border-radius:12px;padding:20px;text-align:center;border:1px solid;}}
.risk-count{{font-size:3em;font-weight:800;line-height:1;}}
.risk-label{{font-size:.72em;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-top:6px;opacity:.85;}}
.sec-head{{font-size:.72em;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{MUTED};margin:20px 0 12px;}}
.plot-card{{background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:16px;box-shadow:{SHADOW};}}
.plot-title{{font-size:.85em;font-weight:600;color:{TEXT};margin-bottom:10px;}}
.info-box{{background:rgba(124,111,247,.06);border:1px solid rgba(124,111,247,.2);
  border-radius:10px;padding:20px 24px;margin-bottom:16px;}}
.info-box p{{color:{TEXT}!important;font-size:.88em!important;line-height:1.7!important;margin:0!important;}}
.step-badge{{display:inline-flex;align-items:center;justify-content:center;
  width:22px;height:22px;border-radius:50%;background:{A1};color:white;
  font-size:.72em;font-weight:700;margin-right:8px;flex-shrink:0;}}
.stButton>button{{background:linear-gradient(135deg,{A1},{A2})!important;color:#fff!important;
  border:none!important;border-radius:8px!important;font-weight:600!important;
  font-size:.88em!important;padding:9px 18px!important;width:100%!important;
  transition:opacity .15s,transform .1s!important;}}
.stButton>button:hover{{opacity:.88!important;transform:translateY(-1px)!important;}}
[data-testid="stProgressBar"]>div{{background:linear-gradient(90deg,{A1},{A2})!important;border-radius:4px!important;}}
[data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {BORDER}!important;border-radius:10px!important;}}
[data-testid="stFileUploader"]{{background:rgba(124,111,247,.06)!important;
  border:1px dashed rgba(124,111,247,.4)!important;border-radius:8px!important;padding:4px!important;}}
[data-testid="stAlert"]{{border-radius:8px!important;border:none!important;}}
#MainMenu,footer{{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none!important;}}
[data-testid="stDecoration"]{{display:none!important;}}
[data-testid="stHeader"]{{display:none!important;}}
section.main>div.block-container{{padding-top:1rem!important;}}
[data-testid="stSidebar"]>div:first-child{{padding-top:0!important;}}
iframe[height="0"]{{display:none!important;}}
@media(max-width:768px){{
  .kpi-val{{font-size:1.6em!important;}} .kpi{{padding:14px 10px 12px!important;}}
  .hero{{padding:20px 18px!important;}} .hero h1{{font-size:1.3em!important;}}
  .risk-count{{font-size:2.2em!important;}} .plot-card{{padding:10px!important;}}
  [data-testid="stSidebar"]>div:first-child{{position:fixed!important;z-index:999990!important;height:100vh!important;overflow-y:auto!important;}}
}}
@media(max-width:480px){{
  .hero{{padding:16px!important;}} .hero h1{{font-size:1.1em!important;}}
  .kpi-val{{font-size:1.4em!important;}} .sec-head{{font-size:.65em!important;}}
  [data-testid="column"]{{min-width:100%!important;}}
}}
</style>
""", unsafe_allow_html=True)

# Floating hamburger toggle
components.html("""
<script>
(function(){
  var pd=window.parent,pdoc=pd.document;
  function sidebarOpen(){
    var sb=pdoc.querySelector('[data-testid="stSidebar"]');
    return sb&&sb.getAttribute('aria-expanded')==='true';
  }
  function sync(){
    var btn=pdoc.getElementById('sb-float-btn');
    if(!btn)return;
    btn.style.display=sidebarOpen()?'none':'flex';
  }
  function inject(){
    if(pdoc.getElementById('sb-float-btn'))return;
    var s=pdoc.createElement('style');
    s.textContent='#sb-float-btn{position:fixed!important;top:14px!important;left:14px!important;'+
      'z-index:9999999!important;width:36px!important;height:36px!important;background:#7c6ff7!important;'+
      'border:none!important;border-radius:8px!important;cursor:pointer!important;display:flex!important;'+
      'align-items:center!important;justify-content:center!important;flex-direction:column!important;'+
      'gap:4px!important;padding:8px!important;box-shadow:0 2px 12px rgba(0,0,0,.4)!important;'+
      'transition:background .15s,transform .1s!important;}'+
      '#sb-float-btn:hover{background:#4d9cf7!important;transform:scale(1.06)!important;}'+
      '#sb-float-btn span{display:block!important;width:16px!important;height:2px!important;'+
      'background:white!important;border-radius:2px!important;flex-shrink:0!important;}'+
      'body:has([data-testid="stSidebar"][aria-expanded="true"]) #sb-float-btn{display:none!important;}';
    pdoc.head.appendChild(s);
    var btn=pdoc.createElement('button');
    btn.id='sb-float-btn';btn.title='Toggle sidebar';
    btn.innerHTML='<span></span><span></span><span></span>';
    btn.addEventListener('click',function(){
      var inner=pdoc.querySelector('[data-testid="stSidebarCollapseButton"] button');
      if(inner)inner.click();
    });
    pdoc.body.appendChild(btn);
    // MutationObserver for instant response
    var sb=pdoc.querySelector('[data-testid="stSidebar"]');
    if(sb){
      new MutationObserver(sync).observe(sb,{attributes:true,attributeFilter:['aria-expanded']});
      // Auto-close sidebar when a nav radio item is clicked
      sb.addEventListener('click',function(e){
        var radio=e.target.closest('label[data-baseweb="radio"]');
        if(radio){
          setTimeout(function(){
            var inner=pdoc.querySelector('[data-testid="stSidebarCollapseButton"] button');
            if(inner)inner.click();
          },120);
        }
      });
    }
    // Polling fallback — catches any state the observer misses
    setInterval(sync,300);
    sync();
  }
  pdoc.readyState==='loading'?pdoc.addEventListener('DOMContentLoaded',inject):inject();
})();
</script>
""", height=0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(icon, label, value, sub=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-lbl">{label}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )

def hero(title, subtitle, badges=None):
    badge_html = "".join(f'<span class="hero-badge">{b}</span>' for b in (badges or []))
    st.markdown(
        f'<div class="hero"><p style="margin-bottom:10px">{badge_html}</p>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def section_head(text):
    st.markdown(f'<div class="sec-head">{text}</div>', unsafe_allow_html=True)

def show_img(path, title=""):
    if not Path(path).exists():
        st.markdown(
            f'<div class="plot-card" style="text-align:center;padding:32px;color:{MUTED}">'
            f'Figure not generated yet</div>',
            unsafe_allow_html=True,
        )
        return
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    t = f'<div class="plot-title">{title}</div>' if title else ""
    st.markdown(
        f'<div class="plot-card">{t}<img src="data:image/png;base64,{data}" '
        f'style="width:100%;border-radius:6px;display:block"></div>',
        unsafe_allow_html=True,
    )

def pcfg(fig, h=340):
    fig.update_layout(
        template=PTMPL, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif", color=TEXT, size=12),
        height=h, margin=dict(t=24, b=24, l=8, r=8),
    )
    return fig


# ── Load models & training results ────────────────────────────────────────────
models, missing = load_models()
training_results = load_training_results()
models_ready = models is not None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 12px;border-bottom:1px solid rgba(255,255,255,.08)">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="font-size:1.6em;line-height:1">🎓</div>
        <div>
          <div style="font-size:.95em;font-weight:700;color:#e6edf3">Dropout Predictor</div>
          <div style="font-size:.68em;color:rgba(255,255,255,.35);margin-top:1px">ML Analytics · v2</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Theme toggle
    if st.button("☀️ Light" if DARK else "🌙 Dark", use_container_width=True):
        st.session_state["theme"] = "light" if DARK else "dark"
        st.rerun()

    # ── Upload dataset ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin:18px 0 8px;font-size:.68em;font-weight:700;text-transform:uppercase;
                letter-spacing:.1em;color:rgba(255,255,255,.3)">
      <span style="background:#7c6ff7;color:white;border-radius:50%;padding:1px 6px;margin-right:6px;font-size:.85em">1</span>
      Upload Dataset
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "CSV with a 'target' column",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.markdown(
            f'<div style="font-size:.78em;color:#3fb950;margin-top:4px;padding:6px 8px;'
            f'background:rgba(63,185,80,.1);border-radius:6px;border:1px solid rgba(63,185,80,.2)">'
            f'✓ {uploaded_file.name}</div>',
            unsafe_allow_html=True,
        )

    # ── Score / navigation (only when models ready) ────────────────────────
    if models_ready:
        if uploaded_file is not None:
            st.markdown("""
            <div style="margin:18px 0 8px;font-size:.68em;font-weight:700;text-transform:uppercase;
                        letter-spacing:.1em;color:rgba(255,255,255,.3)">
              <span style="background:#7c6ff7;color:white;border-radius:50%;padding:1px 6px;margin-right:6px;font-size:.85em">2</span>
              Score
            </div>
            """, unsafe_allow_html=True)
            score_btn = st.button("▶ Score Dataset", use_container_width=True)
        else:
            score_btn = False
            st.markdown(
                f'<div style="font-size:.75em;color:rgba(255,255,255,.3);'
                f'text-align:center;margin-top:12px">Upload a CSV to score it</div>',
                unsafe_allow_html=True,
            )

        # Navigation
        st.markdown("""
        <div style="margin:18px 0 8px;font-size:.68em;font-weight:700;text-transform:uppercase;
                    letter-spacing:.1em;color:rgba(255,255,255,.3)">Pages</div>
        """, unsafe_allow_html=True)

        PAGES = ["Overview", "Model Evaluation", "SHAP Explainability", "LSTM Attendance", "Risk Dashboard"]
        PAGE_ICONS = {"Overview":"🏠","Model Evaluation":"📊",
                      "SHAP Explainability":"🔍","LSTM Attendance":"📈","Risk Dashboard":"⚠️"}
        page_idx = PAGES.index(st.session_state["page"]) if st.session_state["page"] in PAGES else 0
        page = st.radio("nav",
                        [f"{PAGE_ICONS[p]}  {p}" for p in PAGES],
                        index=page_idx, label_visibility="collapsed")
        page = page.split("  ", 1)[1]
        st.session_state["page"] = page

    else:
        score_btn = False
        page = "Overview"
        st.markdown(
            f'<div style="margin-top:20px;padding:12px;background:rgba(248,81,73,.1);'
            f'border:1px solid rgba(248,81,73,.3);border-radius:8px;'
            f'font-size:.78em;color:#f85149;line-height:1.6">'
            f'⚠ No trained models found.<br><br>'
            f'After generating a dataset, run:<br>'
            f'<code style="font-size:.9em">python train.py</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if training_results and training_results.get("run_timestamp"):
        st.markdown(
            f'<div style="font-size:.65em;color:rgba(255,255,255,.25);text-align:center;'
            f'margin-top:12px">🕐 Trained {training_results["run_timestamp"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style="margin-top:auto;padding:16px 0 4px;font-size:.66em;color:rgba(255,255,255,.2);line-height:1.7">
      RF · XGBoost · LightGBM<br>→ Logistic Regression<br>LSTM · SHAP
    </div>
    """, unsafe_allow_html=True)


# ── Score trigger ─────────────────────────────────────────────────────────────
if score_btn and uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        if "target" not in user_df.columns:
            st.error("CSV must have a **'target'** column (0=enrolled, 1=dropout).")
        elif user_df["target"].nunique() < 2:
            st.error("The **'target'** column must contain both 0s and 1s.")
        elif len(user_df) < 50:
            st.error("Need at least **50 rows** to score.")
        else:
            tier_thresholds = (training_results or {}).get("tier_thresholds")
            st.session_state["score_results"] = score_dataset(user_df, models, tier_thresholds)
            st.rerun()
    except Exception as e:
        st.error(f"Could not process CSV: {e}")


# ── Page rendering ────────────────────────────────────────────────────────────

# No models → always show setup screen
if not models_ready:
    hero("🎓 Student Dropout Prediction",
         "Ensemble ML · SHAP Explainability · LSTM Feature Fusion · Risk Stratification",
         badges=["RF + XGBoost + LightGBM", "SHAP", "LSTM"])
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:48px 32px">
      <div style="font-size:3em;margin-bottom:16px">🔧</div>
      <div style="font-size:1.2em;font-weight:700;color:{TEXT};margin-bottom:12px">Setup required</div>
      <div style="color:{MUTED};font-size:.9em;max-width:420px;margin:0 auto;line-height:1.8">
        <div style="margin-bottom:8px"><span class="step-badge">1</span>
          In your terminal: <code>python train.py</code></div>
        <div><span class="step-badge">2</span>
          Upload your CSV and click <strong>▶ Score Dataset</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

tr = training_results or {}
m  = tr.get("ensemble_metrics", {})
lm = tr.get("lstm_metrics", {})
score_res = st.session_state.get("score_results")

# risk_df only exists after the user uploads and scores a CSV this session
risk_df = score_res["risk_df"] if score_res else None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    hero("🎓 Student Dropout Prediction",
         "Ensemble ML · SHAP Explainability · LSTM Feature Fusion · Risk Stratification",
         badges=["RF + XGBoost + LightGBM", "SHAP", "LSTM"])

    if not m:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:56px 32px">
          <div style="font-size:3.5em;margin-bottom:16px">📂</div>
          <div style="font-size:1.25em;font-weight:700;color:{TEXT};margin-bottom:8px">Models loaded ✓</div>
          <div style="color:{MUTED};font-size:.92em">
            Upload a CSV and click <strong style="color:{A1}">▶ Score Dataset</strong>
            to generate a risk report.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    section_head("Training Performance — Ensemble Classifier")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1: kpi("🎯", "Accuracy",      m.get("accuracy","—"),      "Test set")
    with c2: kpi("📈", "ROC-AUC",       m.get("roc_auc","—"),       "Area under curve")
    with c3: kpi("⚡", "Avg Precision", m.get("avg_precision","—"), "PR curve")
    with c4: kpi("✅", "F1 (Dropout)",  m.get("f1_dropout","—"),    "Dropout class")

    if tr.get("n_students"):
        st.markdown(
            f'<div style="font-size:.75em;color:{MUTED};margin-top:4px">'
            f'Trained on <strong>{tr["n_students"]:,}</strong> students · '
            f'{tr.get("training_time_s","?")}s training time</div>',
            unsafe_allow_html=True,
        )

    if risk_df is None:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:40px 32px;margin-top:20px">
          <div style="font-size:2.5em;margin-bottom:12px">📂</div>
          <div style="font-size:1.05em;font-weight:700;color:{TEXT};margin-bottom:8px">Upload your dataset to see predictions</div>
          <div style="color:{MUTED};font-size:.88em">Open the sidebar → upload your CSV → click <strong style="color:{A1}">▶ Score Dataset</strong></div>
        </div>
        """, unsafe_allow_html=True)

    if risk_df is not None:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            section_head("Risk Distribution")
            counts = risk_df["risk_tier"].value_counts().reset_index()
            counts.columns = ["Tier", "Students"]
            fig = px.pie(counts, names="Tier", values="Students", hole=0.5,
                         color="Tier",
                         color_discrete_map={"High":HIGH_C,"Medium":MED_C,"Low":LOW_C})
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont=dict(size=13, color="white"),
                              marker=dict(line=dict(color=CARD, width=3)))
            pcfg(fig, 300)
            fig.update_layout(showlegend=False, margin=dict(t=8,b=8,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section_head("Top Predictive Features")
            feats = tr.get("top_features", [])
            if feats:
                colors = [f"rgba(124,111,247,{0.4+0.06*i})" for i in range(len(feats))]
                fig2 = go.Figure(go.Bar(
                    y=feats[::-1], x=list(range(len(feats), 0, -1)), orientation="h",
                    marker=dict(color=colors[::-1], line=dict(width=0)),
                    text=[f"#{i+1}" for i in range(len(feats)-1,-1,-1)],
                    textposition="inside",
                    textfont=dict(color="#fff", size=11, family="monospace"),
                ))
                pcfg(fig2, 300)
                fig2.update_layout(
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(tickfont=dict(size=11), gridcolor="rgba(0,0,0,0)"),
                    margin=dict(t=8,b=8,l=0,r=0),
                )
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    hero("📊 Model Evaluation",
         "Stacking classifier performance on the held-out test set",
         badges=["Test set", "No data leakage"])

    if not m:
        st.info("No training metrics yet — run train.py first."); st.stop()

    section_head("Dropout Class Performance")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1: kpi("🎯", "Precision", m.get("precision_dropout","—"), "Dropout class")
    with c2: kpi("🔎", "Recall",    m.get("recall_dropout","—"),    "Dropout class")
    with c3: kpi("✅", "F1 Score",  m.get("f1_dropout","—"),        "Dropout class")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    section_head("Full Metrics (Test Set)")
    keys = [
        ("Accuracy","accuracy"),("ROC-AUC","roc_auc"),("Average Precision","avg_precision"),
        ("Precision (Dropout)","precision_dropout"),("Recall (Dropout)","recall_dropout"),
        ("F1 (Dropout)","f1_dropout"),("Precision (Non-Dropout)","precision_nondropout"),
        ("Recall (Non-Dropout)","recall_nondropout"),("F1 (Non-Dropout)","f1_nondropout"),
    ]
    st.dataframe(
        pd.DataFrame({"Metric":[l for l,_ in keys], "Value":[m.get(k,"—") for _,k in keys]}),
        use_container_width=True, hide_index=True, height=300,
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_head("Diagnostic Plots")
    r1, r2 = st.columns(2, gap="medium")
    with r1:
        show_img(FIGURES/"confusion_matrix.png", "Confusion Matrix")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        show_img(FIGURES/"pr_curve.png", "Precision-Recall Curve")
    with r2:
        show_img(FIGURES/"roc_curve.png", "ROC Curve")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        show_img(FIGURES/"calibration_curve.png", "Calibration Curve")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    hero("🔍 SHAP Explainability",
         "TreeExplainer on XGBoost — attendance features now included in importance",
         badges=["TreeExplainer", "Global", "Local"])

    section_head("Global Feature Importance")
    g1, g2 = st.columns(2, gap="medium")
    with g1: show_img(FIGURES/"shap_summary_bar.png", "Feature Importance (Mean |SHAP|)")
    with g2: show_img(FIGURES/"shap_beeswarm.png", "Beeswarm — Value & Direction")

    wf = sorted(FIGURES.glob("shap_waterfall_*.png"))
    if wf:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        section_head("Local Explanations — Highest-Risk Students")
        for png in wf:
            sid = png.stem.replace("shap_waterfall_","")
            with st.expander(f"Student {sid}", expanded=False):
                show_img(png)

    st.markdown(f"""
    <div class="info-box" style="margin-top:16px">
      <p>The LSTM's two outputs — <strong>attendance_trend_score</strong> (rolling 30-day
      attendance mean) and <strong>lstm_next_day_prob</strong> (predicted next-day attendance
      probability) — are now <em>features inside the ensemble</em>. SHAP can directly attribute
      how much each attendance signal drives each individual dropout prediction.</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LSTM Attendance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "LSTM Attendance":
    hero("📈 LSTM Attendance",
         "Trained first — outputs feed directly into the ensemble as features",
         badges=["Feature fusion", "EarlyStopping"])

    if not lm:
        st.info("No LSTM metrics yet — run train.py first."); st.stop()

    icons = {"lstm_accuracy":"🎯","lstm_mae":"📉","lstm_rmse":"📐"}
    cols = st.columns(min(len(lm), 3), gap="small")
    for col, (k, v) in zip(cols, lm.items()):
        with col:
            kpi(icons.get(k,"📊"), k.replace("lstm_","").replace("_"," ").upper(), f"{v:.4f}")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_head("Architecture")
    st.markdown(f"""
    <div class="info-box">
      <p>
        <strong>Input:</strong> 30-day binary attendance window (1=present, 0=absent)<br>
        <strong>Architecture:</strong> LSTM(32) → Dropout(0.2) → Dense(1, sigmoid)<br>
        <strong>Training:</strong> up to 8 epochs · batch 4096 · EarlyStopping patience 3<br><br>
        <strong>Role in pipeline:</strong> the LSTM is trained <em>before</em> the ensemble.
        Its two outputs per student — rolling 30-day attendance mean
        (<code>attendance_trend_score</code>) and predicted next-day probability
        (<code>lstm_next_day_prob</code>) — are appended to the tabular feature matrix.
        The stacking ensemble then learns when and how much to weight attendance patterns
        alongside grades, fees, and demographics. SHAP explains both signals automatically.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Risk Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Dashboard":
    _t = (training_results or {}).get("tier_thresholds", {"High": 0.70, "Medium": 0.40})
    hero("⚠️ Risk Dashboard",
         f"High ≥ {_t['High']:.2f} · Medium ≥ {_t['Medium']:.2f} · Low < {_t['Medium']:.2f}",
         badges=["Real-time filter", "CSV export"])

    if risk_df is None:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:48px">
          <div style="font-size:2.5em;margin-bottom:12px">📋</div>
          <div style="color:{MUTED};font-size:.92em">
            Upload a CSV and click <strong style="color:{A1}">▶ Score Dataset</strong>
            to generate the risk report.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    counts = risk_df["risk_tier"].value_counts()
    rc1, rc2, rc3 = st.columns(3, gap="small")
    with rc1:
        st.markdown(f'<div class="risk-big" style="background:{HIGH_BG};border-color:{HIGH_C}30">'
                    f'<div class="risk-count" style="color:{HIGH_C}">{counts.get("High",0)}</div>'
                    f'<div class="risk-label" style="color:{HIGH_C}">⚠ High Risk</div></div>',
                    unsafe_allow_html=True)
    with rc2:
        st.markdown(f'<div class="risk-big" style="background:{MED_BG};border-color:{MED_C}30">'
                    f'<div class="risk-count" style="color:{MED_C}">{counts.get("Medium",0)}</div>'
                    f'<div class="risk-label" style="color:{MED_C}">◆ Medium Risk</div></div>',
                    unsafe_allow_html=True)
    with rc3:
        st.markdown(f'<div class="risk-big" style="background:{LOW_BG};border-color:{LOW_C}30">'
                    f'<div class="risk-count" style="color:{LOW_C}">{counts.get("Low",0)}</div>'
                    f'<div class="risk-label" style="color:{LOW_C}">✓ Low Risk</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_head("Filters")
    fc1, fc2, fc3 = st.columns([1,2,1], gap="small")
    with fc1:
        tier_filter = st.multiselect("Tier", ["High","Medium","Low"],
                                     default=["High","Medium","Low"],
                                     label_visibility="collapsed")
    with fc2:
        prob_min, prob_max = st.slider("Probability range", 0.0, 1.0, (0.0,1.0),
                                       step=0.01, label_visibility="collapsed")
    with fc3:
        n_show = st.number_input("Rows", 5, len(risk_df), len(risk_df), step=5,
                                 label_visibility="collapsed")

    filtered = risk_df[
        risk_df["risk_tier"].isin(tier_filter) &
        risk_df["dropout_probability"].between(prob_min, prob_max)
    ].head(int(n_show))

    section_head(f"Distribution — {len(filtered)} students")
    fig = px.histogram(filtered, x="dropout_probability", color="risk_tier", nbins=40,
                       color_discrete_map={"High":HIGH_C,"Medium":MED_C,"Low":LOW_C},
                       barmode="overlay", opacity=0.75)
    pcfg(fig, 260)
    fig.update_layout(xaxis_title="Dropout Probability", yaxis_title="Count",
                      legend_title="", bargap=0.04,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    section_head("Student Risk Table")
    display_cols = ["student_id","dropout_probability","risk_tier","interventions"]

    def _tier_color(val):
        return {"High":f"background-color:{HIGH_BG};color:{HIGH_C}",
                "Medium":f"background-color:{MED_BG};color:{MED_C}",
                "Low":f"background-color:{LOW_BG};color:{LOW_C}"}.get(val,"")

    st.dataframe(
        filtered[display_cols].style.map(_tier_color, subset=["risk_tier"]),
        use_container_width=True, hide_index=True,
    )

    if RISK_CSV.exists():
        st.download_button("⬇ Download Full Risk Report (CSV)",
                           data=RISK_CSV.read_bytes(),
                           file_name="risk_report.csv", mime="text/csv")
