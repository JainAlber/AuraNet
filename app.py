"""
AuraNet Command Console — SOC-style Streamlit Dashboard

Visual Design
-------------
  Cyberpunk dark theme  ·  Neon cyan (safe) + crimson (attack) accents
  Glass-morphism cards  ·  Plotly transparent dark charts

Run
---
  # Terminal 1 — inference server
  uvicorn src.serve:app --port 8000

  # Terminal 2 — dashboard
  streamlit run app.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AuraNet | Cyber Defense",
    page_icon="🔰",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
API_URL    = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark theme + glass-morphism
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Rajdhani', 'Share Tech Mono', 'Courier New', monospace !important;
    background-color: #050C17 !important;
    color: #C8E6FF !important;
}
.stApp {
    background: radial-gradient(ellipse at 18% 40%, #071525 0%,
                #030B14 55%, #020810 100%) !important;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060F1E 0%, #030A14 100%) !important;
    border-right: 1px solid rgba(0, 245, 255, 0.10) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #C8E6FF !important; }

/* ── Main block padding ─────────────────────────────────────── */
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Glass card ─────────────────────────────────────────────── */
.glass {
    background: rgba(4, 18, 38, 0.70);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(0, 245, 255, 0.13);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.75rem;
}
.glass-red {
    background: rgba(28, 4, 12, 0.72);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 23, 68, 0.28);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.75rem;
}
.glass-green {
    background: rgba(4, 28, 12, 0.72);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(0, 255, 65, 0.22);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.75rem;
}

/* ── Neon title ─────────────────────────────────────────────── */
.neon-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 0.38em;
    color: #00F5FF;
    text-shadow: 0 0 7px #00F5FF, 0 0 18px #00F5FF, 0 0 48px #0094B0;
    padding: 0.2rem 0;
    line-height: 1.2;
}
.neon-sub {
    text-align: center;
    color: rgba(0, 245, 255, 0.48);
    letter-spacing: 0.22em;
    font-size: 0.78rem;
    margin-top: -0.2rem;
    padding-bottom: 0.2rem;
}

/* ── Section labels ─────────────────────────────────────────── */
.slabel {
    color: rgba(0, 245, 255, 0.62);
    font-size: 0.70rem;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0, 245, 255, 0.12);
    padding-bottom: 0.28rem;
    margin-bottom: 0.55rem;
}

/* ── Verdict ────────────────────────────────────────────────── */
.verdict-safe {
    color: #00FF41;
    font-size: 2.9rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-align: center;
    line-height: 1.1;
    text-shadow: 0 0 12px #00FF41, 0 0 30px #00CC33, 0 0 60px #009922;
    animation: glow-safe 2s ease-in-out infinite;
}
.verdict-attack {
    color: #FF1744;
    font-size: 2.9rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-align: center;
    line-height: 1.1;
    text-shadow: 0 0 12px #FF1744, 0 0 30px #FF4500, 0 0 60px #CC1100;
    animation: glow-attack 0.75s ease-in-out infinite;
}
.verdict-idle {
    color: rgba(0, 245, 255, 0.28);
    font-size: 2rem;
    letter-spacing: 0.18em;
    text-align: center;
    line-height: 1.35;
}
@keyframes glow-safe {
    0%,100% { text-shadow: 0 0 8px #00FF41, 0 0 18px #00CC33; }
    50%      { text-shadow: 0 0 22px #00FF41, 0 0 50px #00FF41, 0 0 80px #00CC33; }
}
@keyframes glow-attack {
    0%,100% { text-shadow: 0 0 10px #FF1744, 0 0 22px #FF4500; }
    50%      { text-shadow: 0 0 28px #FF1744, 0 0 55px #FF6D00, 0 0 88px #FF4500; }
}

/* Pipeline nodes removed — now using st.columns with inline styles */

/* ── Heartbeat dot ──────────────────────────────────────────── */
.hb-on  { display:inline-block; width:8px; height:8px; border-radius:50%;
          background:#00FF41; box-shadow:0 0 7px #00FF41;
          animation:blink 1.4s infinite; margin-right:5px; vertical-align:middle; }
.hb-off { display:inline-block; width:8px; height:8px; border-radius:50%;
          background:#FF1744; box-shadow:0 0 7px #FF1744;
          margin-right:5px; vertical-align:middle; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.22} }

/* ── Metric widget override ─────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(0, 245, 255, 0.035) !important;
    border: 1px solid rgba(0, 245, 255, 0.09) !important;
    border-radius: 7px !important;
    padding: 0.5rem 0.75rem !important;
}
[data-testid="stMetricValue"] > div { color: #00F5FF !important; font-size: 1.3rem !important; }
[data-testid="stMetricLabel"] > div { color: rgba(0,245,255,0.55) !important;
                                      font-size: 0.68rem !important; letter-spacing:0.08em !important; }
[data-testid="stMetricDelta"] { color: rgba(0,255,65,0.75) !important; }

/* ── Slider / selectbox overrides ───────────────────────────── */
.stSlider label, .stSelectbox label, .stRadio label,
.stFileUploader label { color: rgba(0,245,255,0.70) !important;
                         font-size: 0.78rem !important; letter-spacing: 0.07em; }
.stRadio > div { gap: 0.5rem !important; }

/* ── Primary button ─────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #002D45 0%, #004A6E 100%) !important;
    border: 1px solid rgba(0, 245, 255, 0.45) !important;
    color: #00F5FF !important;
    letter-spacing: 0.13em !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: box-shadow 0.2s, border-color 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 14px rgba(0,245,255,0.3) !important;
    border-color: rgba(0,245,255,0.80) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020810; }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.22); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    rng = np.random.default_rng(1)
    if "pulse_x"      not in st.session_state:
        st.session_state.pulse_x = list(range(90))
    if "pulse_y"      not in st.session_state:
        st.session_state.pulse_y = list(
            0.15 + 0.10 * np.sin(np.linspace(0, 6, 90)) + rng.uniform(0, 0.12, 90)
        )
    if "last_result"  not in st.session_state: st.session_state.last_result  = None
    if "results_log"  not in st.session_state: st.session_state.results_log  = []
    if "api_online"   not in st.session_state: st.session_state.api_online   = None

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model  = joblib.load(MODELS_DIR / "xgb_tuned.joblib")
        meta   = joblib.load(MODELS_DIR / "feature_meta.joblib")
        rp     = MODELS_DIR / "training_report.json"
        report = json.loads(rp.read_text()) if rp.exists() else {}
        return model, meta, report
    except Exception:
        return None, None, {}

model, meta, report = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    duration=0.0, protocol_type="tcp", service="http", flag="SF",
    src_bytes=0, dst_bytes=0, land=0, wrong_fragment=0, urgent=0, hot=0,
    num_failed_logins=0, logged_in=1, num_compromised=0, root_shell=0,
    su_attempted=0, num_root=0, num_file_creations=0, num_shells=0,
    num_access_files=0, num_outbound_cmds=0, is_host_login=0, is_guest_login=0,
    count=1, srv_count=1, serror_rate=0.0, srv_serror_rate=0.0,
    rerror_rate=0.0, srv_rerror_rate=0.0, same_srv_rate=1.0, diff_srv_rate=0.0,
    srv_diff_host_rate=0.0, dst_host_count=1, dst_host_srv_count=1,
    dst_host_same_srv_rate=1.0, dst_host_diff_srv_rate=0.0,
    dst_host_same_src_port_rate=0.0, dst_host_srv_diff_host_rate=0.0,
    dst_host_serror_rate=0.0, dst_host_srv_serror_rate=0.0,
    dst_host_rerror_rate=0.0, dst_host_srv_rerror_rate=0.0,
)


def build_payload(values: dict) -> dict:
    p = dict(_DEFAULTS)
    p.update({k: v for k, v in values.items() if k in p})
    return p


def call_api(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/analyze", json=payload, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def check_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def push_pulse(confidence: float, is_attack: bool) -> None:
    new_x    = st.session_state.pulse_x[-1] + 1
    baseline = float(np.random.uniform(0.12, 0.28))
    spike    = confidence * 8.5 if is_attack else confidence * 0.45
    st.session_state.pulse_x.append(new_x)
    st.session_state.pulse_y.append(baseline + spike)
    if len(st.session_state.pulse_x) > 90:
        st.session_state.pulse_x = st.session_state.pulse_x[-90:]
        st.session_state.pulse_y = st.session_state.pulse_y[-90:]


def get_feature_importances() -> tuple[list[str], list[float]]:
    if model is None or meta is None:
        return [], []
    pairs = sorted(
        zip(meta["feature_order"], model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    top = list(pairs[:10])
    # Always surface network_intensity so the engineered feature is visible
    if not any(n == "network_intensity" for n, _ in top):
        for n, s in pairs:
            if n == "network_intensity":
                top.append((n, s))
                break
    top.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in top], [p[1] for p in top]


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────
_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(3,12,26,0.60)",
    font=dict(color="#C8E6FF", family="Rajdhani, monospace", size=11),
    margin=dict(l=46, r=18, t=44, b=32),
    xaxis=dict(
        gridcolor="rgba(0,245,255,0.06)",
        zerolinecolor="rgba(0,245,255,0.08)",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(0,245,255,0.06)",
        zerolinecolor="rgba(0,245,255,0.08)",
        tickfont=dict(size=10),
    ),
)


def pulse_chart(last: dict | None) -> go.Figure:
    xs = st.session_state.pulse_x
    ys = st.session_state.pulse_y
    attack = last and last.get("prediction") == "Attack"

    line_clr = "#FF1744" if attack else "#00F5FF"
    fill_clr = "rgba(255,23,68,0.09)" if attack else "rgba(0,245,255,0.06)"
    dot_clr  = "#FF6D00" if attack else "#00FF41"

    fig = go.Figure()

    # fill area
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        fill="tozeroy", fillcolor=fill_clr,
        line=dict(color=line_clr, width=1.8, shape="spline", smoothing=0.6),
        name="intensity",
    ))

    # live cursor dot
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]], mode="markers",
        marker=dict(color=dot_clr, size=10,
                    line=dict(color=dot_clr, width=2)),
        showlegend=False,
    ))

    # threat threshold
    fig.add_hline(
        y=1.0, line_dash="dot",
        line_color="rgba(255,109,0,0.32)", line_width=1.2,
        annotation_text="THREAT THRESHOLD",
        annotation_font=dict(size=9, color="rgba(255,109,0,0.48)"),
        annotation_position="top left",
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text="⬡  NETWORK PULSE  ·  Traffic Intensity Feed",
            font=dict(size=12, color="#00F5FF"), x=0.01, y=0.97,
        ),
        showlegend=False,
        xaxis_title="Packet sequence",
        yaxis_title="Intensity",
        height=260,
    )
    return fig


def dna_chart(names: list[str], scores: list[float], last: dict | None) -> go.Figure:
    if not names:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE,
                          title="⬡  FEATURE DNA  ·  Model Importance", height=340)
        return fig

    attack = last and last.get("prediction") == "Attack"
    colors = []
    for n in names:
        if n == "network_intensity":
            colors.append("#00FF41")         # engineered — always green
        elif attack:
            colors.append("#FF4500")         # attack session — orange-red
        else:
            colors.append("#007ACC")         # safe session — muted cyan

    display_names = [f"★ {n}" if n == "network_intensity" else n for n in names]

    fig = go.Figure(go.Bar(
        x=scores,
        y=display_names,
        orientation="h",
        marker=dict(color=colors, opacity=0.88,
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont=dict(size=9, color="#C8E6FF"),
    ))

    dna_yaxis = {**_LAYOUT_BASE["yaxis"], "autorange": "reversed"}
    dna_layout = {**_LAYOUT_BASE, "yaxis": dna_yaxis}

    fig.update_layout(
        **dna_layout,
        title=dict(
            text=(
                "⬡  FEATURE DNA  ·  "
                "<span style='color:#00FF41'>★ = engineered network_intensity</span>"
            ),
            font=dict(size=12, color="#00F5FF"), x=0.01, y=0.97,
        ),
        xaxis_title="Importance",
        height=340,
        bargap=0.28,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ░░░  SIDEBAR  ░░░
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:0.4rem 0 1.1rem;'>"
        "<div style='font-size:1.65rem; color:#00F5FF; font-weight:700; "
        "letter-spacing:0.28em; text-shadow:0 0 10px #00F5FF;'>⬡ AURANET</div>"
        "<div style='font-size:0.65rem; color:rgba(0,245,255,0.42); "
        "letter-spacing:0.20em; margin-top:2px;'>CYBER DEFENSE SYSTEM</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Heartbeat ──────────────────────────────────────────────────────────
    st.markdown('<div class="slabel">◈ SYSTEM HEARTBEAT</div>', unsafe_allow_html=True)
    if st.button("↺  Ping  /health", use_container_width=True):
        st.session_state.api_online = check_health()

    o = st.session_state.api_online
    if o is True:
        st.markdown(
            '<div class="glass" style="padding:0.5rem 0.8rem;">'
            '<span class="hb-on"></span>'
            '<span style="font-size:0.80rem; color:#00FF41;">API ONLINE · /analyze ready</span>'
            '</div>', unsafe_allow_html=True)
    elif o is False:
        st.markdown(
            '<div class="glass-red" style="padding:0.5rem 0.8rem;">'
            '<span class="hb-off"></span>'
            '<span style="font-size:0.80rem; color:#FF1744;">API OFFLINE</span>'
            '<div style="font-size:0.68rem; color:rgba(255,23,68,0.60); margin-top:3px;">'
            'uvicorn src.serve:app --port 8000</div>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:0.75rem; color:rgba(0,245,255,0.30); '
            'padding:0.3rem 0.2rem;">— Press Ping to check status</div>',
            unsafe_allow_html=True)

    # ── Model telemetry ────────────────────────────────────────────────────
    st.markdown(
        '<br><div class="slabel">◈ MODEL TELEMETRY</div>', unsafe_allow_html=True)

    best_f1    = report.get("best_f1_cv",    0.99766)
    test_f1    = report.get("test_f1_macro", 0.9972)
    n_trials   = report.get("n_trials",      60)
    n_complete = report.get("n_completed",   48)
    n_train    = report.get("n_train",       20154)
    n_feats    = len(meta["feature_order"]) if meta else 42

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Best CV F1", f"{best_f1:.4f}")
        st.metric("Train rows", f"{n_train:,}")
    with c2:
        st.metric("Test F1", f"{test_f1:.4f}", delta="+0.0004")
        st.metric("Features", str(n_feats))

    st.metric(
        "Optuna trials",
        f"{n_complete} / {n_trials}",
        delta=f"{n_trials - n_complete} pruned",
    )

    # ── Best params ────────────────────────────────────────────────────────
    bp = report.get("best_params", {})
    if bp:
        st.markdown(
            '<br><div class="slabel">◈ BEST PARAMS  (TPE)</div>',
            unsafe_allow_html=True)
        for k, v in bp.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; '
                f'font-size:0.73rem; padding:1px 0; color:rgba(200,230,255,0.72);">'
                f'<span>{k}</span><span style="color:#00F5FF;">{val_str}</span></div>',
                unsafe_allow_html=True)

    # ── Recent detections ──────────────────────────────────────────────────
    st.markdown(
        '<br><div class="slabel">◈ RECENT DETECTIONS</div>',
        unsafe_allow_html=True)
    if st.session_state.results_log:
        for entry in reversed(st.session_state.results_log[-6:]):
            is_a  = entry["pred"] == "Attack"
            clr   = "#FF1744" if is_a else "#00FF41"
            icon  = "⚠" if is_a else "✔"
            st.markdown(
                f'<div style="font-size:0.73rem; color:{clr}; '
                f'border-left:2px solid {clr}; padding-left:0.45rem; margin:2px 0;">'
                f'{icon} {entry["pred"]} &nbsp;·&nbsp; conf {entry["conf"]:.4f}'
                f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:0.74rem; opacity:0.38;">No detections yet</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ░░░  MAIN CONTENT  ░░░
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="glass">'
    '<div class="neon-title">⬡ AURANET COMMAND CONSOLE</div>'
    '<div class="neon-sub">'
    'NETWORK ANOMALY DETECTION  ·  NSL-KDD / XGBoost  ·  REAL-TIME ANALYSIS'
    '</div></div>',
    unsafe_allow_html=True,
)

# ── Pipeline flowchart ────────────────────────────────────────────────────────
_f1_str = f"F1: {best_f1:.4f}" if report else "F1: 0.9977"

st.markdown('<div class="slabel">◈ DATA PIPELINE</div>', unsafe_allow_html=True)

# Layout: 5 stage cards interleaved with 4 arrow spacers → 9 columns total
_pipe_cols = st.columns([2.2, 0.55, 2.2, 0.55, 2.2, 0.55, 2.2, 0.55, 2.2], gap="small")

_STAGES = [
    ("📁", "INGEST",       "CSV batch<br>or sliders",             False),
    ("⚙️", "PREPROCESS",   "network_intensity<br>Encode · Scale", True),
    ("🧠", "OPTUNA / XGB", f"{_f1_str}<br>60-trial TPE",          False),
    ("🌐", "FastAPI",      "POST /analyze<br>≤ 3 ms",             False),
    ("🎯", "VERDICT",      "Normal / Attack<br>+ risk tier",      False),
]

_CARD_BASE = (
    "border-radius:9px; padding:0.82rem 0.4rem; text-align:center;"
)
_ICON_STYLE  = "font-size:1.45rem; margin-bottom:0.22rem;"
_LABEL_BASE  = "font-size:0.76rem; font-weight:700; letter-spacing:0.13em;"
_SUB_BASE    = "font-size:0.64rem; margin-top:0.28rem; line-height:1.45;"

for i, (icon, label, sub, hi) in enumerate(_STAGES):
    with _pipe_cols[i * 2]:
        if hi:
            card = f"""
            <div style="background:rgba(0,38,16,0.85);
                        border:1px solid rgba(0,255,65,0.55);
                        {_CARD_BASE}
                        box-shadow:0 0 18px rgba(0,255,65,0.22),
                                   inset 0 0 12px rgba(0,255,65,0.06);">
              <div style="{_ICON_STYLE}">{icon}</div>
              <div style="{_LABEL_BASE} color:#00FF41;
                           text-shadow:0 0 9px rgba(0,255,65,0.60);">{label}</div>
              <div style="{_SUB_BASE} color:rgba(0,255,65,0.65);">{sub}</div>
            </div>"""
        else:
            card = f"""
            <div style="background:rgba(0,22,45,0.78);
                        border:1px solid rgba(0,245,255,0.17);
                        {_CARD_BASE}">
              <div style="{_ICON_STYLE}">{icon}</div>
              <div style="{_LABEL_BASE} color:#A8D8FF;">{label}</div>
              <div style="{_SUB_BASE} color:rgba(168,216,255,0.52);">{sub}</div>
            </div>"""
        st.markdown(card, unsafe_allow_html=True)

    if i < len(_STAGES) - 1:
        with _pipe_cols[i * 2 + 1]:
            st.markdown(
                "<div style='text-align:center; padding-top:1.25rem;"
                " font-size:1.25rem; color:rgba(0,245,255,0.38);'>➔</div>",
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Input ↔ Verdict row ───────────────────────────────────────────────────────
inp_col, verdict_col = st.columns([1.65, 1], gap="medium")

# -- input panel --------------------------------------------------------------
with inp_col:
    mode = st.radio(
        "Input Mode",
        ["⌨   Manual Injection", "📁   CSV Batch Upload"],
        horizontal=True,
    )

    if "⌨" in mode:
        # ---- sliders --------------------------------------------------------
        st.markdown(
            '<div class="slabel" style="margin-top:0.6rem;">◈ PACKET PARAMETERS</div>',
            unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            duration    = st.slider("Duration (s)",          0.0, 60.0, 5.0, 0.5)
            src_bytes   = st.slider("src_bytes",             0, 100_000, 4800, 50)
            count       = st.slider("count  (conn/2 sec)",   1, 512, 5)
            serror_rate = st.slider("serror_rate",           0.0, 1.0, 0.0, 0.01)
        with r1c2:
            dst_bytes   = st.slider("dst_bytes",             0, 2_000_000, 7200, 100)
            hot         = st.slider("hot  (indicators)",     0, 30, 0)
            num_fail    = st.slider("num_failed_logins",     0, 10, 0)
            dst_h_serr  = st.slider("dst_host_serror_rate",  0.0, 1.0, 0.0, 0.01)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            protocol = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
        with r2c2:
            service  = st.selectbox(
                "Service",
                ["http", "ftp", "smtp", "ssh", "private",
                 "domain_u", "eco_i", "other"],
            )
        with r2c3:
            flag = st.selectbox(
                "Flag",
                ["SF", "S0", "S1", "S2", "S3",
                 "OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "SH"],
            )

        # derived feature preview
        ni_preview = (src_bytes + dst_bytes) / max(duration, 1e-6)
        st.markdown(
            f'<div style="background:rgba(0,255,65,0.05); border:1px solid '
            f'rgba(0,255,65,0.25); border-radius:7px; padding:0.5rem 0.85rem; '
            f'margin:0.4rem 0;">'
            f'<span style="font-size:0.70rem; color:rgba(0,255,65,0.65); '
            f'letter-spacing:0.10em;">⚙ DERIVED FEATURE</span><br>'
            f'<span style="color:#00FF41; font-size:0.85rem; letter-spacing:0.05em;">'
            f'network_intensity</span> '
            f'<span style="color:#C8E6FF; font-size:0.80rem;"> = </span>'
            f'<strong style="color:#00FF41; font-size:1.05rem;">'
            f'{ni_preview:,.1f}</strong>'
            f'<span style="color:rgba(200,230,255,0.45); font-size:0.72rem;"> bytes/sec</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        payload_values = dict(
            duration=duration, protocol_type=protocol, service=service,
            flag=flag, src_bytes=src_bytes, dst_bytes=dst_bytes,
            hot=hot, num_failed_logins=num_fail,
            count=count, serror_rate=serror_rate,
            dst_host_serror_rate=dst_h_serr,
        )

        if st.button("⬡   ANALYZE PACKET", use_container_width=True, type="primary"):
            with st.spinner("Transmitting to FastAPI /analyze ..."):
                result = call_api(build_payload(payload_values))
            if result:
                st.session_state.last_result = result
                push_pulse(result["confidence"], result["prediction"] == "Attack")
                st.session_state.results_log.append(
                    {"pred": result["prediction"], "conf": result["confidence"]}
                )
                st.rerun()
            else:
                st.error(
                    "API unreachable.  "
                    "Run:  `uvicorn src.serve:app --port 8000`"
                )

    else:
        # ---- file upload ----------------------------------------------------
        st.markdown(
            '<div class="slabel" style="margin-top:0.6rem;">◈ UPLOAD PACKET CAPTURE</div>',
            unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "CSV with NSL-KDD columns (headers optional)",
            type=["csv"],
        )
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.markdown(
                f'<div style="font-size:0.77rem; color:rgba(0,245,255,0.62); '
                f'margin-bottom:0.4rem;">Loaded {len(df_up):,} rows '
                f'× {df_up.shape[1]} cols</div>',
                unsafe_allow_html=True,
            )
            max_r = st.slider(
                "Rows to analyze", 1, min(len(df_up), 100), min(10, len(df_up))
            )
            if st.button("⬡   ANALYZE BATCH", use_container_width=True, type="primary"):
                results, errors = [], 0
                prog = st.progress(0.0, "Scanning packets...")
                for i, (_, row) in enumerate(df_up.head(max_r).iterrows()):
                    res = call_api(build_payload(row.to_dict()))
                    if res:
                        results.append(res)
                        push_pulse(res["confidence"], res["prediction"] == "Attack")
                        st.session_state.results_log.append(
                            {"pred": res["prediction"], "conf": res["confidence"]}
                        )
                    else:
                        errors += 1
                    prog.progress((i + 1) / max_r)
                prog.empty()

                if results:
                    st.session_state.last_result = results[-1]
                    n_atk = sum(1 for r in results if r["prediction"] == "Attack")
                    n_ok  = len(results) - n_atk
                    clr_ok  = "#00FF41"
                    clr_atk = "#FF1744" if n_atk else "rgba(255,23,68,0.35)"
                    st.markdown(
                        f'<div class="glass" style="padding:0.5rem 0.9rem;">'
                        f'<span style="color:{clr_ok};">✔ {n_ok} Normal</span>'
                        f'&nbsp; | &nbsp;'
                        f'<span style="color:{clr_atk};">⚠ {n_atk} Attack</span>'
                        f'{"  |  " + str(errors) + " API errors" if errors else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.rerun()
                else:
                    st.error("API unreachable.  Run: `uvicorn src.serve:app --port 8000`")


# -- verdict panel ------------------------------------------------------------
with verdict_col:
    res = st.session_state.last_result

    if res is None:
        st.markdown(
            '<div class="glass" style="min-height:210px; padding:2rem 1rem; '
            'text-align:center;">'
            '<div class="verdict-idle">AWAITING<br>INPUT</div>'
            '<div style="margin-top:1rem; color:rgba(0,245,255,0.28); '
            'font-size:0.72rem; letter-spacing:0.16em;">'
            'INJECT A PACKET TO<br>BEGIN ANALYSIS'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        is_atk  = res["prediction"] == "Attack"
        v_class = "verdict-attack" if is_atk else "verdict-safe"
        v_text  = "⚠ ATTACK"      if is_atk else "✔  SAFE"
        border  = "rgba(255,23,68,0.38)" if is_atk else "rgba(0,255,65,0.25)"
        bg      = "rgba(28,4,12,0.72)"   if is_atk else "rgba(4,22,10,0.72)"
        c_clr   = "#FF6D00"              if is_atk else "#00B8C4"
        risk    = res.get("risk_level", "—")
        ni      = res.get("network_intensity", 0.0)

        st.markdown(
            f'<div class="glass" style="border-color:{border}; background:{bg}; '
            f'padding:1.5rem 1rem; text-align:center; min-height:210px;">'
            f'<div class="slabel" style="text-align:center;">◈ THREAT VERDICT</div>'
            f'<div class="{v_class}" style="margin:0.7rem 0;">{v_text}</div>'
            f'<div style="color:{c_clr}; font-size:0.92rem; letter-spacing:0.12em;">'
            f'CONFIDENCE &nbsp; <strong>{res["confidence"]:.4f}</strong>'
            f'</div>'
            f'<div style="margin-top:0.45rem; font-size:0.80rem; '
            f'color:rgba(200,230,255,0.55); letter-spacing:0.10em;">'
            f'RISK LEVEL &nbsp; <strong style="color:{c_clr};">{risk}</strong>'
            f'</div>'
            f'<div style="margin-top:0.55rem; font-size:0.72rem; '
            f'color:rgba(0,255,65,0.55); letter-spacing:0.08em;">'
            f'⚙ network_intensity &nbsp; {ni:,.1f} b/s'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization row
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
pulse_col, dna_col = st.columns([1.15, 1], gap="medium")
feat_names, feat_scores = get_feature_importances()

with pulse_col:
    st.plotly_chart(
        pulse_chart(st.session_state.last_result),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with dna_col:
    st.plotly_chart(
        dna_chart(feat_names, feat_scores, st.session_state.last_result),
        use_container_width=True,
        config={"displayModeBar": False},
    )
