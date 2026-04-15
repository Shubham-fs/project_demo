"""Streamlit dashboard for the calibrated Knowledge Gate conversion model."""

from __future__ import annotations

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import streamlit as st
from data_validation import format_validation_messages, validate_dataset
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from kg_features import ID_COLUMN, TARGET_COLUMN, engineer_features, missing_columns

DATA_PATH = "knowledge_gate_Shubham_dataset.csv"
MODELS_DIR = "models"

st.set_page_config(
    page_title="Knowledge Gate Conversion Intelligence",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
/* Dark Dashboard background */
.stApp {
    background-color: #0f172a;
    background-image: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 60%);
    font-family: 'Inter', sans-serif;
    color: #f8fafc;
}

/* Global text overrides inside Streamlit */
.stMarkdown p, .stMarkdown label, .stMarkdown span {
    color: #cbd5e1 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #020617 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebarNav"] * {
    color: #cbd5e1 !important;
}

/* Typography headers with Neon Gradient */
h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900 !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 24px !important;
    text-shadow: 0px 4px 15px rgba(129, 140, 248, 0.2);
}
h2 {
    color: #f8fafc !important;
    font-weight: 800 !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
}
h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* Styled metric cards with Electric Colors */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, #2dd4bf, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] {
    font-size: 0.95rem !important;
    color: #94a3b8 !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Metric container shadow, glow, and borders */
[data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    border: 1px solid #334155;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="metric-container"]:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 30px rgba(56, 189, 248, 0.25);
    border-color: #38bdf8;
}

/* Glowing Button styling */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 800 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4) !important;
    transition: all 0.3s !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%) !important;
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.6) !important;
    transform: translateY(-2px);
}

/* Tabs styling */
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    font-weight: 700 !important;
    color: #64748b !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 3px solid #38bdf8 !important;
    text-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
}

/* Dataframe padding & dark mode */
.stDataFrame {
    padding: 1rem;
    background: #0f172a;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    border: 1px solid #1e293b;
}

/* Inputs and interactive elements */
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    border-color: #334155 !important;
    color: #f8fafc !important;
}
input {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border-color: #334155 !important;
}

/* ── KEYFRAME ANIMATIONS ───────────────────────────────────── */

/* 1. Page content fade-up entrance */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container {
    animation: fadeInUp 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* 2. Pulsing neon glow on metric cards */
@keyframes pulseGlow {
    0%   { box-shadow: 0 4px 15px rgba(56,189,248,0.12); }
    50%  { box-shadow: 0 4px 30px rgba(56,189,248,0.4), 0 0 20px rgba(168,85,247,0.15); }
    100% { box-shadow: 0 4px 15px rgba(56,189,248,0.12); }
}
[data-testid="metric-container"] {
    animation: pulseGlow 3.5s ease-in-out infinite;
}

/* 3. Shimmer sweep on h1 gradient */
@keyframes shimmer {
    0%   { background-position: -300% center; }
    100% { background-position: 300% center; }
}
h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #38bdf8, #818cf8);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    font-weight: 900 !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 24px !important;
}

/* 4. Active tab neon glow pulse */
@keyframes tabGlow {
    0%,100% { text-shadow: 0 0 6px rgba(56,189,248,0.4); }
    50%     { text-shadow: 0 0 22px rgba(56,189,248,1); }
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 3px solid #38bdf8 !important;
    animation: tabGlow 2.5s ease-in-out infinite;
}

/* 5. Button ripple ring on hover */
@keyframes buttonRing {
    0%   { box-shadow: 0 4px 15px rgba(168,85,247,0.4), 0 0 0 0 rgba(168,85,247,0.55); }
    70%  { box-shadow: 0 4px 15px rgba(168,85,247,0.4), 0 0 0 12px rgba(168,85,247,0); }
    100% { box-shadow: 0 4px 15px rgba(168,85,247,0.4), 0 0 0 0 rgba(168,85,247,0); }
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%) !important;
    animation: buttonRing 1s ease-out infinite !important;
    transform: translateY(-2px);
}

/* 6. Sidebar slide-in from left */
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-22px); }
    to   { opacity: 1; transform: translateX(0); }
}
[data-testid="stSidebar"] > div:first-child {
    animation: slideInLeft 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* 7. Floating logo bounce in sidebar */
@keyframes floatY {
    0%,100% { transform: translateY(0px); }
    50%     { transform: translateY(-6px); }
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:first-child div {
    animation: floatY 4s ease-in-out infinite;
}

/* 8. Dataframe row hover */
.stDataFrame tr:hover td {
    background: rgba(56,189,248,0.07) !important;
    transition: background 0.2s ease;
}

/* 9. Spinner pulse */
@keyframes spinnerPulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%     { opacity: 0.5; transform: scale(0.95); }
}
.stSpinner > div { animation: spinnerPulse 1.2s ease-in-out infinite !important; }
</style>
""", unsafe_allow_html=True)

def premium_metric_card(label, value, icon="⚡", color_start="#2dd4bf", color_end="#3b82f6"):
    return f"""
    <div style="background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid #334155; padding: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.4); text-align: left; transition: all 0.3s ease-in-out;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <p style="color: #94a3b8; font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin: 0; font-family: 'Inter', sans-serif;">{label}</p>
            <div style="background: rgba(255,255,255,0.1); border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 14px;">{icon}</div>
        </div>
        <p style="background: linear-gradient(90deg, {color_start}, {color_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 36px; font-weight: 900; font-family: 'Inter', sans-serif; margin: 0;">{value}</p>
    </div>
    """

# ── Chart theme constants ────────────────────────────────────────────────────
_BG       = "#0d1117"
_PANEL    = "#0f1923"
_GRID_C   = "#1a2535"
_TEXT_PRI = "#e2e8f0"
_TEXT_SEC = "#4b6080"
_CYAN     = "#22d3ee"
_PURPLE   = "#a78bfa"
_GREEN    = "#34d399"
_AMBER    = "#fbbf24"
_ROSE     = "#fb7185"

def dark_chart_style(fig, axes=None):
    """Apply a premium neon-dark Matplotlib theme."""
    fig.patch.set_facecolor(_BG)
    targets = axes if axes else fig.axes
    for ax in targets:
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_TEXT_SEC, labelsize=9)
        ax.xaxis.label.set_color(_TEXT_SEC)
        ax.yaxis.label.set_color(_TEXT_SEC)
        ax.title.set_color(_TEXT_PRI)
        ax.title.set_fontweight("bold")
        ax.title.set_fontsize(12)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_C)
        ax.grid(color=_GRID_C, linestyle="--", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor("#1e293b")
            legend.get_frame().set_edgecolor(_GRID_C)
            for text in legend.get_texts():
                text.set_color(_TEXT_PRI)
    fig.tight_layout()
    return fig

# ── Shared Plotly dark layout ────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_PANEL,
    font=dict(color=_TEXT_PRI, family="Inter, sans-serif", size=12),
    title_font=dict(color=_TEXT_PRI, size=14, family="Inter, sans-serif"),
    legend=dict(
        bgcolor="rgba(15,25,35,0.9)",
        bordercolor=_GRID_C, borderwidth=1,
        font=dict(color=_TEXT_PRI, size=11),
    ),
    xaxis=dict(
        gridcolor=_GRID_C, gridwidth=0.5,
        tickcolor=_TEXT_SEC, tickfont=dict(color=_TEXT_SEC, size=10),
        linecolor=_GRID_C, zerolinecolor=_GRID_C,
    ),
    yaxis=dict(
        gridcolor=_GRID_C, gridwidth=0.5,
        tickcolor=_TEXT_SEC, tickfont=dict(color=_TEXT_SEC, size=10),
        linecolor=_GRID_C, zerolinecolor=_GRID_C,
    ),
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor=_CYAN,
        font=dict(color=_TEXT_PRI, size=12),
    ),
    margin=dict(l=40, r=20, t=50, b=40),
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "kg_model.pkl"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "kg_preprocessor.pkl"))
    calibrator = joblib.load(os.path.join(MODELS_DIR, "kg_calibrator.pkl"))
    config = joblib.load(os.path.join(MODELS_DIR, "kg_config.pkl"))
    return model, preprocessor, calibrator, config


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def format_model_load_error(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError) and exc.name == "xgboost":
        return (
            "Model artifacts were found, but this Streamlit environment cannot import "
            "`xgboost`.\n\n"
            f"Current Python: `{sys.executable}`\n\n"
            "Start the app with the same interpreter you used for training:\n"
            "`python -m streamlit run app.py`\n\n"
            "Or install `xgboost` into the environment that provides the `streamlit` command."
        )
    return f"Model not available. Run `python train.py` first.\n\n{exc}"


def build_model_inputs(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    engineered = engineer_features(df)
    drop_columns = config.get("dropped_training_columns", [])
    model_frame = engineered.drop(
        columns=[ID_COLUMN, TARGET_COLUMN, *drop_columns],
        errors="ignore",
    )
    model_input_columns = config.get("model_input_columns", list(model_frame.columns))
    missing_model_columns = [
        column for column in model_input_columns if column not in model_frame.columns
    ]
    if missing_model_columns:
        raise ValueError(
            "Missing model columns after feature engineering: "
            + ", ".join(missing_model_columns[:10])
        )
    return engineered, model_frame.reindex(columns=model_input_columns)


def score_rows(
    df: pd.DataFrame,
    model,
    preprocessor,
    calibrator,
    config: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    engineered, model_frame = build_model_inputs(df, config)
    encoded = preprocessor.transform(model_frame)
    raw_scores = model.predict_proba(encoded)[:, 1]
    scores = np.clip(calibrator.transform(raw_scores), 0.0, 1.0)
    return engineered, scores


def get_tier(
    score: float,
    high_threshold: float,
    outreach_threshold: float,
    capacity: int = 0,
    rank: int = 0,
) -> str:
    """Assign HIGH/MID/LOW based on thresholds or call-capacity rank."""
    if capacity > 0:
        if rank <= capacity:
            return "HIGH"
        if rank <= capacity * 3:
            return "MID"
        return "LOW"
    if score >= high_threshold:
        return "HIGH"
    if score >= outreach_threshold:
        return "MID"
    return "LOW"


def get_top_signal(row: pd.Series) -> str:
    if row["ai_mentor_recency_weight"] > 5:
        return "Recent heavy AI mentor usage"
    if row["pricing_page_visits"] > 3 and row["cart_items_count"] > 0:
        return "Repeated pricing and cart activity"
    if row["engagement_intensity"] > 500:
        return "High platform engagement intensity"
    if row["current_streak_days"] >= 5:
        return "Strong recent learning streak"
    return "Active community and study behavior"


def score_label(score: float) -> str:
    return f"{score * 100:.1f}%"


def get_holdout_raw(df_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    saved_ids = config.get("test_student_ids", [])
    if not saved_ids:
        raise ValueError("No saved holdout IDs found in model config.")

    lookup = df_raw.assign(_student_id_key=df_raw[ID_COLUMN].astype(str)).set_index(
        "_student_id_key",
        drop=False,
    )
    holdout = lookup.loc[saved_ids].copy()
    return holdout.reset_index(drop=True)


@st.cache_data
def get_holdout_bundle(df_raw: pd.DataFrame, _model, _preprocessor, _calibrator, config: dict):
    holdout_raw = get_holdout_raw(df_raw, config)
    holdout_engineered, holdout_scores = score_rows(
        holdout_raw,
        _model,
        _preprocessor,
        _calibrator,
        config,
    )
    y_holdout = holdout_engineered[TARGET_COLUMN].astype(int).reset_index(drop=True)
    return holdout_raw, holdout_engineered, y_holdout, holdout_scores


def build_scored_frame(
    source_df: pd.DataFrame,
    scores: np.ndarray,
    high_threshold: float,
    outreach_threshold: float,
    actuals: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Student_ID": source_df[ID_COLUMN].astype(str).values,
            "Purchase_Probability": np.round(scores * 100, 1),
            "Predicted_Tier": [
                get_tier(score, high_threshold, outreach_threshold) for score in scores
            ],
        }
    )
    if actuals is not None:
        frame["Actual_Purchased"] = np.asarray(actuals)
    return frame.sort_values("Purchase_Probability", ascending=False).reset_index(drop=True)


def make_decile_table(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    y_array = np.asarray(y_true)
    order = np.argsort(scores)[::-1]
    sorted_actual = y_array[order]
    sorted_scores = scores[order]
    base_rate = sorted_actual.mean()
    rows = []
    n_rows = len(sorted_actual)

    for decile in range(10):
        start = int(decile * n_rows / 10)
        end = int((decile + 1) * n_rows / 10)
        decile_actual = sorted_actual[start:end]
        decile_scores = sorted_scores[start:end]
        rows.append(
            {
                "Decile": decile + 1,
                "Students": int(len(decile_actual)),
                "Actual_Buyers": int(decile_actual.sum()),
                "Conversion_Rate": f"{decile_actual.mean() * 100:.1f}%",
                "Avg_Probability": f"{decile_scores.mean() * 100:.1f}%",
                "Lift_vs_Baseline": f"{decile_actual.mean() / (base_rate + 1e-9):.2f}x",
            }
        )

    return pd.DataFrame(rows)


with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 20px 0;">
        <div style="
            display: inline-flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            border-radius: 50%; width: 56px; height: 56px;
            box-shadow: 0 0 20px rgba(168, 85, 247, 0.5);
            margin-bottom: 12px; font-size: 26px;
        ">🎓</div>
        <div style="
            font-size: 18px; font-weight: 900;
            background: linear-gradient(90deg, #38bdf8, #a855f7);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px; line-height: 1.2;
        ">Knowledge Gate</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 4px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;">Intelligence Suite</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='border-top: 1px solid #1e293b; margin-bottom: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; color:#475569; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px; padding: 0 8px;'>Navigation</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "Evaluation Report",
        "Daily Sales Sheet",
        "Student Explorer",
        "Live Predictor",
        "New Data Upload",
    ],
    index=0,
    label_visibility="collapsed"
)

try:
    model, preprocessor, calibrator, config = load_artifacts()
    df_raw = load_data()
    data_validation = validate_dataset(
        df_raw,
        expected_columns=[ID_COLUMN, *config.get("raw_feature_columns", []), TARGET_COLUMN],
        require_target=True,
    )
    if data_validation.errors:
        raise ValueError(format_validation_messages(data_validation.errors))
    df, all_scores = score_rows(df_raw, model, preprocessor, calibrator, config)
    holdout_raw, holdout_df, y_holdout, holdout_scores = get_holdout_bundle(
        df_raw,
        model,
        preprocessor,
        calibrator,
        config,
    )
    high_threshold = float(config.get("high_threshold", config["hot_threshold"]))
    outreach_threshold = float(config.get("outreach_threshold", config["warm_threshold"]))
    model_loaded = True
except Exception as exc:
    model_loaded = False
    st.sidebar.error(format_model_load_error(exc))

if model_loaded:
    metrics = config.get("test_metrics", {})
    roc_val = metrics.get('roc_auc', config.get('test_roc_auc', 0.0))
    ap_val = metrics.get('average_precision', config.get('average_precision', 0.0))
    st.sidebar.markdown(f"""
    <div style="margin: 8px 0 4px 0; padding: 14px 16px; background: rgba(30,41,59,0.7); border-radius: 12px; border: 1px solid #1e293b;">
        <div style="font-size:10px; color:#475569; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;">Model Health</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:12px; color:#94a3b8;">ROC-AUC</span>
            <span style="font-size:13px; font-weight:800; color:#38bdf8;">{roc_val:.4f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:12px; color:#94a3b8;">Avg Precision</span>
            <span style="font-size:13px; font-weight:800; color:#a855f7;">{ap_val:.4f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:12px; color:#94a3b8;">HIGH Threshold</span>
            <span style="font-size:13px; font-weight:800; color:#10b981;">{high_threshold:.2f}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size:12px; color:#94a3b8;">MID Threshold</span>
            <span style="font-size:13px; font-weight:800; color:#f59e0b;">{outreach_threshold:.2f}</span>
        </div>
    </div>
    <div style="margin: 10px 0 4px 0; padding: 12px 16px; background: rgba(30,41,59,0.7); border-radius: 12px; border: 1px solid #1e293b;">
        <div style="font-size:10px; color:#475569; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;">Tier Legend</div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="font-size:14px;">🔥</span><span style="font-size:12px; color:#10b981; font-weight:700;">HIGH</span><span style="font-size:11px; color:#64748b; margin-left:auto;">Top prospects</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="font-size:14px;">⚡</span><span style="font-size:12px; color:#f59e0b; font-weight:700;">MID</span><span style="font-size:11px; color:#64748b; margin-left:auto;">Outreach ready</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:14px;">🧊</span><span style="font-size:12px; color:#3b82f6; font-weight:700;">LOW</span><span style="font-size:11px; color:#64748b; margin-left:auto;">Nurture only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if data_validation.warnings:
        st.sidebar.warning(format_validation_messages(data_validation.warnings))

if page == "Overview":
    st.markdown("<h1>Knowledge Gate Conversion Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    high_count = int((all_scores >= high_threshold).sum())
    mid_count = int(((all_scores >= outreach_threshold) & (all_scores < high_threshold)).sum())
    low_count = int((all_scores < outreach_threshold).sum())

    holdout_summary = config.get("test_high_threshold_summary", {})
    outreach_summary = config.get("test_outreach_threshold_summary", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(premium_metric_card("HIGH leads", f"{high_count:,}", "🔥", "#10b981", "#34d399"), unsafe_allow_html=True)
    with col2:
        st.markdown(premium_metric_card("MID leads", f"{mid_count:,}", "⚡", "#f59e0b", "#fbbf24"), unsafe_allow_html=True)
    with col3:
        st.markdown(premium_metric_card("LOW leads", f"{low_count:,}", "🧊", "#3b82f6", "#60a5fa"), unsafe_allow_html=True)
    with col4:
        st.markdown(premium_metric_card("Top 10% Lift", f"{config.get('top10_lift', 0.0):.2f}x", "📈", "#8b5cf6", "#c084fc"), unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = go.Figure(go.Pie(
            labels=["HIGH", "MID", "LOW"],
            values=[high_count, mid_count, low_count],
            marker=dict(
                colors=[_GREEN, _AMBER, _CYAN],
                line=dict(color=_BG, width=3),
            ),
            hole=0.45,
            textfont=dict(color=_BG, size=13),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            title="Students by Lead Tier",
            showlegend=True,
            transition={"duration": 800, "easing": "cubic-in-out"},
            annotations=[dict(text="Tiers", x=0.5, y=0.5, font_size=13, showarrow=False,
                              font_color=_TEXT_PRI)],
        )
        fig_pie.update_traces(rotation=90, pull=[0.05, 0.03, 0.02])
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        non_buyer_scores = all_scores[df[TARGET_COLUMN] == 0]
        buyer_scores = all_scores[df[TARGET_COLUMN] == 1]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=non_buyer_scores, nbinsx=40, name="Non-buyer",
            marker_color=_CYAN, opacity=0.55, histnorm="density",
            hovertemplate="Prob: %{x:.2f}<br>Density: %{y:.3f}<extra>Non-buyer</extra>",
        ))
        fig_hist.add_trace(go.Histogram(
            x=buyer_scores, nbinsx=40, name="Buyer",
            marker_color=_ROSE, opacity=0.55, histnorm="density",
            hovertemplate="Prob: %{x:.2f}<br>Density: %{y:.3f}<extra>Buyer</extra>",
        ))
        fig_hist.add_vline(x=high_threshold, line=dict(color=_ROSE, dash="dash", width=2),
                           annotation_text=f"HIGH {high_threshold:.2f}",
                           annotation_font_color=_ROSE)
        fig_hist.add_vline(x=outreach_threshold, line=dict(color=_AMBER, dash="dash", width=2),
                           annotation_text=f"MID {outreach_threshold:.2f}",
                           annotation_font_color=_AMBER)
        fig_hist.update_layout(**PLOTLY_LAYOUT, title="Score Separation",
                               barmode="overlay",
                               transition={"duration": 700, "easing": "elastic-out"},
                               xaxis_title="Predicted Purchase Probability",
                               yaxis_title="Density")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Holdout Decision Policy")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(premium_metric_card("HIGH precision", f"{holdout_summary.get('precision', 0.0) * 100:.1f}%", "🎯", "#2dd4bf", "#3b82f6"), unsafe_allow_html=True)
    with c2:
        st.markdown(premium_metric_card("HIGH recall", f"{holdout_summary.get('recall', 0.0) * 100:.1f}%", "🔁", "#8b5cf6", "#d946ef"), unsafe_allow_html=True)
    with c3:
        st.markdown(premium_metric_card("Outreach precision", f"{outreach_summary.get('precision', 0.0) * 100:.1f}%", "🎯", "#f43f5e", "#f97316"), unsafe_allow_html=True)
    with c4:
        st.markdown(premium_metric_card("Outreach recall", f"{outreach_summary.get('recall', 0.0) * 100:.1f}%", "🔁", "#14b8a6", "#10b981"), unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown("<h1>Model Performance Metrics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top:-10px;'>Detailed evaluation curves and calibration data.</p><br>", unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    y_holdout_arr = np.asarray(y_holdout)
    fpr, tpr, _ = roc_curve(y_holdout_arr, holdout_scores)
    precision_arr, recall_arr, _ = precision_recall_curve(y_holdout_arr, holdout_scores)
    prob_true, prob_pred = calibration_curve(y_holdout_arr, holdout_scores, n_bins=10)
    predicted = (holdout_scores >= high_threshold).astype(int)
    cm = confusion_matrix(y_holdout_arr, predicted)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["ROC & PR", "Calibration", "Confusion Matrix", "Feature Importance"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            roc_auc_val = roc_auc_score(y_holdout_arr, holdout_scores)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"ROC-AUC = {roc_auc_val:.4f}",
                line=dict(color=_CYAN, width=2.5),
                fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>Model</extra>",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(color=_GRID_C, dash="dash", width=1),
                hoverinfo="skip",
            ))
            fig_roc.update_layout(**PLOTLY_LAYOUT, title="ROC Curve",
                                  xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate",
                                  transition={"duration": 900, "easing": "cubic-in-out"})
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            ap_val = average_precision_score(y_holdout_arr, holdout_scores)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=recall_arr, y=precision_arr, mode="lines",
                name=f"Avg Precision = {ap_val:.4f}",
                line=dict(color=_PURPLE, width=2.5),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.06)",
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra>Model</extra>",
            ))
            fig_pr.update_layout(**PLOTLY_LAYOUT, title="Precision-Recall Curve",
                                 xaxis_title="Recall", yaxis_title="Precision",
                                 transition={"duration": 900, "easing": "cubic-in-out"})
            st.plotly_chart(fig_pr, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=prob_pred, y=prob_true, mode="lines+markers", name="Model",
                line=dict(color=_GREEN, width=2.5),
                marker=dict(color=_BG, size=8, line=dict(color=_GREEN, width=2)),
                hovertemplate="Predicted: %{x:.2f}<br>Actual rate: %{y:.2f}<extra>Calibration</extra>",
            ))
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                line=dict(color=_GRID_C, dash="dash", width=1.5), hoverinfo="skip",
            ))
            fig_cal.update_layout(**PLOTLY_LAYOUT, title="Calibration Curve",
                                  xaxis_title="Mean Predicted Probability",
                                  yaxis_title="Observed Conversion Rate",
                                  transition={"duration": 800, "easing": "back-out"})
            st.plotly_chart(fig_cal, use_container_width=True)

        with col2:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=holdout_scores, nbinsx=30, name="Score distribution",
                marker_color=_PURPLE, opacity=0.8,
                hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
            ))
            fig_dist.add_vline(x=high_threshold, line=dict(color=_ROSE, dash="dash", width=2.5),
                               annotation_text=f"HIGH {high_threshold:.2f}",
                               annotation_font_color=_ROSE)
            fig_dist.add_vline(x=outreach_threshold, line=dict(color=_AMBER, dash="dash", width=2.5),
                               annotation_text=f"MID {outreach_threshold:.2f}",
                               annotation_font_color=_AMBER)
            fig_dist.update_layout(**PLOTLY_LAYOUT, title="Holdout Score Distribution",
                                   xaxis_title="Predicted Purchase Probability",
                                   transition={"duration": 700, "easing": "elastic-out"})
            st.plotly_chart(fig_dist, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(premium_metric_card("ROC-AUC", f"{roc_auc_score(y_holdout_arr, holdout_scores):.4f}", "📊", "#6366f1", "#a855f7"), unsafe_allow_html=True)
        with m2:
            st.markdown(premium_metric_card("Average Precision", f"{average_precision_score(y_holdout_arr, holdout_scores):.4f}", "🎯", "#ef4444", "#f97316"), unsafe_allow_html=True)
        with m3:
            st.markdown(premium_metric_card("Top 20% Capture", f"{config.get('top20_capture', 0.0) * 100:.1f}%", "📈", "#10b981", "#3b82f6"), unsafe_allow_html=True)

    with tab3:
        labels = ["No purchase", "Purchase"]
        z_text = [[str(cm[r][c]) for c in range(2)] for r in range(2)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=labels, y=labels,
            colorscale=[[0, _PANEL], [1, _PURPLE]],
            text=z_text, texttemplate="<b>%{text}</b>",
            textfont=dict(size=22, color="white"),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=False,
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, title="HIGH-Threshold Confusion Matrix",
                             xaxis_title="Predicted", yaxis_title="Actual",
                             transition={"duration": 600, "easing": "cubic-in-out"})
        st.plotly_chart(fig_cm, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(premium_metric_card("True positives", int(cm[1][1]), "✅", "#10b981", "#34d399"), unsafe_allow_html=True)
        with c2:
            st.markdown(premium_metric_card("False positives", int(cm[0][1]), "⚠️", "#f43f5e", "#fb7185"), unsafe_allow_html=True)
        with c3:
            st.markdown(premium_metric_card("False negatives", int(cm[1][0]), "❌", "#8b5cf6", "#c084fc"), unsafe_allow_html=True)
        with c4:
            st.markdown(premium_metric_card("True negatives", int(cm[0][0]), "☑️", "#3b82f6", "#60a5fa"), unsafe_allow_html=True)

    with tab4:
        feature_names = config.get("feature_names", preprocessor.get_feature_names_out())
        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        top15 = feature_importance.nlargest(15).sort_values()
        colors_feat = px.colors.sample_colorscale("Plasma", [i / len(top15) for i in range(len(top15))])
        fig_fi = go.Figure(go.Bar(
            x=top15.values,
            y=top15.index,
            orientation="h",
            marker=dict(color=colors_feat, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, title="Top Feature Drivers",
                             xaxis_title="Relative Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

elif page == "Evaluation Report":
    st.title("Evaluation Report")
    st.markdown("Blind holdout evaluation using the exact student IDs saved at training time.")

    if not model_loaded:
        st.stop()

    y_holdout_arr = np.asarray(y_holdout)
    y_pred_high = (holdout_scores >= high_threshold).astype(int)

    st.markdown("### Holdout Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{roc_auc_score(y_holdout_arr, holdout_scores):.4f}")
    c2.metric("Average Precision", f"{average_precision_score(y_holdout_arr, holdout_scores):.4f}")
    c3.metric("High-confidence precision", f"{config.get('test_high_threshold_summary', {}).get('precision', 0.0) * 100:.1f}%")
    c4.metric("Holdout size", f"{len(y_holdout_arr):,}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Lift Chart", "Prediction Table", "Full Dataset", "Decile Analysis"]
    )

    with tab1:
        st.subheader("Cumulative Lift")
        order = np.argsort(holdout_scores)[::-1]
        sorted_actual = y_holdout_arr[order]
        cumulative_buyers = np.cumsum(sorted_actual)
        total_buyers = sorted_actual.sum()
        pct_leads = np.arange(1, len(sorted_actual) + 1) / len(sorted_actual) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(
            pct_leads,
            cumulative_buyers / (total_buyers + 1e-9) * 100,
            color="teal",
            lw=2,
            label="Model",
        )
        axes[0].plot([0, 100], [0, 100], "k--", lw=1, label="Random baseline")
        axes[0].set_xlabel("% students contacted")
        axes[0].set_ylabel("% buyers captured")
        axes[0].set_title("Cumulative buyer capture")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        base_rate = sorted_actual.mean()
        decile_lifts = []
        for decile in range(10):
            start = int(decile * len(sorted_actual) / 10)
            end = int((decile + 1) * len(sorted_actual) / 10)
            decile_rate = sorted_actual[start:end].mean()
            decile_lifts.append(decile_rate / (base_rate + 1e-9))
        axes[1].bar(range(1, 11), decile_lifts, color="steelblue", alpha=0.85)
        axes[1].axhline(1.0, color="red", linestyle="--", label="Random = 1.0x")
        axes[1].set_xlabel("Decile (1 = highest probability)")
        axes[1].set_ylabel("Lift over random")
        axes[1].set_title("Lift by decile")
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Holdout Predictions")
        pred_df = build_scored_frame(
            holdout_raw,
            holdout_scores,
            high_threshold,
            outreach_threshold,
            actuals=y_holdout_arr,
        )
        pred_df["Correct"] = (y_pred_high == y_holdout_arr)[
            np.argsort(holdout_scores)[::-1]
        ].astype(int)

        col1, col2 = st.columns(2)
        with col1:
            show_filter = st.selectbox(
                "Show",
                ["All", "Correct only", "Wrong only", "Buyers only", "Non-buyers only"],
            )
        with col2:
            tier_filter = st.multiselect(
                "Tier",
                ["HIGH", "MID", "LOW"],
                default=["HIGH", "MID", "LOW"],
            )

        display_df = pred_df[pred_df["Predicted_Tier"].isin(tier_filter)]
        if show_filter == "Correct only":
            display_df = display_df[display_df["Correct"] == 1]
        elif show_filter == "Wrong only":
            display_df = display_df[display_df["Correct"] == 0]
        elif show_filter == "Buyers only":
            display_df = display_df[display_df["Actual_Purchased"] == 1]
        elif show_filter == "Non-buyers only":
            display_df = display_df[display_df["Actual_Purchased"] == 0]

        st.markdown(f"Showing **{len(display_df):,}** rows")
        st.dataframe(display_df, use_container_width=True, height=450)

        csv_test = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download holdout predictions CSV",
            csv_test,
            "holdout_predictions.csv",
            "text/csv",
        )

    with tab3:
        st.subheader("Full Dataset Score Preview")
        full_df = pd.DataFrame(
            {
                "Student_ID": df_raw[ID_COLUMN].astype(str).values,
                "Purchase_Probability": np.round(all_scores * 100, 1),
                "Predicted_Tier": [
                    get_tier(score, high_threshold, outreach_threshold)
                    for score in all_scores
                ],
                "Actual_Purchased": df[TARGET_COLUMN].values,
                "Course_Interest": df_raw["course_interest"].values,
                "Days_Since_Login": df_raw["last_login_days_ago"].values,
            }
        ).sort_values("Purchase_Probability", ascending=False).reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total students", f"{len(full_df):,}")
        c2.metric("Actual buyers", f"{full_df['Actual_Purchased'].sum():,}")
        c3.metric("HIGH tier count", f"{(full_df['Predicted_Tier'] == 'HIGH').sum():,}")

        st.dataframe(full_df.head(100), use_container_width=True, height=400)
        st.caption("Preview shows the top 100 rows ranked by probability.")

        csv_full = full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full scored dataset",
            csv_full,
            "full_scored_dataset.csv",
            "text/csv",
        )

    with tab4:
        st.subheader("Decile Analysis")
        decile_table = make_decile_table(y_holdout_arr, holdout_scores)
        st.dataframe(decile_table, use_container_width=True)
        st.info(
            "A healthy ranking model should show the highest conversion rate in Decile 1 "
            "and drop steadily as you move toward Decile 10."
        )

elif page == "Daily Sales Sheet":
    st.markdown("<h1>Daily Sales Priority Sheet</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top:-10px;'>Filter and download optimized outreach pipelines.</p><br>", unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    unconverted_mask = df[TARGET_COLUMN] == 0
    df_unconverted = df.loc[unconverted_mask].reset_index(drop=True)
    unconverted_scores = all_scores[unconverted_mask.values]

    st.markdown("### Settings")
    capacity_col, course_col, days_col = st.columns(3)
    with capacity_col:
        daily_capacity = st.number_input(
            "Daily call capacity (0 = use thresholds)",
            min_value=0,
            max_value=5000,
            value=100,
            step=10,
            help="Top N students by score = HIGH, next 3N = MID, rest = LOW",
        )

    sorted_indices = np.argsort(unconverted_scores)[::-1]
    ranks = np.empty(len(unconverted_scores), dtype=int)
    ranks[sorted_indices] = np.arange(1, len(unconverted_scores) + 1)

    sales_df = pd.DataFrame(
        {
            "Student_ID": df_unconverted[ID_COLUMN].astype(str).values,
            "Purchase_Probability": np.round(unconverted_scores * 100, 1),
            "Tier": [
                get_tier(score, high_threshold, outreach_threshold, daily_capacity, rank)
                for score, rank in zip(unconverted_scores, ranks)
            ],
            "Top_Signal": [get_top_signal(row) for _, row in df_unconverted.iterrows()],
            "Course_Interest": df_unconverted["course_interest"].values,
            "Days_Since_Login": df_unconverted["last_login_days_ago"].values,
        }
    ).sort_values("Purchase_Probability", ascending=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        tier_filter = st.multiselect("Tier", ["HIGH", "MID", "LOW"], default=["HIGH", "MID"])
    with col2:
        course_options = ["All"] + sorted(df_unconverted["course_interest"].unique().tolist())
        course_filter = st.selectbox("Course Interest", course_options)
    with col3:
        days_filter = st.slider("Max Days Since Login", 0, 60, 30)

    filtered = sales_df[sales_df["Tier"].isin(tier_filter)]
    if course_filter != "All":
        filtered = filtered[filtered["Course_Interest"] == course_filter]
    filtered = filtered[filtered["Days_Since_Login"] <= days_filter]

    st.markdown(f"### Showing {len(filtered):,} students")
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.markdown(premium_metric_card("HIGH", int((filtered["Tier"] == "HIGH").sum()), "🔥", "#10b981", "#34d399"), unsafe_allow_html=True)
    with metric_b:
        st.markdown(premium_metric_card("MID", int((filtered["Tier"] == "MID").sum()), "⚡", "#f59e0b", "#fbbf24"), unsafe_allow_html=True)
    with metric_c:
        st.markdown(premium_metric_card("LOW", int((filtered["Tier"] == "LOW").sum()), "🧊", "#3b82f6", "#60a5fa"), unsafe_allow_html=True)

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=450)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sales sheet CSV",
        data=csv_bytes,
        file_name="Daily_Sales_Priorities.csv",
        mime="text/csv",
    )

elif page == "Student Explorer":
    st.markdown("<h1>Student Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top:-10px;'>Inspect individual student probability and behavior signals.</p><br>", unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    student_ids = df[ID_COLUMN].astype(str).tolist()
    selected_id = st.selectbox("Select Student ID", student_ids)

    if selected_id:
        idx = df[df[ID_COLUMN].astype(str) == selected_id].index[0]
        score = all_scores[df.index.get_loc(idx)]
        tier = get_tier(score, high_threshold, outreach_threshold)
        row = df.loc[idx]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(premium_metric_card("Probability", score_label(score), "🎯", "#2dd4bf", "#3b82f6"), unsafe_allow_html=True)
        with col2:
            st.markdown(premium_metric_card("Lead tier", tier, "🏆", "#f43f5e", "#f97316"), unsafe_allow_html=True)
        with col3:
            st.markdown(premium_metric_card("Course", row["course_interest"], "📚", "#8b5cf6", "#c084fc"), unsafe_allow_html=True)
        with col4:
            st.markdown(premium_metric_card("Converted", "Yes" if row[TARGET_COLUMN] == 1 else "No", "✅", "#10b981", "#34d399"), unsafe_allow_html=True)

        st.subheader("Key behavioral signals")
        signals = {
            "AI Mentor Messages": row["ai_mentor_total_messages"],
            "AI Mentor Recency Score": round(row["ai_mentor_recency_weight"], 2),
            "Pricing Page Visits": row["pricing_page_visits"],
            "Cart Items": row["cart_items_count"],
            "Community Depth Score": round(row["community_depth"], 2),
            "Mock Test Commitment": round(row["mock_commitment"], 2),
            "Current Streak Days": row["current_streak_days"],
            "Days Since Login": row["last_login_days_ago"],
        }

        left_col, right_col = st.columns(2)
        items = list(signals.items())
        for key, value in items[:4]:
            with left_col:
                st.markdown(premium_metric_card(key, value, "🔮", "#6366f1", "#a855f7"), unsafe_allow_html=True)
        for key, value in items[4:]:
            with right_col:
                st.markdown(premium_metric_card(key, value, "✨", "#14b8a6", "#2dd4bf"), unsafe_allow_html=True)

        st.subheader("Top engagement signal")
        st.info(get_top_signal(row))

        try:
            import shap

            st.subheader("SHAP explanation")
            st.caption("SHAP explains the base XGBoost ranking model before calibration.")
            with st.spinner("Computing SHAP values..."):
                _, student_input = build_model_inputs(df.loc[[idx]], config)
                student_encoded = preprocessor.transform(student_input)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(student_encoded)
                feature_names = config.get("feature_names", preprocessor.get_feature_names_out())

                shap_series = (
                    pd.Series(shap_values[0], index=feature_names)
                    .abs()
                    .sort_values(ascending=False)
                    .head(10)
                )
                fig, ax = plt.subplots(figsize=(7, 4))
                shap_series.sort_values().plot(kind="barh", color="teal", ax=ax)
                ax.set_title(f"Top SHAP drivers for {selected_id}")
                ax.set_xlabel("Absolute SHAP value")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        except ImportError:
            st.info("Install shap to see feature-level explanations.")

elif page == "New Data Upload":
    st.markdown("<h1>Score New Students</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-top:-10px;'>Upload a CSV in the training schema to score new students.</p><br>", unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        new_df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(new_df):,} rows")
        st.dataframe(new_df.head(), use_container_width=True)

        required_columns = [ID_COLUMN, *config.get("raw_feature_columns", [])]
        upload_validation = validate_dataset(
            new_df,
            expected_columns=required_columns,
            require_target=False,
        )
        missing_required = missing_columns(new_df, required_columns)
        if missing_required:
            st.error(
                "Upload is missing required columns: "
                + ", ".join(missing_required[:15])
            )
            st.stop()

        if upload_validation.errors:
            st.error(format_validation_messages(upload_validation.errors))
            st.stop()
        if upload_validation.warnings:
            st.warning(format_validation_messages(upload_validation.warnings))

        if st.button("Score uploaded students"):
            with st.spinner("Scoring..."):
                try:
                    engineered_upload, new_scores = score_rows(
                        new_df,
                        model,
                        preprocessor,
                        calibrator,
                        config,
                    )
                    scored_results = pd.DataFrame(
                        {
                            "Student_ID": new_df[ID_COLUMN].astype(str).values,
                            "Purchase_Probability": np.round(new_scores * 100, 1),
                            "Tier": [
                                get_tier(score, high_threshold, outreach_threshold)
                                for score in new_scores
                            ],
                            "Top_Signal": [
                                get_top_signal(row)
                                for _, row in engineered_upload.iterrows()
                            ],
                            "Course_Interest": new_df["course_interest"].values,
                        }
                    ).sort_values("Purchase_Probability", ascending=False)

                    st.subheader("Scored results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(premium_metric_card("HIGH", int((scored_results["Tier"] == "HIGH").sum()), "🔥", "#10b981", "#34d399"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(premium_metric_card("MID", int((scored_results["Tier"] == "MID").sum()), "⚡", "#f59e0b", "#fbbf24"), unsafe_allow_html=True)
                    with col3:
                        st.markdown(premium_metric_card("LOW", int((scored_results["Tier"] == "LOW").sum()), "🧊", "#3b82f6", "#60a5fa"), unsafe_allow_html=True)

                    st.dataframe(scored_results, use_container_width=True)

                    csv_bytes = scored_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download scored sheet",
                        csv_bytes,
                        "Scored_New_Students.csv",
                        "text/csv",
                    )

                    st.info(
                        "Direct appends into the training dataset are disabled. "
                        "Review new labels through a controlled retraining workflow."
                    )
                except Exception as exc:
                    st.error(f"Error scoring uploaded data: {exc}")

elif page == "Live Predictor":
    st.markdown("<h1>⚡ Live Student Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    st.markdown("""
    <div style="background: rgba(99,102,241,0.1); border: 1px solid #6366f1; border-radius: 14px; padding: 16px 20px; margin-bottom: 24px;">
        <p style="margin:0; color:#a5b4fc; font-size:14px; font-weight:600;">
            🎯 Fill in the student's behavioral signals below. The model will instantly predict the probability of purchasing a paid course.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predictor_form"):

        # --- Section 1: Student Profile ---
        st.markdown("### 👤 Student Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            course_interest = st.selectbox("Course Interest", ["GATE", "Placement", "GovtJobs", "NET", "Other"])
        with c2:
            student_stage = st.selectbox("Student Stage", ["Awareness", "Consideration", "Decision", "Cold Lead"])
        with c3:
            acquisition_channel = st.selectbox("Acquisition Channel", ["Referral", "YouTube", "Self Login"])

        c4, c5, c6 = st.columns(3)
        with c4:
            registration_days_ago = st.number_input("Days Since Registration", 0, 3000, 90)
        with c5:
            profile_completion_pct = st.slider("Profile Completion (%)", 0, 100, 60)
        with c6:
            targets_count = st.number_input("Targets Count", 0, 10, 1)

        st.markdown("---")

        # --- Section 2: Platform Engagement ---
        st.markdown("### 📊 Platform Engagement")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            login_frequency_14d = st.number_input("Logins (last 14 days)", 0, 28, 5)
        with c2:
            total_platform_minutes = st.number_input("Total Platform Minutes", 0, 50000, 600)
        with c3:
            active_days_30d = st.number_input("Active Days (30d)", 0, 30, 8)
        with c4:
            current_streak_days = st.number_input("Current Streak (days)", 0, 365, 3)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            session_depth = st.number_input("Session Depth", 0, 500, 20)
        with c6:
            last_login_days_ago = st.number_input("Days Since Last Login", 0, 365, 2)
        with c7:
            free_platform_lectures_watched = st.number_input("Free Lectures Watched", 0, 1000, 15)
        with c8:
            onboarding_video_watched = st.selectbox("Onboarding Video Watched", [0, 1])

        st.markdown("---")

        # --- Section 3: AI Mentor ---
        st.markdown("### 🤖 AI Mentor Activity")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ai_mentor_total_messages = st.number_input("AI Mentor Messages", 0, 2000, 10)
        with c2:
            ai_mentor_sessions = st.number_input("AI Mentor Sessions", 0, 500, 3)
        with c3:
            ai_mentor_last_used_days_ago = st.number_input("AI Mentor Last Used (days ago)", 0, 999, 5)
        with c4:
            ai_mentor_topic = st.selectbox("AI Mentor Topic", ["none", "GATE", "Maths", "Coding", "Resume", "Interview", "Other"])

        c5, c6 = st.columns(2)
        with c5:
            ai_mentor_daily_limit_hit = st.selectbox("Hit Daily Limit?", [0, 1])
        with c6:
            ai_mentor_limit_hit_days = st.number_input("Limit Hit Days", 0, 100, 0)

        st.markdown("---")

        # --- Section 4: Purchase Intent ---
        st.markdown("### 🛒 Purchase Intent Signals")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pricing_page_visits = st.number_input("Pricing Page Visits", 0, 100, 2)
        with c2:
            cart_items_count = st.number_input("Cart Items Count", 0, 20, 0)
        with c3:
            coupon_code_applied = st.selectbox("Coupon Applied?", [0, 1])
        with c4:
            wishlist_count = st.number_input("Wishlist Count", 0, 50, 0)

        c5, c6, c7 = st.columns(3)
        with c5:
            demo_class_attended = st.selectbox("Demo Class Attended?", [0, 1])
        with c6:
            rank_predictor_used = st.selectbox("Rank Predictor Used?", [0, 1])
        with c7:
            rank_predictor_course_clicked = st.selectbox("Course Clicked via Rank Predictor?", [0, 1])

        st.markdown("---")

        # --- Section 5: Community & Social ---
        st.markdown("### 👥 Community & Social")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            community_posts_count = st.number_input("Community Posts", 0, 500, 2)
        with c2:
            community_comments_count = st.number_input("Community Comments", 0, 1000, 5)
        with c3:
            community_saved_posts = st.number_input("Saved Posts", 0, 500, 3)
        with c4:
            communities_joined_count = st.number_input("Communities Joined", 0, 50, 1)

        st.markdown("---")

        # --- Section 6: Tests & Academic ---
        st.markdown("### 📝 Tests & Academic Performance")
        c1, c2, c3 = st.columns(3)
        with c1:
            free_mock_test_count = st.number_input("Mock Tests Taken", 0, 200, 3)
        with c2:
            free_mock_test_avg_score = st.slider("Mock Test Avg Score (%)", 0, 100, 50)
        with c3:
            pyq_attempted_count = st.number_input("PYQs Attempted", 0, 5000, 50)

        c4, c5 = st.columns(2)
        with c4:
            pyq_accuracy_pct = st.slider("PYQ Accuracy (%)", 0, 100, 55)
        with c5:
            gate_exam_urgency = st.selectbox("GATE Exam Urgency", ["low", "medium", "high"])

        st.markdown("---")

        # --- Section 7: Counsellor & Outreach ---
        st.markdown("### 📞 Counsellor & Outreach")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            counsellor_call_received = st.selectbox("Call Received?", [0, 1])
        with c2:
            counsellor_call_picked_up = st.selectbox("Call Picked Up?", [0, 1])
        with c3:
            counsellor_call_duration_mins = st.number_input("Call Duration (mins)", 0, 60, 0)
        with c4:
            call_duration_bucket = st.selectbox("Call Duration Bucket", ["none", "short", "medium", "long"])

        c5, c6 = st.columns(2)
        with c5:
            whatsapp_outreach_received = st.selectbox("WhatsApp Outreach?", [0, 1])
        with c6:
            whatsapp_replied = st.selectbox("WhatsApp Replied?", [0, 1])

        st.markdown("---")

        # --- Section 8: Urgency & Timing ---
        st.markdown("### ⏰ Urgency & Timing")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            urgency_weight = st.slider("Urgency Weight (0–10)", 0.0, 10.0, 1.0, step=0.1)
        with c2:
            course_urgency_score = st.slider("Course Urgency Score (0–1)", 0.0, 1.0, 0.3, step=0.01)
        with c3:
            days_to_next_exam = st.number_input("Days to Next Exam", 0, 730, 90)
        with c4:
            is_peak_placement_season = st.selectbox("Peak Placement Season?", [0, 1])

        c5, c6 = st.columns(2)
        with c5:
            post_result_window = st.number_input("Post Result Window (days)", 0, 180, 0)
        with c6:
            active_month = st.number_input("Active Month (1–12)", 1, 12, 6)

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🚀 Predict Purchase Probability", use_container_width=True)

    if submitted:
        # Build raw input row matching the expected schema
        input_data = {
            "student_id": ["PREVIEW_001"],
            "acquisition_channel": [acquisition_channel],
            "course_interest": [course_interest],
            "student_stage": [student_stage],
            "registration_days_ago": [registration_days_ago],
            "targets_count": [targets_count],
            "target_gate": [1 if course_interest == "GATE" else 0],
            "target_placement": [1 if course_interest == "Placement" else 0],
            "target_govt_jobs": [1 if course_interest == "GovtJobs" else 0],
            "login_frequency_14d": [login_frequency_14d],
            "total_platform_minutes": [total_platform_minutes],
            "session_depth": [session_depth],
            "last_login_days_ago": [last_login_days_ago],
            "most_active_time_window": ["evening"],
            "current_streak_days": [current_streak_days],
            "active_days_30d": [active_days_30d],
            "beta_mode_switched": [0],
            "free_platform_lectures_watched": [free_platform_lectures_watched],
            "pyq_attempted_count": [pyq_attempted_count],
            "pyq_accuracy_pct": [pyq_accuracy_pct],
            "free_mock_test_count": [free_mock_test_count],
            "free_mock_test_avg_score": [free_mock_test_avg_score],
            "onboarding_video_watched": [onboarding_video_watched],
            "ai_mentor_sessions": [ai_mentor_sessions],
            "ai_mentor_total_messages": [ai_mentor_total_messages],
            "ai_mentor_last_used_days_ago": [ai_mentor_last_used_days_ago],
            "ai_mentor_daily_limit_hit": [ai_mentor_daily_limit_hit],
            "ai_mentor_limit_hit_days": [ai_mentor_limit_hit_days],
            "ai_mentor_image_uploaded": [0],
            "ai_mentor_topic": [ai_mentor_topic],
            "rank_predictor_used": [rank_predictor_used],
            "rank_predictor_coupon_copied": [0],
            "rank_predictor_course_clicked": [rank_predictor_course_clicked],
            "gate_exam_urgency": [gate_exam_urgency],
            "communities_joined_count": [communities_joined_count],
            "community_posts_count": [community_posts_count],
            "community_comments_count": [community_comments_count],
            "community_saved_posts": [community_saved_posts],
            "community_image_posted": [0],
            "chat_initiated_count": [0],
            "study_group_joined": [0],
            "study_group_created": [0],
            "followers_count": [0],
            "following_count": [0],
            "pricing_page_visits": [pricing_page_visits],
            "demo_class_attended": [demo_class_attended],
            "cart_items_count": [cart_items_count],
            "wishlist_count": [wishlist_count],
            "coupon_code_applied": [coupon_code_applied],
            "profile_completion_pct": [profile_completion_pct],
            "has_profile_picture": [1 if profile_completion_pct > 40 else 0],
            "has_education_filled": [1 if profile_completion_pct > 60 else 0],
            "has_career_goals": [1 if profile_completion_pct > 70 else 0],
            "resume_uploaded": [0],
            "standard_certificates": [0],
            "job_notifications_enabled": [0],
            "job_alerts_clicked_count": [0],
            "job_applications_count": [0],
            "resume_review_requested": [0],
            "resume_review_completed": [0],
            "mock_interview_requested": [0],
            "mock_interview_completed": [0],
            "question_reports_count": [0],
            "feedback_submitted": [0],
            "platform_rating": [0],
            "referral_program_visited": [0],
            "successful_referrals": [0],
            "active_month": [active_month],
            "course_urgency_score": [course_urgency_score],
            "days_to_next_exam": [days_to_next_exam],
            "post_result_window": [post_result_window],
            "is_peak_placement_season": [is_peak_placement_season],
            "whatsapp_outreach_received": [whatsapp_outreach_received],
            "whatsapp_replied": [whatsapp_replied],
            "counsellor_call_received": [counsellor_call_received],
            "counsellor_call_picked_up": [counsellor_call_picked_up],
            "counsellor_call_duration_mins": [counsellor_call_duration_mins],
            "call_duration_bucket": [call_duration_bucket],
            "purchased_paid_course": [0],
            "daily_avg_minutes": [total_platform_minutes / max(active_days_30d, 1)],
            "engagement_intensity": [total_platform_minutes * login_frequency_14d / 14],
            "intent_score": [pricing_page_visits * 10 + cart_items_count * 20],
            "is_professional": [0],
            "pro_intent_interaction": [0],
            "effective_call_minutes": [counsellor_call_duration_mins if counsellor_call_picked_up else 0],
            "urgency_weight": [urgency_weight],
        }

        try:
            with st.spinner("🔮 Model is computing prediction..."):
                input_df = pd.DataFrame(input_data)
                engineered_input, score_arr = score_rows(input_df, model, preprocessor, calibrator, config)
                predicted_score = float(score_arr[0])
                tier = get_tier(predicted_score, high_threshold, outreach_threshold)
                top_signal = get_top_signal(engineered_input.iloc[0])

            pct = predicted_score * 100

            if tier == "HIGH":
                tier_color = "#10b981"
                tier_glow = "rgba(16, 185, 129, 0.35)"
                tier_emoji = "🔥"
                tier_advice = "Immediate priority call — high buying intent confirmed."
                bar_color = "#10b981, #34d399"
            elif tier == "MID":
                tier_color = "#f59e0b"
                tier_glow = "rgba(245, 158, 11, 0.35)"
                tier_emoji = "⚡"
                tier_advice = "Outreach ready — warm lead, schedule a follow-up call."
                bar_color = "#f59e0b, #fbbf24"
            else:
                tier_color = "#3b82f6"
                tier_glow = "rgba(59, 130, 246, 0.25)"
                tier_emoji = "🧊"
                tier_advice = "Nurture via email drip. Not yet ready for direct sales."
                bar_color = "#3b82f6, #60a5fa"

            st.markdown(f"""
            <div style="background: rgba(15,23,42,0.9); border: 1px solid {tier_color}; border-radius: 20px; padding: 32px; margin-top: 24px;
                        box-shadow: 0 0 40px {tier_glow};">
                <div style="text-align:center; margin-bottom: 24px;">
                    <div style="font-size: 52px; margin-bottom: 8px;">{tier_emoji}</div>
                    <div style="font-size: 13px; font-weight: 700; color: #64748b; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;">Purchase Probability</div>
                    <div style="font-size: 80px; font-weight: 900; color: {tier_color}; text-shadow: 0 0 30px {tier_glow}, 0 0 60px {tier_glow}; line-height: 1; -webkit-text-fill-color: {tier_color};">
                        {pct:.1f}%
                    </div>
                    <div style="margin: 16px auto; background: rgba(30,41,59,0.8); border-radius: 50px; height: 12px; width: 80%; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, {bar_color}); height: 100%; width: {pct:.1f}%; border-radius: 50px; box-shadow: 0 0 10px {tier_glow};"></div>
                    </div>
                </div>
                <div style="display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;">
                    <div style="background: rgba(30,41,59,0.7); border-radius: 12px; padding: 16px 24px; text-align:center; border: 1px solid #1e293b; flex:1; min-width: 140px;">
                        <div style="font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px;">Lead Tier</div>
                        <div style="font-size: 24px; font-weight: 900; color: {tier_color};">{tier_emoji} {tier}</div>
                    </div>
                    <div style="background: rgba(30,41,59,0.7); border-radius: 12px; padding: 16px 24px; text-align:center; border: 1px solid #1e293b; flex: 2; min-width: 200px;">
                        <div style="font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px;">Recommended Action</div>
                        <div style="font-size: 14px; font-weight: 700; color: #e2e8f0;">{tier_advice}</div>
                    </div>
                    <div style="background: rgba(30,41,59,0.7); border-radius: 12px; padding: 16px 24px; text-align:center; border: 1px solid #1e293b; flex: 2; min-width: 200px;">
                        <div style="font-size: 11px; color: #64748b; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px;">Top Engagement Signal</div>
                        <div style="font-size: 14px; font-weight: 700; color: #a5b4fc;">{top_signal}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Prediction error: {exc}")
