"""Streamlit dashboard for the calibrated Knowledge Gate conversion model."""

from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    page_icon=None,
    layout="wide",
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


st.sidebar.title("Knowledge Gate")
st.sidebar.caption("Conversion Intelligence Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Model Performance",
        "Evaluation Report",
        "Daily Sales Sheet",
        "Student Explorer",
        "New Data Upload",
    ],
    index=0,
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
    st.sidebar.error(f"Model not available. Run python train.py first.\n\n{exc}")

if model_loaded:
    metrics = config.get("test_metrics", {})
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model version:** `{config.get('model_version', 'N/A')}`")
    st.sidebar.markdown(f"**Holdout ROC-AUC:** `{metrics.get('roc_auc', config.get('test_roc_auc', 0.0)):.4f}`")
    st.sidebar.markdown(f"**Holdout AP:** `{metrics.get('average_precision', config.get('average_precision', 0.0)):.4f}`")
    st.sidebar.markdown(f"**High-confidence threshold:** `{high_threshold:.2f}`")
    st.sidebar.markdown(f"**Outreach threshold:** `{outreach_threshold:.2f}`")
    if data_validation.warnings:
        st.sidebar.warning(format_validation_messages(data_validation.warnings))

if page == "Overview":
    st.title("Knowledge Gate Conversion Intelligence")
    st.markdown("Calibrated purchase-probability model for sales prioritization.")

    if not model_loaded:
        st.stop()

    high_count = int((all_scores >= high_threshold).sum())
    mid_count = int(((all_scores >= outreach_threshold) & (all_scores < high_threshold)).sum())
    low_count = int((all_scores < outreach_threshold).sum())

    holdout_summary = config.get("test_high_threshold_summary", {})
    outreach_summary = config.get("test_outreach_threshold_summary", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HIGH leads", f"{high_count:,}")
    col2.metric("MID leads", f"{mid_count:,}")
    col3.metric("LOW leads", f"{low_count:,}")
    col4.metric("Top 10% Lift", f"{config.get('top10_lift', 0.0):.2f}x")

    st.caption(
        "HIGH = strongest prospects, MID = outreach-ready prospects, LOW = nurture."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [high_count, mid_count, low_count],
            labels=["HIGH", "MID", "LOW"],
            colors=["#c0392b", "#f39c12", "#2980b9"],
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.set_title("Students by lead tier")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            all_scores[df[TARGET_COLUMN] == 0],
            bins=40,
            alpha=0.6,
            color="steelblue",
            density=True,
            label="Non-buyer",
        )
        ax.hist(
            all_scores[df[TARGET_COLUMN] == 1],
            bins=40,
            alpha=0.6,
            color="tomato",
            density=True,
            label="Buyer",
        )
        ax.axvline(high_threshold, color="red", linestyle="--", label=f"HIGH ({high_threshold:.2f})")
        ax.axvline(
            outreach_threshold,
            color="orange",
            linestyle="--",
            label=f"MID ({outreach_threshold:.2f})",
        )
        ax.set_xlabel("Predicted purchase probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title("Score separation")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Holdout Decision Policy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HIGH precision", f"{holdout_summary.get('precision', 0.0) * 100:.1f}%")
    c2.metric("HIGH recall", f"{holdout_summary.get('recall', 0.0) * 100:.1f}%")
    c3.metric("Outreach precision", f"{outreach_summary.get('precision', 0.0) * 100:.1f}%")
    c4.metric("Outreach recall", f"{outreach_summary.get('recall', 0.0) * 100:.1f}%")

elif page == "Model Performance":
    st.title("Model Performance")

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
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC-AUC = {roc_auc_score(y_holdout_arr, holdout_scores):.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curve")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(
                recall_arr,
                precision_arr,
                color="purple",
                lw=2,
                label=f"Average precision = {average_precision_score(y_holdout_arr, holdout_scores):.4f}",
            )
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-recall curve")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed conversion rate")
            ax.set_title("Calibration curve")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(holdout_scores, bins=30, kde=False, ax=ax, color="teal")
            ax.axvline(high_threshold, color="red", linestyle="--", label=f"HIGH ({high_threshold:.2f})")
            ax.axvline(
                outreach_threshold,
                color="orange",
                linestyle="--",
                label=f"Outreach ({outreach_threshold:.2f})",
            )
            ax.set_xlabel("Predicted purchase probability")
            ax.set_title("Holdout score distribution")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("ROC-AUC", f"{roc_auc_score(y_holdout_arr, holdout_scores):.4f}")
        m2.metric("Average Precision", f"{average_precision_score(y_holdout_arr, holdout_scores):.4f}")
        m3.metric("Top 20% Capture", f"{config.get('top20_capture', 0.0) * 100:.1f}%")

    with tab3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No purchase", "Purchase"],
            yticklabels=["No purchase", "Purchase"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("HIGH-threshold confusion matrix")
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True positives", int(cm[1][1]))
        c2.metric("False positives", int(cm[0][1]))
        c3.metric("False negatives", int(cm[1][0]))
        c4.metric("True negatives", int(cm[0][0]))

    with tab4:
        feature_names = config.get("feature_names", preprocessor.get_feature_names_out())
        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        top15 = feature_importance.nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        top15.plot(kind="barh", color="teal", ax=ax)
        ax.set_title("Top feature drivers")
        ax.set_xlabel("Relative importance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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

        st.success(
            f"Top 20% of students capture {config.get('top20_capture', 0.0) * 100:.1f}% "
            f"of all buyers on the holdout set."
        )

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
    st.title("Daily Sales Priority Sheet")

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
    metric_a.metric("HIGH", int((filtered["Tier"] == "HIGH").sum()))
    metric_b.metric("MID", int((filtered["Tier"] == "MID").sum()))
    metric_c.metric("LOW", int((filtered["Tier"] == "LOW").sum()))

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=450)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sales sheet CSV",
        data=csv_bytes,
        file_name="Daily_Sales_Priorities.csv",
        mime="text/csv",
    )

elif page == "Student Explorer":
    st.title("Student Explorer")
    st.markdown("Inspect an individual student probability and behavior signals.")

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
        col1.metric("Purchase probability", score_label(score))
        col2.metric("Lead tier", tier)
        col3.metric("Course interest", row["course_interest"])
        col4.metric("Already converted", "Yes" if row[TARGET_COLUMN] == 1 else "No")

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
            left_col.metric(key, value)
        for key, value in items[4:]:
            right_col.metric(key, value)

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
    st.title("Score New Students")
    st.markdown("Upload a CSV in the training schema to score new students.")

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
                    col1.metric("HIGH", int((scored_results["Tier"] == "HIGH").sum()))
                    col2.metric("MID", int((scored_results["Tier"] == "MID").sum()))
                    col3.metric("LOW", int((scored_results["Tier"] == "LOW").sum()))

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
