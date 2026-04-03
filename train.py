"""Train a calibrated Knowledge Gate student conversion model."""

from __future__ import annotations

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from data_validation import (
    MODEL_SCHEMA_COLUMNS,
    format_validation_messages,
    validate_dataset,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from kg_features import (
    ID_COLUMN,
    TARGET_COLUMN,
    engineer_features,
    training_leak_columns,
)

DATA_PATH = "knowledge_gate_Shubham_dataset.csv"
MODELS_DIR = "models"
COURSE_PRICE_INR = 4999
CALL_COST_INR = 150
RANDOM_STATE = 42
TEST_SIZE = 0.15
CALIBRATION_SIZE = 0.18
CV_FOLDS = 3
HIGH_PRECISION_TARGET = 0.80

MODEL_CANDIDATES = [
    {
        "name": "balanced_depth4",
        "params": {
            "max_depth": 4,
            "learning_rate": 0.03,
            "n_estimators": 400,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
        },
    },
    {
        "name": "balanced_depth5",
        "params": {
            "max_depth": 5,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "min_child_weight": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_lambda": 1.0,
        },
    },
    {
        "name": "steady_depth6",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.02,
            "n_estimators": 700,
            "min_child_weight": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_lambda": 1.5,
        },
    },
]

os.makedirs(MODELS_DIR, exist_ok=True)


def build_model_frame(df: pd.DataFrame, dropped_training_columns: list[str]) -> pd.DataFrame:
    """Return the exact feature frame used by the model."""
    return df.drop(
        columns=[ID_COLUMN, TARGET_COLUMN, *dropped_training_columns],
        errors="ignore",
    )


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a robust preprocessor for mixed numeric/categorical data."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [column for column in X.columns if column not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    return preprocessor, cat_cols, num_cols


def make_model(scale_pos_weight: float, params: dict) -> XGBClassifier:
    """Create an XGBoost classifier with stable defaults for tabular ranking."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **params,
    )


def probability_metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> dict:
    """Return probability-quality metrics for a score vector."""
    y_array = np.asarray(y_true)
    precision_arr, recall_arr, _ = precision_recall_curve(y_array, scores)
    return {
        "roc_auc": float(roc_auc_score(y_array, scores)),
        "average_precision": float(average_precision_score(y_array, scores)),
        "pr_auc": float(auc(recall_arr, precision_arr)),
        "brier_score": float(brier_score_loss(y_array, scores)),
    }


def summarize_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Summarize business and classification performance at a given threshold."""
    y_array = np.asarray(y_true)
    predictions = (scores >= threshold).astype(int)
    true_positive = int(((predictions == 1) & (y_array == 1)).sum())
    false_positive = int(((predictions == 1) & (y_array == 0)).sum())
    false_negative = int(((predictions == 0) & (y_array == 1)).sum())
    precision = true_positive / (true_positive + false_positive + 1e-9)
    recall = true_positive / (true_positive + false_negative + 1e-9)
    coverage = float(predictions.mean())
    profit = (true_positive * COURSE_PRICE_INR) - (
        (true_positive + false_positive) * CALL_COST_INR
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "coverage": coverage,
        "profit": int(profit),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Build a compact, score-aware threshold grid."""
    grid = np.unique(np.round(scores, 4))
    anchors = np.array(
        [
            CALL_COST_INR / COURSE_PRICE_INR,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]
    )
    return np.unique(np.clip(np.concatenate([grid, anchors]), 0.001, 0.999))


def select_profit_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> dict:
    """Pick the outreach threshold that maximizes validation profit."""
    best_summary: dict | None = None
    best_key: tuple[int, float, float] | None = None

    for threshold in candidate_thresholds(scores):
        summary = summarize_threshold(y_true, scores, float(threshold))
        key = (summary["profit"], summary["precision"], summary["recall"])
        if best_key is None or key > best_key:
            best_key = key
            best_summary = summary

    if best_summary is None:
        raise ValueError("Unable to select an outreach threshold.")

    return best_summary


def select_high_confidence_threshold(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    min_precision: float,
) -> dict:
    """Pick the lowest threshold that still meets the required precision."""
    candidates = [
        summarize_threshold(y_true, scores, float(threshold))
        for threshold in candidate_thresholds(scores)
    ]
    valid = [
        summary
        for summary in candidates
        if summary["true_positive"] > 0 and summary["precision"] >= min_precision
    ]
    if valid:
        return max(valid, key=lambda summary: (summary["recall"], summary["precision"]))

    return max(
        candidates,
        key=lambda summary: (
            ((1 + 0.5**2) * summary["precision"] * summary["recall"])
            / ((0.5**2 * summary["precision"]) + summary["recall"] + 1e-9),
            summary["precision"],
        ),
    )


def lift_at_percent(y_true: pd.Series | np.ndarray, scores: np.ndarray, fraction: float) -> dict:
    """Compute capture and lift for the top score fraction."""
    y_array = np.asarray(y_true)
    n_top = max(1, int(np.ceil(len(y_array) * fraction)))
    order = np.argsort(scores)[::-1]
    top_y = y_array[order[:n_top]]
    base_rate = float(y_array.mean())
    top_rate = float(top_y.mean())
    return {
        "fraction": float(fraction),
        "buyers_captured": float(top_y.sum() / (y_array.sum() + 1e-9)),
        "lift": float(top_rate / (base_rate + 1e-9)),
        "top_conversion_rate": top_rate,
    }


def cross_validate_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, list[dict]]:
    """Evaluate candidate hyperparameter sets with stratified CV."""
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict] = []

    for candidate in MODEL_CANDIDATES:
        fold_metrics = []
        print(f"\nCV candidate: {candidate['name']}")

        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
            y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
            X_fold_val = X_train.iloc[val_idx].reset_index(drop=True)
            y_fold_val = y_train.iloc[val_idx].reset_index(drop=True)

            preprocessor, _, _ = build_preprocessor(X_fold_train)
            X_fold_train_enc = preprocessor.fit_transform(X_fold_train)
            X_fold_val_enc = preprocessor.transform(X_fold_val)

            scale_pos_weight = y_fold_train.value_counts()[0] / y_fold_train.value_counts()[1]
            model = make_model(scale_pos_weight=scale_pos_weight, params=candidate["params"])
            model.fit(X_fold_train_enc, y_fold_train)

            fold_scores = model.predict_proba(X_fold_val_enc)[:, 1]
            metrics = probability_metrics(y_fold_val, fold_scores)
            fold_metrics.append(metrics)
            print(
                "  "
                f"Fold {fold_index}: AP={metrics['average_precision']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"Brier={metrics['brier_score']:.4f}"
            )

        mean_metrics = {
            metric_name: float(np.mean([fold[metric_name] for fold in fold_metrics]))
            for metric_name in fold_metrics[0]
        }
        result = {
            "name": candidate["name"],
            "params": candidate["params"],
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
        }
        results.append(result)
        print(
            "  "
            f"Mean: AP={mean_metrics['average_precision']:.4f} | "
            f"ROC-AUC={mean_metrics['roc_auc']:.4f} | "
            f"Brier={mean_metrics['brier_score']:.4f}"
        )

    best_result = max(
        results,
        key=lambda item: (
            item["mean_metrics"]["average_precision"],
            item["mean_metrics"]["roc_auc"],
            -item["mean_metrics"]["brier_score"],
        ),
    )
    return best_result, results


def main() -> None:
    print("=" * 60)
    print("KNOWLEDGE GATE MODEL TRAINING")
    print("=" * 60)

    df_raw = pd.read_csv(DATA_PATH)
    validation = validate_dataset(
        df_raw,
        expected_columns=MODEL_SCHEMA_COLUMNS,
        require_target=True,
    )
    if validation.errors:
        raise ValueError(
            "Training data validation failed:\n"
            + format_validation_messages(validation.errors)
        )
    if validation.warnings:
        print("\nTraining data warnings:")
        print(format_validation_messages(validation.warnings))

    print(f"Loaded {len(df_raw):,} rows | {df_raw.shape[1]} columns")

    counts = df_raw[TARGET_COLUMN].value_counts()
    print("\nClass distribution:")
    print(f"  Non-buyers : {counts[0]:,} ({counts[0] / len(df_raw) * 100:.1f}%)")
    print(f"  Buyers     : {counts[1]:,} ({counts[1] / len(df_raw) * 100:.1f}%)")

    df = engineer_features(df_raw)
    dropped_training_columns = training_leak_columns(df.columns)
    print("\nApplied shared feature engineering.")
    print(f"Training-safe columns dropped: {dropped_training_columns or 'none'}")

    raw_feature_columns = [
        column for column in df_raw.columns if column not in {ID_COLUMN, TARGET_COLUMN}
    ]

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    train_df, calibration_df = train_test_split(
        train_val_df,
        test_size=CALIBRATION_SIZE,
        stratify=train_val_df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    print("\nData splits:")
    print(f"  Model-train rows : {len(train_df):,}")
    print(f"  Calibration rows : {len(calibration_df):,}")
    print(f"  Holdout test rows: {len(test_df):,}")

    X_train = build_model_frame(train_df, dropped_training_columns)
    y_train = train_df[TARGET_COLUMN].astype(int).reset_index(drop=True)
    X_calibration = build_model_frame(calibration_df, dropped_training_columns)
    y_calibration = calibration_df[TARGET_COLUMN].astype(int).reset_index(drop=True)
    X_test = build_model_frame(test_df, dropped_training_columns)
    y_test = test_df[TARGET_COLUMN].astype(int).reset_index(drop=True)

    best_candidate, cv_results = cross_validate_candidates(X_train, y_train)
    print(f"\nSelected candidate: {best_candidate['name']}")
    print(f"Parameters: {best_candidate['params']}")

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_calibration_enc = preprocessor.transform(X_calibration)
    X_test_enc = preprocessor.transform(X_test)

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    model = make_model(scale_pos_weight=scale_pos_weight, params=best_candidate["params"])

    print("\nTraining final base model...")
    model.fit(X_train_enc, y_train)

    train_raw_scores = model.predict_proba(X_train_enc)[:, 1]
    calibration_raw_scores = model.predict_proba(X_calibration_enc)[:, 1]
    test_raw_scores = model.predict_proba(X_test_enc)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_raw_scores, y_calibration)

    train_scores = np.clip(calibrator.transform(train_raw_scores), 0.0, 1.0)
    calibration_scores = np.clip(calibrator.transform(calibration_raw_scores), 0.0, 1.0)
    test_scores = np.clip(calibrator.transform(test_raw_scores), 0.0, 1.0)

    raw_test_metrics = probability_metrics(y_test, test_raw_scores)
    calibration_metrics = probability_metrics(y_calibration, calibration_scores)
    test_metrics = probability_metrics(y_test, test_scores)

    outreach_threshold = select_profit_threshold(y_calibration, calibration_scores)
    high_confidence_threshold = select_high_confidence_threshold(
        y_calibration,
        calibration_scores,
        min_precision=HIGH_PRECISION_TARGET,
    )
    if high_confidence_threshold["threshold"] < outreach_threshold["threshold"]:
        high_confidence_threshold = outreach_threshold.copy()

    test_outreach = summarize_threshold(y_test, test_scores, outreach_threshold["threshold"])
    test_high_confidence = summarize_threshold(
        y_test,
        test_scores,
        high_confidence_threshold["threshold"],
    )

    top10 = lift_at_percent(y_test, test_scores, 0.10)
    top20 = lift_at_percent(y_test, test_scores, 0.20)

    print("\nCalibration quality:")
    print(
        f"  Calibration AP     : {calibration_metrics['average_precision']:.4f}\n"
        f"  Calibration Brier  : {calibration_metrics['brier_score']:.4f}"
    )

    print("\nHoldout evaluation:")
    print(
        f"  Raw test ROC-AUC   : {raw_test_metrics['roc_auc']:.4f}\n"
        f"  Raw test AP        : {raw_test_metrics['average_precision']:.4f}\n"
        f"  Raw test Brier     : {raw_test_metrics['brier_score']:.4f}"
    )
    print(
        f"  Cal test ROC-AUC   : {test_metrics['roc_auc']:.4f}\n"
        f"  Cal test AP        : {test_metrics['average_precision']:.4f}\n"
        f"  Cal test Brier     : {test_metrics['brier_score']:.4f}"
    )

    print("\nThreshold policy:")
    print(
        f"  Outreach threshold : {outreach_threshold['threshold']:.3f} | "
        f"Precision={outreach_threshold['precision'] * 100:.1f}% | "
        f"Recall={outreach_threshold['recall'] * 100:.1f}% | "
        f"Coverage={outreach_threshold['coverage'] * 100:.1f}%"
    )
    print(
        f"  High-confidence    : {high_confidence_threshold['threshold']:.3f} | "
        f"Precision={high_confidence_threshold['precision'] * 100:.1f}% | "
        f"Recall={high_confidence_threshold['recall'] * 100:.1f}% | "
        f"Coverage={high_confidence_threshold['coverage'] * 100:.1f}%"
    )
    print(
        f"  Holdout outreach   : Rs {test_outreach['profit']:,.0f} | "
        f"Precision={test_outreach['precision'] * 100:.1f}% | "
        f"Recall={test_outreach['recall'] * 100:.1f}%"
    )
    print(
        f"  Holdout high-conf  : Precision={test_high_confidence['precision'] * 100:.1f}% | "
        f"Recall={test_high_confidence['recall'] * 100:.1f}%"
    )

    print("\nRanking quality:")
    print(
        f"  Top 10% lift       : {top10['lift']:.2f}x | "
        f"Buyer capture={top10['buyers_captured'] * 100:.1f}%"
    )
    print(
        f"  Top 20% lift       : {top20['lift']:.2f}x | "
        f"Buyer capture={top20['buyers_captured'] * 100:.1f}%"
    )

    feature_names = list(preprocessor.get_feature_names_out())
    feature_importance = pd.Series(model.feature_importances_, index=feature_names)
    top_features = feature_importance.nlargest(15).round(6).to_dict()

    config = {
        "model_version": "kg_xgb_calibrated_v2",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "calibration_size": CALIBRATION_SIZE,
        "cv_folds": CV_FOLDS,
        "calibration_method": "isotonic_regression",
        "high_precision_target": HIGH_PRECISION_TARGET,
        "high_threshold": float(high_confidence_threshold["threshold"]),
        "outreach_threshold": float(outreach_threshold["threshold"]),
        "hot_threshold": float(high_confidence_threshold["threshold"]),
        "warm_threshold": float(outreach_threshold["threshold"]),
        "high_threshold_summary": high_confidence_threshold,
        "outreach_threshold_summary": outreach_threshold,
        "test_high_threshold_summary": test_high_confidence,
        "test_outreach_threshold_summary": test_outreach,
        "expected_value_threshold": float(CALL_COST_INR / COURSE_PRICE_INR),
        "course_price_inr": COURSE_PRICE_INR,
        "call_cost_inr": CALL_COST_INR,
        "train_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rate": float(train_df[TARGET_COLUMN].mean()),
        "calibration_positive_rate": float(calibration_df[TARGET_COLUMN].mean()),
        "test_positive_rate": float(test_df[TARGET_COLUMN].mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "raw_feature_columns": raw_feature_columns,
        "model_input_columns": list(X_train.columns),
        "dropped_training_columns": dropped_training_columns,
        "feature_names": feature_names,
        "top_features": top_features,
        "candidate_results": cv_results,
        "selected_candidate": {
            "name": best_candidate["name"],
            "params": best_candidate["params"],
            "mean_metrics": best_candidate["mean_metrics"],
        },
        "raw_test_metrics": raw_test_metrics,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "average_precision": float(test_metrics["average_precision"]),
        "pr_auc": float(test_metrics["pr_auc"]),
        "brier_score": float(test_metrics["brier_score"]),
        "validation_profit_inr": int(outreach_threshold["profit"]),
        "test_profit_inr": int(test_outreach["profit"]),
        "max_profit_inr": int(test_outreach["profit"]),
        "top10_lift": float(top10["lift"]),
        "top10_capture": float(top10["buyers_captured"]),
        "top20_lift": float(top20["lift"]),
        "top20_capture": float(top20["buyers_captured"]),
        "test_student_ids": test_df[ID_COLUMN].astype(str).tolist(),
        "calibration_student_ids": calibration_df[ID_COLUMN].astype(str).tolist(),
    }

    print("\nSaving model artifacts...")
    joblib.dump(model, os.path.join(MODELS_DIR, "kg_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "kg_preprocessor.pkl"))
    joblib.dump(calibrator, os.path.join(MODELS_DIR, "kg_calibrator.pkl"))
    joblib.dump(config, os.path.join(MODELS_DIR, "kg_config.pkl"))

    print("Saved kg_model.pkl, kg_preprocessor.pkl, kg_calibrator.pkl, kg_config.pkl")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Holdout ROC-AUC     : {test_metrics['roc_auc']:.4f}")
    print(f"  Holdout AP          : {test_metrics['average_precision']:.4f}")
    print(f"  Holdout Brier       : {test_metrics['brier_score']:.4f}")
    print(f"  High Threshold      : {high_confidence_threshold['threshold']:.3f}")
    print(f"  Outreach Threshold  : {outreach_threshold['threshold']:.3f}")
    print(f"  Holdout Profit      : Rs {test_outreach['profit']:,.0f}")
    print("=" * 60)
    print("\nRun: streamlit run app.py")


if __name__ == "__main__":
    main()
