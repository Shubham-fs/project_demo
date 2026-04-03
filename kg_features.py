"""Shared feature engineering for the Knowledge Gate conversion model."""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "student_id"
TARGET_COLUMN = "purchased_paid_course"

# Features that are only available after a student has already purchased.
_LEAKY_COLUMNS = [
    "standard_certificates",
    "completion_certificate",
    "paid_course_completed",
    "paid_content_hours_watched",
]

_DERIVED_INPUT_DEFAULTS = {
    "pyq_accuracy_pct": 0,
    "free_mock_test_avg_score": 0,
    "ai_mentor_topic": "none",
    "ai_mentor_total_messages": 0,
    "ai_mentor_last_used_days_ago": 999,
    "registration_days_ago": 0,
    "login_frequency_14d": 0,
    "counsellor_call_picked_up": 0,
    "counsellor_call_duration_mins": 0,
    "cart_items_count": 0,
    "coupon_code_applied": 0,
    "community_posts_count": 0,
    "community_comments_count": 0,
    "community_saved_posts": 0,
    "free_mock_test_count": 0,
    "pricing_page_visits": 0,
    "urgency_weight": 0,
    "total_platform_minutes": 0,
    "active_days_30d": 0,
    "session_depth": 0,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived, scoring-safe features added."""
    df = df.copy()

    for column, default_value in _DERIVED_INPUT_DEFAULTS.items():
        if column in df.columns:
            if isinstance(default_value, str):
                df[column] = df[column].where(df[column].notna(), default_value)
            else:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default_value)

    df["ai_mentor_recency_weight"] = (
        df["ai_mentor_total_messages"] / (df["ai_mentor_last_used_days_ago"] + 1)
    )
    df["ai_mentor_msgs_per_day"] = (
        df["ai_mentor_total_messages"] / (df["registration_days_ago"] + 1)
    )
    df["daily_login_rate"] = df["login_frequency_14d"] / 14
    df["effective_call_flag"] = (
        (df["counsellor_call_picked_up"] == 1)
        & (df["counsellor_call_duration_mins"] > 2)
    ).astype(int)
    df["cart_seriousness"] = df["cart_items_count"] * (1 + df["coupon_code_applied"])
    df["community_depth"] = (
        df["community_posts_count"]
        + df["community_comments_count"] * 0.5
        + df["community_saved_posts"] * 0.3
    )
    df["mock_commitment"] = (
        df["free_mock_test_count"] * (df["free_mock_test_avg_score"] / 100 + 0.01)
    )
    df["urgent_pricing_interest"] = df["pricing_page_visits"] * df["urgency_weight"]
    df["minutes_per_active_day"] = (
        df["total_platform_minutes"] / (df["active_days_30d"] + 1)
    )
    df["depth_per_session_proxy"] = (
        df["session_depth"] / (df["login_frequency_14d"] + 1)
    )

    return df


def training_leak_columns(columns) -> list[str]:
    """Return the subset of known leaky columns present in ``columns``."""
    return [column for column in _LEAKY_COLUMNS if column in columns]


def missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return required columns that are missing from ``df``."""
    return [column for column in required if column not in df.columns]
