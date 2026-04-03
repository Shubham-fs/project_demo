"""Dataset validation helpers for training and scoring workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kg_features import ID_COLUMN, TARGET_COLUMN

MODEL_SCHEMA_COLUMNS = [
    "student_id",
    "acquisition_channel",
    "course_interest",
    "student_stage",
    "registration_days_ago",
    "targets_count",
    "target_gate",
    "target_placement",
    "target_govt_jobs",
    "login_frequency_14d",
    "total_platform_minutes",
    "session_depth",
    "last_login_days_ago",
    "most_active_time_window",
    "current_streak_days",
    "active_days_30d",
    "beta_mode_switched",
    "free_platform_lectures_watched",
    "pyq_attempted_count",
    "pyq_accuracy_pct",
    "free_mock_test_count",
    "free_mock_test_avg_score",
    "onboarding_video_watched",
    "ai_mentor_sessions",
    "ai_mentor_total_messages",
    "ai_mentor_last_used_days_ago",
    "ai_mentor_daily_limit_hit",
    "ai_mentor_limit_hit_days",
    "ai_mentor_image_uploaded",
    "ai_mentor_topic",
    "rank_predictor_used",
    "rank_predictor_coupon_copied",
    "rank_predictor_course_clicked",
    "gate_exam_urgency",
    "communities_joined_count",
    "community_posts_count",
    "community_comments_count",
    "community_saved_posts",
    "community_image_posted",
    "chat_initiated_count",
    "study_group_joined",
    "study_group_created",
    "followers_count",
    "following_count",
    "pricing_page_visits",
    "demo_class_attended",
    "cart_items_count",
    "wishlist_count",
    "coupon_code_applied",
    "profile_completion_pct",
    "has_profile_picture",
    "has_education_filled",
    "has_career_goals",
    "resume_uploaded",
    "standard_certificates",
    "job_notifications_enabled",
    "job_alerts_clicked_count",
    "job_applications_count",
    "resume_review_requested",
    "resume_review_completed",
    "mock_interview_requested",
    "mock_interview_completed",
    "question_reports_count",
    "feedback_submitted",
    "platform_rating",
    "referral_program_visited",
    "successful_referrals",
    "active_month",
    "course_urgency_score",
    "days_to_next_exam",
    "post_result_window",
    "is_peak_placement_season",
    "whatsapp_outreach_received",
    "whatsapp_replied",
    "counsellor_call_received",
    "counsellor_call_picked_up",
    "counsellor_call_duration_mins",
    "call_duration_bucket",
    "purchased_paid_course",
    "daily_avg_minutes",
    "engagement_intensity",
    "intent_score",
    "is_professional",
    "pro_intent_interaction",
    "effective_call_minutes",
    "urgency_weight",
]

BINARY_COLUMNS = [
    "target_gate",
    "target_placement",
    "target_govt_jobs",
    "beta_mode_switched",
    "onboarding_video_watched",
    "ai_mentor_daily_limit_hit",
    "ai_mentor_image_uploaded",
    "rank_predictor_used",
    "rank_predictor_coupon_copied",
    "rank_predictor_course_clicked",
    "community_image_posted",
    "study_group_joined",
    "study_group_created",
    "demo_class_attended",
    "coupon_code_applied",
    "has_profile_picture",
    "has_education_filled",
    "has_career_goals",
    "resume_uploaded",
    "job_notifications_enabled",
    "resume_review_requested",
    "resume_review_completed",
    "mock_interview_requested",
    "mock_interview_completed",
    "feedback_submitted",
    "referral_program_visited",
    "is_peak_placement_season",
    "whatsapp_outreach_received",
    "whatsapp_replied",
    "counsellor_call_received",
    "counsellor_call_picked_up",
    "purchased_paid_course",
]

NON_NEGATIVE_COLUMNS = [
    "registration_days_ago",
    "targets_count",
    "login_frequency_14d",
    "total_platform_minutes",
    "session_depth",
    "last_login_days_ago",
    "current_streak_days",
    "active_days_30d",
    "free_platform_lectures_watched",
    "pyq_attempted_count",
    "free_mock_test_count",
    "ai_mentor_sessions",
    "ai_mentor_total_messages",
    "ai_mentor_last_used_days_ago",
    "ai_mentor_limit_hit_days",
    "communities_joined_count",
    "community_posts_count",
    "community_comments_count",
    "community_saved_posts",
    "chat_initiated_count",
    "followers_count",
    "following_count",
    "pricing_page_visits",
    "cart_items_count",
    "wishlist_count",
    "standard_certificates",
    "job_alerts_clicked_count",
    "job_applications_count",
    "question_reports_count",
    "successful_referrals",
    "days_to_next_exam",
    "post_result_window",
    "counsellor_call_duration_mins",
    "daily_avg_minutes",
    "engagement_intensity",
    "effective_call_minutes",
]

RANGE_RULES = {
    "pyq_accuracy_pct": (0, 100),
    "free_mock_test_avg_score": (0, 100),
    "profile_completion_pct": (0, 100),
    "platform_rating": (0, 5),
    "course_urgency_score": (0, 1),
    "urgency_weight": (0, 10),
}


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataset(
    df: pd.DataFrame,
    *,
    expected_columns: list[str],
    require_target: bool,
) -> ValidationResult:
    """Validate schema and a few high-signal quality constraints."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns[:15]))

    if ID_COLUMN not in df.columns:
        return ValidationResult(errors=errors, warnings=warnings)

    if df.empty:
        errors.append("Dataset is empty.")
        return ValidationResult(errors=errors, warnings=warnings)

    student_ids = df[ID_COLUMN].astype(str)
    if student_ids.nunique(dropna=False) != len(df):
        errors.append("Duplicate student_id values detected.")
    if student_ids.isna().any() or (student_ids.str.strip() == "").any():
        errors.append("Blank or missing student_id values detected.")

    if require_target:
        if TARGET_COLUMN not in df.columns:
            errors.append(f"Missing target column: {TARGET_COLUMN}.")
        else:
            invalid_target = set(df[TARGET_COLUMN].dropna().unique()) - {0, 1}
            if invalid_target:
                errors.append("Target column must contain only 0/1 values.")

    for column in BINARY_COLUMNS:
        if column == TARGET_COLUMN and not require_target:
            continue
        if column in df.columns:
            invalid_values = set(df[column].dropna().unique()) - {0, 1}
            if invalid_values:
                errors.append(f"Column '{column}' must contain only 0/1 values.")

    for column in NON_NEGATIVE_COLUMNS:
        if column in df.columns and (df[column].dropna() < 0).any():
            errors.append(f"Column '{column}' contains negative values.")

    for column, (min_value, max_value) in RANGE_RULES.items():
        if column not in df.columns:
            continue
        series = df[column].dropna()
        if ((series < min_value) | (series > max_value)).any():
            errors.append(
                f"Column '{column}' must stay between {min_value} and {max_value}."
            )

    if "course_interest" in df.columns and df["course_interest"].nunique(dropna=True) < 2:
        warnings.append("course_interest has fewer than 2 unique values in this file.")

    if require_target and TARGET_COLUMN in df.columns:
        target_rate = float(df[TARGET_COLUMN].mean())
        if target_rate < 0.02 or target_rate > 0.80:
            warnings.append(
                f"Target positive rate looks unusual at {target_rate * 100:.1f}%."
            )

    return ValidationResult(errors=errors, warnings=warnings)


def format_validation_messages(messages: list[str]) -> str:
    """Format validation messages for CLI or UI display."""
    return "\n".join(f"- {message}" for message in messages)
