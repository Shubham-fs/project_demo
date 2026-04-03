import unittest

import pandas as pd

from data_validation import MODEL_SCHEMA_COLUMNS, validate_dataset


STRING_COLUMNS = {
    "student_id": "S001",
    "acquisition_channel": "organic",
    "course_interest": "GATE CSE",
    "student_stage": "student",
    "most_active_time_window": "evening",
    "ai_mentor_topic": "none",
    "gate_exam_urgency": "medium",
    "call_duration_bucket": "short",
}


def make_valid_row(student_id: str = "S001", target: int = 0) -> dict:
    row = {column: 0 for column in MODEL_SCHEMA_COLUMNS}
    row.update(STRING_COLUMNS)
    row["student_id"] = student_id
    row["purchased_paid_course"] = target
    row["registration_days_ago"] = 30
    row["login_frequency_14d"] = 7
    row["total_platform_minutes"] = 180
    row["session_depth"] = 6
    row["last_login_days_ago"] = 1
    row["current_streak_days"] = 5
    row["active_days_30d"] = 12
    row["pyq_accuracy_pct"] = 60
    row["free_mock_test_avg_score"] = 55
    row["profile_completion_pct"] = 80
    row["platform_rating"] = 4
    row["course_urgency_score"] = 0.6
    row["urgency_weight"] = 6
    row["days_to_next_exam"] = 90
    return row


class DataValidationTests(unittest.TestCase):
    def test_valid_dataset_passes(self):
        df = pd.DataFrame([make_valid_row()])
        result = validate_dataset(
            df,
            expected_columns=MODEL_SCHEMA_COLUMNS,
            require_target=True,
        )
        self.assertEqual(result.errors, [])

    def test_duplicate_student_id_is_rejected(self):
        df = pd.DataFrame([make_valid_row("S001"), make_valid_row("S001", 1)])
        result = validate_dataset(
            df,
            expected_columns=MODEL_SCHEMA_COLUMNS,
            require_target=True,
        )
        self.assertTrue(any("Duplicate student_id" in message for message in result.errors))

    def test_invalid_binary_and_range_values_are_rejected(self):
        row = make_valid_row()
        row["coupon_code_applied"] = 2
        row["profile_completion_pct"] = 120
        df = pd.DataFrame([row])
        result = validate_dataset(
            df,
            expected_columns=MODEL_SCHEMA_COLUMNS,
            require_target=True,
        )
        self.assertTrue(any("coupon_code_applied" in message for message in result.errors))
        self.assertTrue(any("profile_completion_pct" in message for message in result.errors))


if __name__ == "__main__":
    unittest.main()
