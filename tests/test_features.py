import unittest

import pandas as pd

from kg_features import engineer_features, training_leak_columns


class FeatureEngineeringTests(unittest.TestCase):
    def test_engineer_features_handles_missing_values_and_adds_columns(self):
        df = pd.DataFrame(
            [
                {
                    "pyq_accuracy_pct": None,
                    "free_mock_test_avg_score": None,
                    "ai_mentor_topic": None,
                    "ai_mentor_total_messages": 10,
                    "ai_mentor_last_used_days_ago": 1,
                    "registration_days_ago": 9,
                    "login_frequency_14d": 7,
                    "counsellor_call_picked_up": 1,
                    "counsellor_call_duration_mins": 4,
                    "cart_items_count": 2,
                    "coupon_code_applied": 1,
                    "community_posts_count": 4,
                    "community_comments_count": 6,
                    "community_saved_posts": 10,
                    "free_mock_test_count": 3,
                    "pricing_page_visits": 5,
                    "urgency_weight": 8,
                    "total_platform_minutes": 240,
                    "active_days_30d": 12,
                    "session_depth": 14,
                }
            ]
        )

        engineered = engineer_features(df)

        self.assertIn("ai_mentor_recency_weight", engineered.columns)
        self.assertIn("mock_commitment", engineered.columns)
        self.assertIn("minutes_per_active_day", engineered.columns)
        self.assertEqual(engineered.loc[0, "ai_mentor_topic"], "none")
        self.assertAlmostEqual(engineered.loc[0, "ai_mentor_recency_weight"], 5.0)
        self.assertEqual(engineered.loc[0, "effective_call_flag"], 1)
        self.assertAlmostEqual(engineered.loc[0, "cart_seriousness"], 4.0)
        self.assertAlmostEqual(engineered.loc[0, "mock_commitment"], 0.03)

    def test_training_leak_columns_only_returns_present_known_columns(self):
        columns = [
            "student_id",
            "standard_certificates",
            "paid_course_completed",
            "some_other_feature",
        ]
        leaks = training_leak_columns(columns)
        self.assertEqual(leaks, ["standard_certificates", "paid_course_completed"])


if __name__ == "__main__":
    unittest.main()
