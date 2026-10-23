"""Regression checks for event timing, unknown outcomes and competing death."""

import unittest

from audit_duke import BLOCKING_FLAGS, endpoint, horizon_counts


class EndpointTests(unittest.TestCase):
    def record(self, **changes):
        return {"mri": 20, "distant": "NP", "death": "NP", "last_dr_free": 1000,
                "surgery": 40, "recurrence": 0, "local": "NP"} | changes

    def test_event_clock_and_negative_mri_date(self):
        result = endpoint(self.record(mri=-10, distant=90, recurrence=1))
        self.assertEqual((result["time_from_mri_days"], result["event_type"]), (100, 1))
        self.assertFalse(set(result["flags"]) & BLOCKING_FLAGS)

    def test_missing_followup_is_not_negative_label(self):
        result = endpoint(self.record(last_dr_free="NC", death=800))
        self.assertIsNone(result["time_from_mri_days"])
        self.assertIn("missing_disease_specific_followup", result["flags"])

    def test_conflicting_flag_does_not_erase_recurrence(self):
        result = endpoint(self.record(distant=120))
        self.assertEqual(result["event_type"], 1)
        self.assertTrue(set(result["flags"]) & BLOCKING_FLAGS)

    def test_death_gap_does_not_extend_disease_free_observation(self):
        result = endpoint(self.record(last_dr_free=100, death=200))
        self.assertEqual((result["time_from_mri_days"], result["event_type"]), (80, 0))
        self.assertIn("death_after_last_dr_free_assessment", result["flags"])

    def test_observed_death_is_competing_not_recurrence(self):
        result = endpoint(self.record(last_dr_free=200, death=200))
        self.assertEqual((result["time_from_mri_days"], result["event_type"]), (180, 2))

    def test_death_before_recurrence_is_quarantined(self):
        result = endpoint(self.record(distant=200, death=100, recurrence=1))
        self.assertIn("death_before_distant_recurrence", result["flags"])

    def test_observation_at_mri_is_not_positive_followup(self):
        result = endpoint(self.record(last_dr_free=20))
        self.assertIn("no_positive_followup_after_mri", result["flags"])

    def test_horizon_does_not_turn_early_censoring_into_negative(self):
        rows = [{"time_from_mri_days": t, "event_type": e}
                for t, e in [(100, 0), (100, 1), (100, 2), (800, 0), (800, 1)]]
        self.assertEqual(horizon_counts(rows, 1), {
            "distant_recurrence": 1, "competing_death": 1,
            "observed_event_free": 2, "early_censored": 1})


if __name__ == "__main__":
    unittest.main()
