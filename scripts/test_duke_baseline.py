"""Check patient separation, leakage barriers and censored metric behavior."""

import unittest

import numpy as np
import pandas as pd
from sksurv.util import Surv

from prepare_duke_baseline import FEATURES, assign_folds, build_tables, volume_value
from run_duke_baseline import evaluate_scenario, ipcw_metrics, validate_feature_schema


class PreparationTests(unittest.TestCase):
    def test_missing_volume_is_preserved_and_invalid_volume_rejected(self):
        for value in (None, "NaN", "NA", float("nan")):
            self.assertIsNone(volume_value(value))
        for value in (0, -1, float("inf"), "unexpected"):
            with self.assertRaises(ValueError):
                volume_value(value)

    def test_folds_cover_unique_patients_and_ignore_input_order(self):
        rows = [{"patient_id": f"P{i:03}", "event_type": int(i % 3 == 0)} for i in range(60)]
        mapping = assign_folds(rows)
        self.assertEqual(mapping, assign_folds(list(reversed(rows))))
        self.assertEqual(set(mapping), {r["patient_id"] for r in rows})
        for fold in range(5):
            train = {p for p, f in mapping.items() if f != fold}
            test = {p for p, f in mapping.items() if f == fold}
            self.assertFalse(train & test)
            self.assertEqual(len(train | test), len(rows))
            self.assertEqual(sum(r["event_type"] for r in rows if r["patient_id"] in test), 4)

    def test_outcomes_and_future_fields_do_not_enter_features(self):
        record = {"patient_id": "Breast_MRI_001", "mri": 10, "birth": -20000,
            "presentation": 0, "stage_m": 0, "recurrence": 0, "local": "NP",
            "distant": "NP", "death": "NP", "last_dr_free": 1200, "surgery": 50}
        _, x1, _ = build_tables([record], {record["patient_id"]: 100})
        changed = record | {"distant": 700, "recurrence": 1, "last_dr_free": 600,
                            "future_treatment": "secret outcome-correlated field"}
        _, x2, _ = build_tables([changed], {record["patient_id"]: 100})
        self.assertEqual(x1, x2)
        self.assertEqual(set(x1[0]), {"patient_id", *FEATURES})

    def test_runner_refuses_outcome_column_in_feature_table(self):
        frame = pd.DataFrame({"patient_id": ["P1"], FEATURES[0]: [50], FEATURES[1]: [2], "time_days": [100]})
        with self.assertRaises(ValueError):
            validate_feature_schema(frame)


class ValidationTests(unittest.TestCase):
    def test_ipcw_retains_late_followed_patients_as_comparators(self):
        train = Surv.from_arrays([True, False, True, False, False], [1, 4, 20, 200, 300])
        test = Surv.from_arrays([True, False, True], [10, 150, 300])
        result = ipcw_metrics(train, test, np.array([2, 1, 0]), horizon=100)
        self.assertEqual(result["cindex"], 1.0)
        self.assertGreater(result["g_train_at_horizon"], 0)

    def test_unsupported_ipcw_horizon_is_explicitly_unavailable(self):
        train = Surv.from_arrays([True, False], [1, 20])
        result = ipcw_metrics(train, train, np.array([1, 0]), horizon=100)
        self.assertIsNone(result["cindex"])
        self.assertIsNotNone(result["unavailable_reason"])

    def test_full_cv_fits_preprocessing_on_training_patients_only(self):
        rng = np.random.default_rng(23)
        index = [f"P{i:03}" for i in range(75)]
        features = pd.DataFrame({FEATURES[0]: rng.uniform(25, 80, 75),
                                 FEATURES[1]: rng.uniform(1, 12, 75)}, index=index)
        features.loc["P003", FEATURES[1]] = np.nan
        outcomes = pd.DataFrame({"time_days": rng.uniform(10, 2000, 75),
            "event_type": [int(i % 3 == 0) for i in range(75)], "death_gap": 0,
            "mri_before_diagnosis": 0}, index=index)
        rows = [{"patient_id": p, "event_type": outcomes.loc[p, "event_type"]} for p in index]
        mapping = assign_folds(rows)
        folds = pd.Series([mapping[p] for p in index], index=index)
        result, scores = evaluate_scenario(features, outcomes, folds, np.ones(75, dtype=bool), "test")
        self.assertEqual(len(scores), 150)
        self.assertEqual(len({(r["patient_id"], r["model"]) for r in scores}), 150)
        for fold in result["models"]["age_volume"]["folds"]:
            training = features.loc[folds != fold["fold"]]
            self.assertAlmostEqual(fold["scaler_train_means"][FEATURES[0]], training[FEATURES[0]].mean())
            self.assertAlmostEqual(fold["imputer_train_medians"][FEATURES[1]], training[FEATURES[1]].median())


if __name__ == "__main__":
    unittest.main()
