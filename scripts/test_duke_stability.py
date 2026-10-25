"""Check paired summaries and shared patient splits across repetitions."""

import unittest

import numpy as np
import pandas as pd

from check_duke_stability import METRICS, run_repeats, summarize
from prepare_duke_baseline import FEATURES, SEED, assign_folds
from run_duke_baseline import evaluate_scenario


class StabilityTests(unittest.TestCase):
    def test_summary_pairs_available_metrics_without_zero_imputation(self):
        repeats = []
        for a, b in ((0.4, 0.7), (None, 0.8), (0.6, 0.5)):
            scenario = {"models": {model: {"aggregate": dict.fromkeys(METRICS, value)}
                                   for model, value in (("age", a), ("age_volume", b))}}
            repeats.append({"scenarios": {name: scenario for name in ("primary", "exclude_death_gap")}})
        result = summarize(repeats)["primary"][METRICS[0]]
        difference = result["paired_difference_age_volume_minus_age"]
        self.assertEqual(difference["available"], 2)
        self.assertEqual(difference["unavailable"], 1)
        self.assertEqual(difference["positive_repeats"], 1)
        self.assertAlmostEqual(difference["mean"], 0.1)

    def test_repeated_fits_reproduce_baseline_and_keep_shared_folds(self):
        rng = np.random.default_rng(41)
        index = [f"P{i:03}" for i in range(75)]
        features = pd.DataFrame({FEATURES[0]: rng.uniform(25, 80, 75),
                                 FEATURES[1]: rng.uniform(1, 12, 75)}, index=index)
        features.loc["P003", FEATURES[1]] = np.nan
        outcomes = pd.DataFrame({"time_days": rng.uniform(10, 2000, 75),
            "event_type": [int(i % 3 == 0) for i in range(75)],
            "death_gap": [int(i == 1) for i in range(75)], "mri_before_diagnosis": 0}, index=index)
        repeats, assignments, scores = run_repeats(features, outcomes, seeds=(SEED, SEED + 1))
        mapping = assign_folds([{"patient_id": p, "event_type": outcomes.loc[p, "event_type"]}
                                for p in index])
        baseline, _ = evaluate_scenario(features, outcomes, pd.Series(mapping).loc[index],
                                       np.ones(75, dtype=bool), "primary")
        self.assertEqual(repeats[0]["scenarios"]["primary"], baseline)
        lookup = {(r["seed"], r["patient_id"]): r["fold"] for r in assignments}
        self.assertEqual(len(lookup), 150)
        self.assertTrue(any(lookup[(SEED, p)] != lookup[(SEED + 1, p)] for p in index))
        keys = {(r["seed"], r["scenario"], r["model"], r["patient_id"]) for r in scores}
        self.assertEqual(len(keys), 2 * 2 * (75 + 74))
        self.assertEqual(len(keys), len(scores))
        for row in scores:
            self.assertEqual(row["fold"], lookup[(row["seed"], row["patient_id"])])
            if row["scenario"] == "exclude_death_gap":
                self.assertNotEqual(row["patient_id"], "P001")
        for repeat in repeats:
            for name, scenario in repeat["scenarios"].items():
                for fold in scenario["models"]["age_volume"]["folds"]:
                    train_ids = [p for p in index if lookup[(repeat["seed"], p)] != fold["fold"]
                                 and (name == "primary" or p != "P001")]
                    self.assertAlmostEqual(fold["imputer_train_medians"][FEATURES[1]],
                                           features.loc[train_ids, FEATURES[1]].median())
                    self.assertAlmostEqual(fold["scaler_train_means"][FEATURES[0]],
                                           features.loc[train_ids, FEATURES[0]].mean())


if __name__ == "__main__":
    unittest.main()
