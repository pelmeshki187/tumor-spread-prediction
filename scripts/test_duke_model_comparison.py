"""Verify shared held-out patients, training-only preprocessing and paired comparisons."""

import unittest

import numpy as np
import pandas as pd

from compare_duke_models import MODELS, build_model, compare, summarize_comparison
from check_duke_stability import METRICS
from prepare_duke_baseline import FEATURES, SEED, assign_folds
from run_duke_baseline import evaluate_scenario


class ComparisonTests(unittest.TestCase):
    def test_all_models_share_folds_and_reproduce_original_cox(self):
        rng = np.random.default_rng(53)
        index = [f"P{i:03}" for i in range(75)]
        features = pd.DataFrame({FEATURES[0]: rng.uniform(25, 80, 75),
                                 FEATURES[1]: rng.uniform(1, 12, 75)}, index=index)
        features.loc["P003", FEATURES[1]] = np.nan
        outcomes = pd.DataFrame({"time_days": rng.uniform(10, 2000, 75),
            "event_type": [int(i % 3 == 0) for i in range(75)],
            "death_gap": [int(i == 1) for i in range(75)], "mri_before_diagnosis": 0}, index=index)
        repeats, assignments, scores = compare(features, outcomes, seeds=(SEED,))
        lookup = {r["patient_id"]: r["fold"] for r in assignments}
        folds = pd.Series(lookup).loc[index]
        baseline, _ = evaluate_scenario(features, outcomes, folds, np.ones(75, dtype=bool), "primary")
        for model in ("age", "age_volume"):
            self.assertEqual(repeats[0]["scenarios"]["primary"]["models"][model], baseline["models"][model])
        self.assertEqual(len(scores), len(MODELS) * (75 + 74))
        self.assertEqual(len({(r["scenario"], r["model"], r["patient_id"]) for r in scores}), len(scores))
        for row in scores:
            self.assertEqual(row["fold"], lookup[row["patient_id"]])
            self.assertTrue(np.isfinite(row["risk_score"]))
            if row["scenario"] == "exclude_death_gap":
                self.assertNotEqual(row["patient_id"], "P001")
        for name, scenario in repeats[0]["scenarios"].items():
            for model, details in scenario["models"].items():
                self.assertEqual(details["features"], MODELS[model])
                for fold in details["folds"]:
                    train_ids = [p for p in index if lookup[p] != fold["fold"]
                                 and (name == "primary" or p != "P001")]
                    for column in MODELS[model]:
                        values = features.loc[train_ids, column]
                        self.assertAlmostEqual(fold["imputer_train_medians"][column], values.median())
                        self.assertAlmostEqual(fold["scaler_train_means"][column], values.fillna(values.median()).mean())

    def test_tree_predictions_are_reproducible_and_have_correct_risk_direction(self):
        # One feature perfectly separates early and late events. Higher values mean earlier events.
        x = pd.DataFrame({"volume": np.repeat([0., 1., 2., 3.], 40)})
        y = np.array([(True, float(t)) for t in np.repeat([400, 300, 200, 100], 40)],
                     dtype=[("event", bool), ("time", float)])
        for name in ("forest", "boosting"):
            a, b = build_model(name), build_model(name)
            a.fit(x, y)
            b.fit(x, y)
            prediction = a.predict(pd.DataFrame({"volume": [0., 3.]}))
            self.assertGreater(prediction[1], prediction[0])
            np.testing.assert_array_equal(a.predict(x), b.predict(x))

    def test_paired_summary_skips_unavailable_reference_without_imputation(self):
        repeats = []
        for reference, forest in ((0.6, 0.7), (None, 0.9), (0.8, 0.7)):
            models = {m: {"aggregate": dict.fromkeys(METRICS, forest if m == "forest" else reference)}
                      for m in MODELS}
            repeats.append({"scenarios": {s: {"models": models} for s in ("primary", "exclude_death_gap")}})
        stats = summarize_comparison(repeats)["primary"][METRICS[0]]["forest"]
        self.assertEqual(stats["scores"]["available"], 3)
        delta = stats["paired_difference_vs_age_volume"]
        self.assertEqual(delta["available"], 2)
        self.assertEqual(delta["unavailable"], 1)
        self.assertEqual(delta["positive_repeats"], 1)
        self.assertAlmostEqual(delta["mean"], 0.)


if __name__ == "__main__":
    unittest.main()
