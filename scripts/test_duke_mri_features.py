"""Check source integrity and isolation of nested MRI preprocessing and tuning."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from openpyxl import Workbook
from sksurv.util import Surv

from prepare_duke_baseline import FEATURES, SEED, assign_folds, sha256
from run_duke_baseline import evaluate_scenario
from run_duke_mri_features import (EXTRA_FEATURES, MODEL_COLUMNS, PANEL, VOLUME, evaluate_outer,
    inner_assignment, numeric_feature, read_panel, summarize, tune_ridge, validate_panel)
from check_duke_stability import METRICS


class SourceTests(unittest.TestCase):
    def test_missing_values_preserved_and_unknown_values_rejected(self):
        for v in (None, "NaN", "NA", "NC", "NP", float("nan")):
            self.assertTrue(math.isnan(numeric_feature(v)))
        self.assertEqual(numeric_feature(-2.5), -2.5)
        self.assertEqual(numeric_feature(0), 0.)
        for v in (True, float("inf"), "unexpected", "12"):
            with self.assertRaises(ValueError):
                numeric_feature(v)

    def test_source_hash_and_fixed_columns_protect_feature_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Imaging_Features.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Patient ID", "Volume_cu_mm_Tumor", *EXTRA_FEATURES, "future_outcome"])
            sheet.append(["P1", 10, *range(1, 9), 999])
            sheet.append(["P2", "NaN", *range(2, 10), 123])
            workbook.save(path)
            baseline = pd.DataFrame({FEATURES[0]: [40, 50], VOLUME: [math.log1p(10), np.nan]},
                                    index=pd.Index(["P1", "P2"], name="patient_id"))
            manifest = {"source_sha256": {"Imaging_Features.xlsx": sha256(path)}}
            frame = read_panel(path, baseline, manifest)
            self.assertEqual(tuple(frame.columns), PANEL)
            self.assertNotIn("future_outcome", frame)
            self.assertTrue(np.isnan(frame.loc["P2", VOLUME]))
            with self.assertRaises(ValueError):
                validate_panel(frame.assign(future_outcome=0))
            with self.assertRaises(ValueError):
                read_panel(path, baseline.assign(**{VOLUME: 42}), manifest)
            sheet.append(["P1", 10, *range(1, 9), 555])
            workbook.save(path)
            with self.assertRaises(ValueError):
                read_panel(path, baseline, manifest)
            manifest["source_sha256"]["Imaging_Features.xlsx"] = sha256(path)
            with self.assertRaises(ValueError):
                read_panel(path, baseline, manifest)


class NestedTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(67)
        index = pd.Index([f"P{i:03}" for i in range(75)], name="patient_id")
        self.features = pd.DataFrame(rng.uniform(1, 12, (75, len(PANEL))), index=index, columns=PANEL)
        self.features.loc["P003", VOLUME] = np.nan
        self.outcomes = pd.DataFrame({"event_type": [int(i % 3 == 0) for i in range(75)],
            "time_days": rng.uniform(10, 2000, 75), "death_gap": 0, "mri_before_diagnosis": 0}, index=index)
        self.y = Surv.from_arrays(self.outcomes.event_type == 1, self.outcomes.time_days)

    def test_inner_statistics_use_only_inner_training_and_ties_favor_stronger_penalty(self):
        # A constant MRI column remains finite even when entirely missing in training.
        x = self.features.copy()
        x[EXTRA_FEATURES[0]] = np.nan
        x[EXTRA_FEATURES[1]] = 1.
        model, tuning = tune_ridge(x, self.y, SEED, alphas=(1., 10.))
        folds = inner_assignment(x, self.y, SEED)
        for details in tuning["inner_preprocessing"]:
            train = x.loc[folds != details["fold"]]
            for column in PANEL:
                expected = train[column].median() if train[column].notna().any() else 0.
                self.assertAlmostEqual(details["imputer_train_medians"][column], expected)
                self.assertAlmostEqual(details["scaler_train_means"][column], train[column].fillna(expected).mean())
        self.assertTrue(np.isfinite(model.predict(x)).all())
        _, volume_tuning = tune_ridge(x[[VOLUME]], self.y, SEED, alphas=(1., 10.))
        scores = [c["inner_cindex"] for c in volume_tuning["candidates"]]
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(volume_tuning["selected_alpha"], 10.)

    def test_outer_test_values_and_outcomes_do_not_change_that_folds_tuning(self):
        rows = [{"patient_id": p, "event_type": self.outcomes.loc[p, "event_type"]}
                for p in self.features.index]
        folds = pd.Series(assign_folds(rows)).loc[self.features.index]
        selected = np.ones(len(self.features), dtype=bool)
        original, scores, inner = evaluate_outer(self.features, self.outcomes, folds, selected,
                                                "primary", SEED, alphas=(1., 10.))
        changed_x, changed_y = self.features.copy(), self.outcomes.copy()
        changed_x.loc[folds == 0] = 10000.
        changed_y.loc[folds == 0, "time_days"] += 100
        changed, _, _ = evaluate_outer(changed_x, changed_y, folds, selected,
                                       "primary", SEED, alphas=(1., 10.))
        for name in MODEL_COLUMNS:
            before = original["models"][name]["folds"][0]
            after = changed["models"][name]["folds"][0]
            for field in ("tuning", "imputer_train_medians", "scaler_train_means", "cox_coefficients_standardized"):
                self.assertEqual(before[field], after[field])
        self.assertEqual(len(scores), len(self.features) * len(MODEL_COLUMNS))
        self.assertEqual(len({(r["patient_id"], r["model"]) for r in scores}), len(scores))
        self.assertEqual(len(inner), len(self.features) * 4)
        for row in inner:
            self.assertNotEqual(folds.loc[row["patient_id"]], row["outer_fold"])
        # The expanded experiment retains the exact old volume-only baseline.
        baseline_features = pd.DataFrame({FEATURES[0]: 50., VOLUME: self.features[VOLUME]}, index=self.features.index)
        baseline, _ = evaluate_scenario(baseline_features, self.outcomes, folds, selected, "primary",
                                        models={"volume": [VOLUME]})
        self.assertEqual(original["models"]["volume"]["aggregate"], baseline["models"]["volume"]["aggregate"])

    def test_paired_summary_does_not_impute_missing_metrics(self):
        repeats = []
        for reference, panel in ((0.6, 0.7), (None, 0.9), (0.8, 0.7)):
            models = {m: {"aggregate": dict.fromkeys(METRICS, panel if m == "ridge_mri" else reference)}
                      for m in MODEL_COLUMNS}
            repeats.append({"scenarios": {s: {"models": models} for s in ("primary", "exclude_death_gap")}})
        stats = summarize(repeats)["primary"][METRICS[0]]["ridge_mri"]["paired_differences"]["volume"]
        self.assertEqual(stats["available"], 2)
        self.assertEqual(stats["unavailable"], 1)
        self.assertAlmostEqual(stats["mean"], 0.)


if __name__ == "__main__":
    unittest.main()
