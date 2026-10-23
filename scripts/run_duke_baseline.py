"""Evaluate two fixed Cox baselines on shared, held-out patient folds."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from prepare_duke_baseline import FEATURES, N_FOLDS, SEED, sha256


MODELS = {"age": ["age_mri_years"], "age_volume": list(FEATURES)}
HORIZON_DAYS = 3 * 365.25


def validate_feature_schema(frame):
    if set(frame.columns) != {"patient_id", *FEATURES}:
        raise ValueError("Feature table must contain only patient_id and the fixed feature allowlist")
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        raise ValueError("Missing or duplicate feature patient IDs")


def load_prepared(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["prepared_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Prepared file changed after preparation: {name}")
    features = pd.read_csv(directory / "features.csv")
    validate_feature_schema(features)
    outcomes = pd.read_csv(directory / "outcomes.csv")
    folds = pd.read_csv(directory / "folds.csv")
    if set(outcomes.columns) != {"patient_id", "time_days", "event_type", "death_gap", "mri_before_diagnosis"}:
        raise ValueError("Unexpected outcome schema")
    if set(folds.columns) != {"patient_id", "fold"}:
        raise ValueError("Unexpected fold schema")
    for frame in (outcomes, folds):
        if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
            raise ValueError("Missing or duplicate outcome/fold patient IDs")
        if set(frame["patient_id"]) != set(features["patient_id"]):
            raise ValueError("Feature/outcome/fold ID sets differ")
    if manifest["seed"] != SEED or manifest["n_folds"] != N_FOLDS:
        raise ValueError("Prepared folds disagree with the fixed protocol")
    features = features.sort_values("patient_id").set_index("patient_id")
    outcomes = outcomes.set_index("patient_id").loc[features.index]
    folds = folds.set_index("patient_id").loc[features.index, "fold"]
    if set(folds) != set(range(N_FOLDS)):
        raise ValueError("Missing or unexpected fold labels")
    if not outcomes["event_type"].isin([0, 1, 2]).all():
        raise ValueError("Unknown event type")
    if not all(outcomes[key].isin([0, 1]).all() for key in ("death_gap", "mri_before_diagnosis")):
        raise ValueError("Unknown quality flag")
    if not np.isfinite(outcomes["time_days"]).all() or (outcomes["time_days"] <= 0).any():
        raise ValueError("Follow-up must be finite and positive")
    if not np.isfinite(features["age_mri_years"]).all():
        raise ValueError("Missing or invalid age")
    volume = features["log1p_tumor_volume_mm3"].to_numpy()
    if np.isinf(volume).any() or np.any(volume[np.isfinite(volume)] <= 0):
        raise ValueError("Invalid logged volume")
    return features, outcomes, folds, manifest


def make_model():
    # No outcome-driven selection or tuning; all preprocessing is fitted in fit().
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         CoxPHSurvivalAnalysis(alpha=0, ties="efron", n_iter=100, tol=1e-9))


def harrell_metrics(y, scores):
    cindex, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        y["event"], y["time"], scores)
    return {"cindex": float(cindex), "concordant": int(concordant),
        "discordant": int(discordant), "tied_risk": int(tied_risk),
        "tied_time": int(tied_time),
        "comparable_pairs": int(concordant + discordant + tied_risk)}


def ipcw_metrics(y_train, y_test, scores, horizon=HORIZON_DAYS):
    # Keep late-followed patients as comparators at tau, not by dropping them.
    # No observed event at/after tau contributes as a failure to this metric.
    train_at_risk = int(np.sum(y_train["time"] >= horizon))
    result = {"horizon_days": horizon, "train_at_risk_at_horizon": train_at_risk,
              "g_train_at_horizon": None, "cindex": None, "unavailable_reason": None}
    if train_at_risk == 0:
        result["unavailable_reason"] = "No training follow-up supports the fixed horizon"
        return result
    g = float(CensoringDistributionEstimator().fit(y_train).predict_proba([horizon])[0])
    result["g_train_at_horizon"] = g
    if not np.isfinite(g) or g <= 0:
        result["unavailable_reason"] = "Training censoring survival is zero/invalid at horizon"
        return result
    restricted = Surv.from_arrays(y_test["event"] & (y_test["time"] < horizon),
                                 np.minimum(y_test["time"], horizon))
    try:
        score = concordance_index_ipcw(y_train, restricted, scores, tau=horizon)[0]
    except ValueError as error:
        result["unavailable_reason"] = str(error)
        return result
    if not np.isfinite(score):
        result["unavailable_reason"] = "IPCW concordance not finite"
    else:
        result["cindex"] = float(score)
    return result


def aggregate_folds(rows):
    pairs = sum(r["harrell"]["comparable_pairs"] for r in rows)
    numerator = sum(r["harrell"]["concordant"] + 0.5*r["harrell"]["tied_risk"] for r in rows)
    ipcw = [r["ipcw_3y"]["cindex"] for r in rows]
    return {"harrell_within_fold_pairs": numerator / pairs,
        "comparable_pairs": pairs,
        "harrell_fold_mean": float(np.mean([r["harrell"]["cindex"] for r in rows])),
        "harrell_fold_range": [min(r["harrell"]["cindex"] for r in rows),
                               max(r["harrell"]["cindex"] for r in rows)],
        "ipcw_3y_fold_mean": float(np.mean(ipcw)) if all(v is not None for v in ipcw) else None,
        "ipcw_3y_available_folds": sum(v is not None for v in ipcw)}


def evaluate_scenario(features, outcomes, folds, selected, name):
    n = int(np.sum(selected))
    scenario = {"name": name, "patients": n,
        "distant_events": int(np.sum(selected & (outcomes["event_type"] == 1))), "models": {}}
    predictions = []
    for model_name, columns in MODELS.items():
        fold_results = []
        for fold in range(N_FOLDS):
            train = np.asarray(selected & (folds != fold))
            test = np.asarray(selected & (folds == fold))
            train_outcome, test_outcome = outcomes.iloc[train], outcomes.iloc[test]
            y_train = Surv.from_arrays(train_outcome["event_type"] == 1, train_outcome["time_days"])
            y_test = Surv.from_arrays(test_outcome["event_type"] == 1, test_outcome["time_days"])
            if not y_train["event"].any() or not y_test["event"].any():
                raise ValueError("Every training/test subset needs observed distant events")
            model = make_model()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(features.iloc[train][columns], y_train)
            scores = model.predict(features.iloc[test][columns])
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite held-out scores")
            imputer, scaler, cox = model.steps[0][1], model.steps[1][1], model.steps[2][1]
            fold_results.append({"fold": fold, "train_patients": int(train.sum()),
                "test_patients": int(test.sum()), "train_events": int(y_train["event"].sum()),
                "test_events": int(y_test["event"].sum()),
                "imputer_train_medians": dict(zip(columns, imputer.statistics_.tolist())),
                "scaler_train_means": dict(zip(columns, scaler.mean_.tolist())),
                "scaler_train_scales": dict(zip(columns, scaler.scale_.tolist())),
                "cox_coefficients_standardized": dict(zip(columns, cox.coef_.tolist())),
                "harrell": harrell_metrics(y_test, scores),
                "ipcw_3y": ipcw_metrics(y_train, y_test, scores)})
            for patient_id, score in zip(features.index[test], scores):
                predictions.append({"patient_id": patient_id, "scenario": name,
                    "model": model_name, "fold": fold, "linear_predictor": float(score)})
        scenario["models"][model_name] = {"features": columns, "folds": fold_results,
                                         "aggregate": aggregate_folds(fold_results)}
    expected = n * len(MODELS)
    keys = {(r["patient_id"], r["model"]) for r in predictions}
    if len(predictions) != expected or len(keys) != expected:
        raise ValueError("Each patient must have exactly one held-out score per model")
    return scenario, predictions


def write_report(path, result):
    primary = result["scenarios"]["primary"]
    inputs = result["inputs"]
    lines = ["# Duke: первые Cox-baseline", "",
        "Результат `scripts/run_duke_baseline.py`. Исследовательское ранжирование "
        "времени до отдалённого рецидива, без оценки абсолютной вероятности.", "",
        f"Основная рабочая когорта: **{primary['patients']} пациентов, "
        f"{primary['distant_events']} отдалённых событий**. "
        f"У {inputs['volume']['missing']} пациента отсутствует объём; "
        "медиана для заполнения и масштабирование вычисляются только на обучающем блоке.", "",
        "## Входы и схема оценки", "",
        "Возраст на МРТ и заранее выбранный `log1p(Volume_cu_mm_Tumor)`; остальные "
        "поля клиники и исходов не поступают в модель. Объём указан издателем в мм³. "
        "Это опубликованный признак по полуавтоматической маске исходной опухоли, "
        "а не независимо воспроизведённая сегментация или объём bounding box. "
        "[Описание извлечения признаков](https://www.nature.com/articles/s41416-018-0185-8).", "",
        f"Пять общих блоков по пациентам, seed={SEED}; стратификация по наблюдаемому "
        "отдалённому событию. Cox без штрафа, ties=Efron. Признаки и параметры "
        "зафиксированы до расчёта метрик; подбора нет. Все оценки ниже — на пациентах, "
        "которых соответствующая модель не видела при обучении.", "",
        "## Основное сравнение", "",
        "| Модель | C-index: пары внутри блоков | Средний IPCW C до 3 лет | Диапазон C по блокам |",
        "| --- | ---: | ---: | --- |"]
    labels = {"age": "Возраст", "age_volume": "Возраст + log-объём МРТ"}
    for model_name, model in primary["models"].items():
        a = model["aggregate"]
        ipcw = f"{a['ipcw_3y_fold_mean']:.3f}" if a["ipcw_3y_fold_mean"] is not None else "недоступен"
        lines.append(f"| {labels[model_name]} | {a['harrell_within_fold_pairs']:.3f} | "
                     f"{ipcw} | {a['harrell_fold_range'][0]:.3f}–{a['harrell_fold_range'][1]:.3f} |")
    lines += ["", "C-index по парам агрегирует только сравнения внутри соответствующего "
        "валидационного блока. Предсказания разных моделей блоков не сравниваются между собой "
        "как единая шкала. IPCW C — среднее по пяти блокам; это дополнительная оценка "
        "с допущением независимого цензурирования. Диапазон по блокам не является "
        "доверительным интервалом. Ориентир C=0.5 соответствует неинформативному ранжированию.", "",
        "IPCW ограничен событиями до 1095.75 дня; пациенты с более длинным наблюдением "
        "сохранены как сравнения на горизонте. Распределение цензурирования оценивается "
        "только по обучающему блоку. Поддержка на горизонте и значения G сохранены "
        "для каждого блока. [Документация метрики](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html).", "",
        "## Проверка чувствительности: неоднозначное наблюдение перед смертью", "",
        "В основном расчёте смерть, наступившая позже последней специализированной оценки "
        "без рецидива, не продлевает наблюдение; цензурируем по этой оценке. Во втором "
        "сценарии такие пациенты исключаются из обучения и оценки, с сохранением исходных "
        "номеров блоков. Наблюдаемая конкурирующая смерть цензурируется в Cox на дату "
        "смерти; она не становится отдалённым рецидивом.", "",
        "| Сценарий | Пациенты / события | C: возраст | C: возраст + объём | IPCW C: возраст | IPCW C: возраст + объём |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, scenario in result["scenarios"].items():
        a, b = (scenario["models"][m]["aggregate"] for m in MODELS)
        def fmt(value):
            return "недоступен" if value is None else f"{value:.3f}"
        lines.append(f"| {name} | {scenario['patients']} / {scenario['distant_events']} | "
                     f"{fmt(a['harrell_within_fold_pairs'])} | {fmt(b['harrell_within_fold_pairs'])} | "
                     f"{fmt(a['ipcw_3y_fold_mean'])} | {fmt(b['ipcw_3y_fold_mean'])} |")
    lines += ["", "## Ограничения", "",
        "Это одно разбиение небольшой ретроспективной одноцентровой когорты, "
        "без внешней проверки и без доверительных интервалов. Разность оценок "
        "не доказывает статистически подтверждённого превосходства. Для оценки "
        "неопределённости нужно повторять весь процесс обучения и проверки.", "",
        "Метрики относятся к ранжированию при принятом правиле цензурирования; "
        "конкурирующая смерть и потери наблюдения могут нарушать допущения. "
        "C-index не является точностью, процентом правильно предсказанных "
        "метастазов или калиброванной вероятностью. CIF, калибровка и clinical utility "
        "не оценивались. Смешанное происхождение M-стадии и доступность исходной "
        "клиники на t0 остаются ограничениями отбора. Признак объёма заимствован "
        "из публичного релиза; его воспроизводимость на сырых МРТ здесь не проверена.", "",
        "Подробные метрики, параметры обработки, версии и SHA-256: "
        "[baseline_results.json](baseline_results.json). Аудит входов: "
        "[baseline_inputs.json](baseline_inputs.json). Индивидуальные признаки, "
        "исходы, номера блоков и held-out scores сохраняются только в `data/`.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/derived/duke/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/duke"))
    args = parser.parse_args()
    started = time.perf_counter()
    features, outcomes, folds, manifest = load_prepared(args.input_dir)
    result = {"status": "exploratory_cause_specific_ranking_only", "inputs": manifest,
        "configuration": {"models": MODELS, "alpha": 0, "ties": "efron", "horizon_days": HORIZON_DAYS,
            "death_handling": "censor at death if ascertained disease-free until death; otherwise last DR-free assessment",
            "aggregation": "Harrell: within-fold pair totals; IPCW: equal-fold mean; no cross-fold pairs"},
        "software": {p: version(p) for p in ("scikit-survival", "scikit-learn", "numpy", "scipy", "pandas", "openpyxl")},
        "python": platform.python_version(),
        "script_sha256": {p.name: sha256(p) for p in (Path(__file__),
            Path(__file__).with_name("prepare_duke_baseline.py"), Path(__file__).with_name("audit_duke.py"))},
        "scenarios": {}}
    predictions = []
    for name, selected in (("primary", np.ones(len(features), dtype=bool)),
                           ("exclude_death_gap", outcomes["death_gap"] == 0)):
        scenario, scores = evaluate_scenario(features, outcomes, folds, selected, name)
        result["scenarios"][name] = scenario
        predictions.extend(scores)
    result["elapsed_seconds"] = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.input_dir / "held_out_scores.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, ["patient_id", "scenario", "model", "fold", "linear_predictor"])
        writer.writeheader()
        writer.writerows(predictions)
    write_report(args.output_dir / "baseline.md", result)
    print(json.dumps({name: {"patients": s["patients"], "distant_events": s["distant_events"],
        "models": {m: v["aggregate"] for m, v in s["models"].items()}}
        for name, s in result["scenarios"].items()}, indent=2))


if __name__ == "__main__":
    main()
