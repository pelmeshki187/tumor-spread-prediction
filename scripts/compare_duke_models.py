"""Compare fixed survival models on the shared Duke patient splits."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

from check_duke_stability import METRICS, SEEDS, describe
from prepare_duke_baseline import FEATURES, N_FOLDS, SEED, assign_folds, save_csv, sha256
from run_duke_baseline import HORIZON_DAYS, evaluate_scenario, load_prepared, make_model


MODELS = {"age": [FEATURES[0]], "volume": [FEATURES[1]],
          "age_volume": list(FEATURES), "forest": list(FEATURES), "boosting": list(FEATURES)}
MODEL_SEED = 20261026
TREE_PARAMETERS = {
    "forest": {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 20,
               "min_samples_split": 40, "max_features": 1.0, "bootstrap": True,
               "n_jobs": 1, "random_state": MODEL_SEED, "low_memory": True},
    "boosting": {"loss": "coxph", "n_estimators": 100, "learning_rate": 0.03,
                 "max_depth": 1, "min_samples_leaf": 20, "min_samples_split": 40,
                 "subsample": 1.0, "max_features": None, "n_iter_no_change": None,
                 "random_state": MODEL_SEED}}
LABELS = {"age": "Cox: возраст", "volume": "Cox: объём", "age_volume": "Cox: возраст + объём",
          "forest": "Random Survival Forest", "boosting": "Survival-бустинг"}
SCENARIOS = ("primary", "exclude_death_gap")


def build_model(name):
    if name in ("age", "volume", "age_volume"):
        return make_model()
    constructors = {"forest": RandomSurvivalForest, "boosting": GradientBoostingSurvivalAnalysis}
    estimator = constructors[name](**TREE_PARAMETERS[name])
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def summarize_comparison(repeats):
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for metric in METRICS:
            values = {model: [r["scenarios"][scenario]["models"][model]["aggregate"][metric]
                              for r in repeats] for model in MODELS}
            summary[scenario][metric] = {}
            for model, scores in values.items():
                differences = [a - b if a is not None and b is not None else None
                               for a, b in zip(scores, values["age_volume"])]
                summary[scenario][metric][model] = {
                    "scores": describe(scores),
                    "paired_difference_vs_age_volume": describe(differences) | {
                        "positive_repeats": sum(v is not None and v > 0 for v in differences)}}
    return summary


def compare(features, outcomes, seeds=SEEDS):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    rows = [{"patient_id": p, "event_type": int(outcomes.loc[p, "event_type"])}
            for p in features.index]
    repeats, assignments, predictions = [], [], []
    for seed in seeds:
        folds = pd.Series(assign_folds(rows, seed=seed)).loc[features.index]
        assignments.extend({"seed": seed, "patient_id": p, "fold": int(folds.loc[p])}
                           for p in features.index)
        repeat = {"seed": seed, "scenarios": {}}
        for name, selected in (("primary", np.ones(len(features), dtype=bool)),
                               ("exclude_death_gap", outcomes["death_gap"] == 0)):
            scenario, scores = evaluate_scenario(features, outcomes, folds, selected, name,
                models=MODELS, model_factory=build_model, score_field="risk_score")
            repeat["scenarios"][name] = scenario
            predictions.extend({"seed": seed, **score} for score in scores)
        repeats.append(repeat)
        print(f"Completed comparison {len(repeats)}/{len(seeds)} (seed={seed})", flush=True)
    return repeats, assignments, predictions


def write_report(path, result):
    def fmt(stats):
        if stats["mean"] is None:
            return "недоступно"
        return f"{stats['mean']:.3f} [{stats['minimum']:.3f}; {stats['maximum']:.3f}]"

    lines = ["# Duke: сравнение пяти survival-моделей", "",
        "674 пациента / 47 отдалённых рецидивов; 20 повторных пятиблочных разбиений. "
        "Все методы обучены заново на общих блоках с обработкой признаков только по "
        "обучающим пациентам. Исход, цензурирование и метрики соответствуют "
        "[протоколу](../../research/duke-protocol.md).", "",
        "## Фиксированные методы", "",
        "Три Cox-модели: возраст, log-объём, возраст + log-объём; alpha=0, ties=Efron. "
        "Лес и бустинг получают возраст и log-объём. Медиана и стандартизация обучаются "
        "в каждом блоке отдельно для каждой модели. Дополнительных признаков нет.", "",
        "Random Survival Forest: 200 деревьев, глубина 3, минимум 20 пациентов в листе, "
        "40 для разделения, оба признака доступны при разделении, bootstrap=True. "
        "Бустинг: Cox loss, 100 деревьев глубины 1, learning_rate=0.03, минимум 20 "
        "пациентов в листе, 40 для разделения, subsample=1, без early stopping. "
        f"Random state обеих моделей — {MODEL_SEED}, один поток леса. Параметры выбраны "
        "до расчёта этого сравнения, без подбора по метрикам. Полные параметры "
        "созданных estimator-ов сохранены в JSON.", "",
        "Ограничения сложности выбраны с учётом малого числа событий. Это сравнение "
        "конкретных конфигураций; оно не определяет лучшую достижимую настройку каждого "
        "семейства. Для подбора понадобится внутренняя валидация. "
        "[Лес](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html), "
        "[бустинг](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html).", "",
        "Прогноз — выход `predict()` соответствующей модели: больший балл означает "
        "больший риск. Для леса это суммарная накопленная интенсивность по временам "
        "событий обучающего блока, для Cox и Cox-бустинга — балл log-hazard. "
        "Сравнивается порядок внутри каждого блока; баллы разных моделей/блоков "
        "не объединяются в общую числовую шкалу.", "",
        "## Основная когорта", "",
        "Среднее [минимум; максимум] по повторениям. Разности парные, относительно "
        "Cox по возрасту и объёму на том же разбиении. Диапазоны не являются "
        "доверительными интервалами.", "",
        "| Метод | C-index | IPCW C, 3 года | Разность C | Разность IPCW C | C выше базового / доступно |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    primary = result["summary"]["primary"]
    for model in MODELS:
        c, ipcw = (primary[metric][model] for metric in METRICS)
        d = c["paired_difference_vs_age_volume"]
        lines.append(f"| {LABELS[model]} | {fmt(c['scores'])} | {fmt(ipcw['scores'])} | "
                     f"{fmt(d)} | {fmt(ipcw['paired_difference_vs_age_volume'])} | "
                     f"{d['positive_repeats']} / {d['available']} |")
    lines += ["", "## Исключение неоднозначного наблюдения перед смертью", "",
        "665 пациентов / 47 событий, номера блоков сохранены из основной когорты.", "",
        "| Метод | C-index | IPCW C, 3 года | Разность C |",
        "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        c, ipcw = (result["summary"]["exclude_death_gap"][metric][model] for metric in METRICS)
        lines.append(f"| {LABELS[model]} | {fmt(c['scores'])} | {fmt(ipcw['scores'])} | "
                     f"{fmt(c['paired_difference_vs_age_volume'])} |")
    lines += ["", "## Ограничения и следующий этап", "",
        "Семейства моделей добавлены после просмотра результатов первых Cox-моделей; "
        "новые параметры зафиксированы до запуска, но эта когорта уже использована "
        "для развития исследования. Повторения зависимы; положительные разности "
        "не являются статистическим тестом. Выбор победителя по этой таблице "
        "потребует дальнейшей независимой проверки. Недоступные IPCW сохраняются "
        "с причиной, доступное число повторений приведено в JSON.", "",
        "Следующий этап — расширение входной информации: заранее обоснованный набор "
        "признаков МРТ и регуляризованная модель, с отбором/настройкой только внутри "
        "обучающих блоков. Отдельно остаются выборочная неопределённость с переобучением, "
        "проверка пропорциональности рисков Cox и внешняя валидация. Абсолютные риски, "
        "калибровка и прогноз появления очагов по органам здесь не оценивались.", "",
        "Полные метрики, настройки, версии и SHA-256: "
        "[model_comparison_results.json](model_comparison_results.json). "
        "Индивидуальные баллы и блоки — только в `data/derived/duke/baseline/` "
        "(`comparison_scores.csv`, `comparison_folds.csv`).", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/derived/duke/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/duke"))
    args = parser.parse_args()
    started = time.perf_counter()
    features, outcomes, original_folds, manifest = load_prepared(args.input_dir)
    rows = [{"patient_id": p, "event_type": int(outcomes.loc[p, "event_type"])}
            for p in features.index]
    if not np.array_equal(original_folds, pd.Series(assign_folds(rows, seed=SEED)).loc[features.index]):
        raise ValueError("Original folds do not match the first comparison split")
    repeats, assignments, predictions = compare(features, outcomes)
    result = {"status": "exploratory_fixed_model_comparison_not_confidence_intervals",
        "inputs": manifest,
        "configuration": {"seeds": list(SEEDS), "n_folds": N_FOLDS, "features": MODELS,
            "horizon_days": HORIZON_DAYS, "tuning": "none; fixed before this comparison",
            "preprocessing": "training-fold median and standard scaling for each model",
            "death_handling": "same as baseline; death is censored, not a distant event",
            "sensitivity": "exclude death_gap, retain primary folds",
            "aggregation": "Harrell: within-fold pair totals; IPCW: equal-fold mean",
            "estimator_parameters": {name: build_model(name).steps[-1][1].get_params() for name in MODELS}},
        "software": {p: version(p) for p in ("scikit-survival", "scikit-learn", "numpy", "scipy", "pandas", "openpyxl")},
        "python": platform.python_version(),
        "script_sha256": {p.name: sha256(p) for p in (Path(__file__),
            Path(__file__).with_name("run_duke_baseline.py"),
            Path(__file__).with_name("check_duke_stability.py"),
            Path(__file__).with_name("prepare_duke_baseline.py"),
            Path(__file__).with_name("audit_duke.py"))},
        "repeats": repeats, "summary": summarize_comparison(repeats),
        "elapsed_seconds": time.perf_counter() - started}
    save_csv(args.input_dir / "comparison_folds.csv", assignments, ["seed", "patient_id", "fold"])
    save_csv(args.input_dir / "comparison_scores.csv", predictions,
             ["seed", "patient_id", "scenario", "model", "fold", "risk_score"])
    result["artifact_sha256"] = {name: sha256(args.input_dir / name)
        for name in ("comparison_folds.csv", "comparison_scores.csv")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_comparison_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(args.output_dir / "model_comparison.md", result)
    print(json.dumps(result["summary"]["primary"], indent=2))


if __name__ == "__main__":
    main()
