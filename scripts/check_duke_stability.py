"""Repeat the fixed Duke Cox comparison on predeclared patient splits."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from prepare_duke_baseline import N_FOLDS, SEED, assign_folds, save_csv, sha256
from run_duke_baseline import HORIZON_DAYS, MODELS, evaluate_scenario, load_prepared


# Includes the original split for a direct reproduction check. No seed selection.
SEEDS = tuple(range(SEED, SEED + 20))
METRICS = ("harrell_within_fold_pairs", "ipcw_3y_fold_mean")


def describe(values):
    available = [float(v) for v in values if v is not None]
    if not all(np.isfinite(available)):
        raise ValueError("Non-finite stability metric")
    return {"available": len(available), "unavailable": len(values) - len(available),
            "mean": float(np.mean(available)) if available else None,
            "minimum": min(available) if available else None,
            "maximum": max(available) if available else None}


def summarize(repeats):
    summary = {}
    for name in ("primary", "exclude_death_gap"):
        summary[name] = {}
        for metric in METRICS:
            values = {model: [r["scenarios"][name]["models"][model]["aggregate"][metric]
                              for r in repeats] for model in MODELS}
            differences = [b - a if a is not None and b is not None else None
                           for a, b in zip(values["age"], values["age_volume"])]
            summary[name][metric] = {
                "models": {model: describe(v) for model, v in values.items()},
                "paired_difference_age_volume_minus_age": describe(differences) | {
                    "positive_repeats": sum(v is not None and v > 0 for v in differences)}}
    return summary


def run_repeats(features, outcomes, seeds=SEEDS):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    rows = [{"patient_id": p, "event_type": int(outcomes.loc[p, "event_type"])}
            for p in features.index]
    repeats, assignments, predictions = [], [], []
    for seed in seeds:
        mapping = assign_folds(rows, seed=seed)
        folds = pd.Series(mapping).loc[features.index]
        assignments.extend({"seed": seed, "patient_id": p, "fold": int(folds.loc[p])}
                           for p in features.index)
        repeat = {"seed": seed, "scenarios": {}}
        for name, selected in (("primary", np.ones(len(features), dtype=bool)),
                               ("exclude_death_gap", outcomes["death_gap"] == 0)):
            scenario, scores = evaluate_scenario(features, outcomes, folds, selected, name)
            repeat["scenarios"][name] = scenario
            predictions.extend({"seed": seed, **score} for score in scores)
        repeats.append(repeat)
        print(f"Completed split {len(repeats)}/{len(seeds)} (seed={seed})", flush=True)
    return repeats, assignments, predictions


def write_report(path, result):
    count = len(result["configuration"]["seeds"])
    primary = result["repeats"][0]["scenarios"]["primary"]
    lines = ["# Duke: устойчивость сравнения Cox-моделей", "",
        f"Рабочая когорта: {primary['patients']} пациентов / {primary['distant_events']} событий. "
        f"Выполнено {count} повторений пятиблочной валидации с полным переобучением "
        "обработки признаков и моделей внутри каждого обучающего блока.", "",
        "## Зафиксированная схема", "",
        f"Seeds: {', '.join(map(str, result['configuration']['seeds']))}. "
        "Первое разбиение воспроизводит исходный эксперимент. Все разбиения включены "
        "в сводку, отбора по качеству нет. Для двух моделей используются общие блоки; "
        "в сценарии исключения неоднозначного наблюдения перед смертью номера блоков "
        "сохраняются. Возраст, log-объём, параметры Cox и правила исхода совпадают "
        "с [первым экспериментом](baseline.md).", "",
        "C-index агрегируется по сравнимым парам внутри блоков, IPCW C до трёх лет — "
        "средним по пяти блокам. Для каждого повторения сначала вычисляется разность "
        "метрик двух моделей. Недоступная метрика не заменяется нулём; сводка показывает "
        "число доступных повторений, а причины сохраняются в JSON.", "",
        "## Результаты", "",
        "В ячейках: среднее [минимум; максимум] по повторениям.", "",
        "| Сценарий | Метрика | Возраст | Возраст + объём | Парная разность | Разность > 0 / доступно |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]

    def fmt(stats):
        if stats["mean"] is None:
            return "недоступно"
        return f"{stats['mean']:.3f} [{stats['minimum']:.3f}; {stats['maximum']:.3f}]"

    labels = {"primary": "Основной", "exclude_death_gap": "Без death gap"}
    metric_labels = {METRICS[0]: "C-index", METRICS[1]: "IPCW C, 3 года"}
    for name, metrics in result["summary"].items():
        for metric, entry in metrics.items():
            diff = entry["paired_difference_age_volume_minus_age"]
            lines.append(f"| {labels[name]} | {metric_labels[metric]} | "
                         f"{fmt(entry['models']['age'])} | {fmt(entry['models']['age_volume'])} | "
                         f"{fmt(diff)} | {diff['positive_repeats']} / {diff['available']} |")
    lines += ["", "## Что проверено и что остаётся", "",
        "Повторения используют одних и тех же пациентов и зависимы между собой. "
        "Диапазоны характеризуют чувствительность к разбиению этой когорты; "
        "это не доверительные интервалы и не оценка неопределённости на новых пациентах. "
        "Доля положительных разностей не является p-value или вероятностью превосходства. "
        "Стратификация сохраняет число событий в блоках и может уменьшать наблюдаемый "
        "разброс. [Пояснения scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-with-stratification-based-on-class-labels).", "",
        "Следующий отдельный этап — оценка неопределённости при повторной выборке пациентов "
        "с переобучением всего процесса и защита от попадания копий одного пациента "
        "одновременно в обучение и проверку. Внешней валидации, абсолютного риска "
        "и калибровки пока нет; ограничения исходов и происхождения признаков сохраняются.", "",
        "Полные метрики блоков, параметры обработки, версии и SHA-256: "
        "[stability_results.json](stability_results.json). Номера блоков и прогнозы "
        "по пациентам сохранены только в игнорируемой Git папке `data/derived/duke/baseline/`.", ""]
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
    expected_folds = pd.Series(assign_folds(rows)).loc[features.index]
    if not np.array_equal(original_folds.to_numpy(), expected_folds.to_numpy()):
        raise ValueError("Original folds do not match the predeclared first split")
    repeats, assignments, predictions = run_repeats(features, outcomes)
    result = {"status": "split_sensitivity_only_not_sampling_confidence_intervals",
        "inputs": manifest,
        "configuration": {"seeds": list(SEEDS), "n_folds": N_FOLDS, "models": MODELS,
            "alpha": 0, "ties": "efron", "horizon_days": HORIZON_DAYS,
            "aggregation": "Harrell: within-fold pairs; IPCW: equal-fold mean; paired differences per repeat",
            "sensitivity": "exclude death_gap patients, retaining primary fold assignments"},
        "software": {p: version(p) for p in ("scikit-survival", "scikit-learn", "numpy", "scipy", "pandas", "openpyxl")},
        "python": platform.python_version(),
        "script_sha256": {p.name: sha256(p) for p in (Path(__file__),
            Path(__file__).with_name("run_duke_baseline.py"),
            Path(__file__).with_name("prepare_duke_baseline.py"),
            Path(__file__).with_name("audit_duke.py"))},
        "repeats": repeats, "summary": summarize(repeats),
        "elapsed_seconds": time.perf_counter() - started}
    save_csv(args.input_dir / "stability_folds.csv", assignments, ["seed", "patient_id", "fold"])
    save_csv(args.input_dir / "stability_scores.csv", predictions,
             ["seed", "patient_id", "scenario", "model", "fold", "linear_predictor"])
    result["artifact_sha256"] = {name: sha256(args.input_dir / name)
        for name in ("stability_folds.csv", "stability_scores.csv")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stability_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(args.output_dir / "stability.md", result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
