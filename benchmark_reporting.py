"""Statistical aggregation and downloadable report formats for benchmark trials."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from vision_pipeline import CONDITIONS

METRIC_KEYS = ("precision", "recall", "f1", "failure_rate")
METHOD_LABELS = {
    "baseline": "Baseline",
    "adaptive": "Adaptive repair",
    "random_search": "Budget-matched random search",
}
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
}


def _t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom in T_CRITICAL_95:
        return T_CRITICAL_95[degrees_of_freedom]
    if degrees_of_freedom > 40:
        return 1.96
    lower_df = max(key for key in T_CRITICAL_95 if key < degrees_of_freedom)
    return T_CRITICAL_95[lower_df]


def summarize_trials(values: list[float]) -> dict[str, Any]:
    """Mean, sample standard deviation, and 95% t interval of the trial mean."""
    if not values:
        raise ValueError("At least one trial value is required.")
    numeric = [float(value) for value in values]
    average = mean(numeric)
    deviation = stdev(numeric) if len(numeric) > 1 else 0.0
    margin = (
        _t_critical_95(len(numeric) - 1) * deviation / math.sqrt(len(numeric))
        if len(numeric) > 1
        else 0.0
    )
    return {
        "n": len(numeric),
        "mean": round(average, 6),
        "std": round(deviation, 6),
        "ci95_low": round(average - margin, 6),
        "ci95_high": round(average + margin, 6),
        "ci_method": "two-sided Student t interval for the trial mean",
    }


def metric_difference(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    return {
        key: round(float(left[key]) - float(right[key]), 6)
        for key in METRIC_KEYS
    }


def paired_image_outcomes(
    left_evaluation: dict[str, Any],
    right_evaluation: dict[str, Any],
    left_method: str,
    right_method: str,
) -> dict[str, Any]:
    left_by_id = {row["sample_id"]: row for row in left_evaluation["image_results"]}
    right_by_id = {row["sample_id"]: row for row in right_evaluation["image_results"]}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("Paired method evaluations must contain identical test sample IDs.")
    counts = {
        f"{left_method}_only_successful": 0,
        f"{right_method}_only_successful": 0,
        "both_successful": 0,
        "both_failing": 0,
    }
    examples: list[dict[str, Any]] = []
    for sample_id, left in left_by_id.items():
        right = right_by_id[sample_id]
        if left["is_success"] and right["is_success"]:
            counts["both_successful"] += 1
            continue
        if not left["is_success"] and not right["is_success"]:
            counts["both_failing"] += 1
            outcome = "both_failing"
        elif left["is_success"]:
            counts[f"{left_method}_only_successful"] += 1
            outcome = f"{left_method}_only_successful"
        else:
            counts[f"{right_method}_only_successful"] += 1
            outcome = f"{right_method}_only_successful"
        examples.append(
            {
                "sample_id": sample_id,
                "condition": left["condition"],
                "outcome": outcome,
                left_method: {
                    key: left[key]
                    for key in ("tp", "fp", "fn", "is_success", "predictions")
                },
                right_method: {
                    key: right[key]
                    for key in ("tp", "fp", "fn", "is_success", "predictions")
                },
                "ground_truth": left["ground_truth"],
            }
        )
    return {
        "same_image_count": len(left_by_id),
        "same_sample_ids": True,
        "counts": counts,
        "examples": examples,
    }


def _summarize_metrics(
    trials: list[dict[str, Any]], method: str, split: str, condition: str | None = None
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = []
        for trial in trials:
            evaluation = trial["methods"][method][split]
            metrics = (
                evaluation["metrics"]
                if condition is None
                else evaluation["by_condition"][condition]
            )
            values.append(float(metrics[key]))
        summaries[key] = summarize_trials(values)
    return summaries


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("At least one completed trial is required for aggregation.")

    methods: dict[str, Any] = {}
    for method in METHOD_LABELS:
        runtime_fields = (
            "experiment_count",
            "experiment_budget",
            "selection_runtime_ms",
            "candidate_pipeline_runtime_ms",
            "test_evaluation_runtime_ms",
            "total_method_runtime_ms",
        )
        runtime_summary = {
            field: summarize_trials(
                [float(trial["methods"][method][field]) for trial in trials]
            )
            for field in runtime_fields
        }
        if method == "baseline":
            improvement_per_experiment = None
        else:
            improvement_per_experiment = summarize_trials(
                [
                    float(
                        trial["paired_differences"][f"{method}_vs_baseline"]["f1"]
                    )
                    / max(1, int(trial["methods"][method]["experiment_count"]))
                    for trial in trials
                ]
            )
        methods[method] = {
            "label": METHOD_LABELS[method],
            "validation_metrics": _summarize_metrics(trials, method, "validation"),
            "validation_by_condition": {
                condition: _summarize_metrics(
                    trials, method, "validation", condition
                )
                for condition in CONDITIONS
            },
            "test_metrics": _summarize_metrics(trials, method, "test"),
            "test_by_condition": {
                condition: _summarize_metrics(trials, method, "test", condition)
                for condition in CONDITIONS
            },
            "runtime_and_budget": runtime_summary,
            "test_f1_improvement_per_experiment": improvement_per_experiment,
        }

    relations = (
        "adaptive_vs_baseline",
        "random_search_vs_baseline",
        "adaptive_vs_random_search",
    )
    paired_differences = {
        relation: {
            metric: summarize_trials(
                [
                    float(trial["paired_differences"][relation][metric])
                    for trial in trials
                ]
            )
            for metric in METRIC_KEYS
        }
        for relation in relations
    }
    comparison_counts: dict[str, Any] = {}
    for relation in relations:
        keys = {
            key
            for trial in trials
            for key in trial["comparisons"][relation]["counts"]
        }
        comparison_counts[relation] = {
            key: sum(
                int(trial["comparisons"][relation]["counts"].get(key, 0))
                for trial in trials
            )
            for key in sorted(keys)
        }
    validation_gains = {
        method: summarize_trials(
            [
                float(trial["validation_gain_over_baseline"][key])
                for trial in trials
            ]
        )
        for method, key in (
            ("adaptive", "adaptive_f1"),
            ("random_search", "random_search_f1"),
        )
    }
    return {
        "trial_count": len(trials),
        "methods": methods,
        "paired_differences": paired_differences,
        "paired_image_outcome_counts": comparison_counts,
        "validation_f1_gain_over_baseline": validation_gains,
    }


def benchmark_conclusion(
    aggregates: dict[str, Any], trials: list[dict[str, Any]]
) -> str:
    comparison = aggregates["paired_differences"]["adaptive_vs_random_search"]["f1"]
    difference = comparison["mean"]
    low, high = comparison["ci95_low"], comparison["ci95_high"]
    adaptive_validation_gain = aggregates["validation_f1_gain_over_baseline"][
        "adaptive"
    ]["mean"]
    adaptive_test_gain = aggregates["paired_differences"]["adaptive_vs_baseline"][
        "f1"
    ]["mean"]
    if low > 0:
        return (
            f"Adaptive repair had higher held-out mean F1 than budget-matched random search by "
            f"{difference:+.4f} (paired 95% CI {low:+.4f} to {high:+.4f}) across "
            f"{len(trials)} trials. This benchmark supports an adaptive advantage for these data "
            "and parameter bounds; it does not establish a result beyond this benchmark."
        )
    if high < 0:
        return (
            f"Budget-matched random search had higher held-out mean F1 than adaptive repair by "
            f"{-difference:.4f} (paired 95% CI {low:+.4f} to {high:+.4f}) across "
            f"{len(trials)} trials. The adaptive policy's condition-targeted proposals likely "
            "covered fewer useful configurations under this candidate budget."
        )
    if adaptive_validation_gain - adaptive_test_gain >= 0.02:
        return (
            f"No clear held-out F1 advantage for adaptive repair over budget-matched random search: "
            f"paired mean difference {difference:+.4f}, 95% CI {low:+.4f} to {high:+.4f}. "
            f"Adaptive validation F1 gain ({adaptive_validation_gain:+.4f}) exceeded its held-out "
            f"gain ({adaptive_test_gain:+.4f}), consistent with a validation-to-test generalization gap."
        )
    return (
        f"No clear held-out F1 advantage for adaptive repair over budget-matched random search: "
        f"paired mean difference {difference:+.4f}, 95% CI {low:+.4f} to {high:+.4f}. "
        "The interval spans zero, so the observed between-trial variation does not support a "
        "repeatable winner at this trial count."
    )


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in report["trials"]:
        for method in METHOD_LABELS:
            result = trial["methods"][method]
            for split in ("validation", "test"):
                scopes: list[tuple[str, str, dict[str, Any]]] = [
                    (f"{split}_overall", "", result[split]["metrics"])
                ]
                scopes.extend(
                    (
                        f"{split}_condition",
                        condition,
                        result[split]["by_condition"][condition],
                    )
                    for condition in CONDITIONS
                )
                for scope, condition, metrics in scopes:
                    rows.append(
                        {
                            "record_type": "trial",
                            "trial_id": trial["trial_id"],
                            "method": method,
                            "scope": scope,
                            "condition": condition,
                            "trial_count": 1,
                            "experiment_count": result["experiment_count"],
                            "experiment_budget": result["experiment_budget"],
                            "selection_runtime_ms": result["selection_runtime_ms"],
                            "candidate_pipeline_runtime_ms": result[
                                "candidate_pipeline_runtime_ms"
                            ],
                            "test_evaluation_runtime_ms": result[
                                "test_evaluation_runtime_ms"
                            ],
                            "total_method_runtime_ms": result[
                                "total_method_runtime_ms"
                            ],
                            "test_f1_improvement_per_experiment": (
                                trial["paired_differences"][
                                    f"{method}_vs_baseline"
                                ]["f1"]
                                / max(1, result["experiment_count"])
                                if method != "baseline"
                                else ""
                            ),
                            **{key: metrics[key] for key in METRIC_KEYS},
                        }
                    )

    for method, method_report in report["aggregates"]["methods"].items():
        for split, summary_key in (
            ("validation", "validation_metrics"),
            ("test", "test_metrics"),
        ):
            metrics = method_report[summary_key]
            rows.append(
                {
                    "record_type": "aggregate_mean",
                    "trial_id": "all",
                    "method": method,
                    "scope": f"{split}_overall",
                    "condition": "",
                    "trial_count": report["trial_count"],
                    "experiment_count": method_report["runtime_and_budget"][
                        "experiment_count"
                    ]["mean"],
                    "experiment_budget": method_report["runtime_and_budget"][
                        "experiment_budget"
                    ]["mean"],
                    "selection_runtime_ms": method_report["runtime_and_budget"][
                        "selection_runtime_ms"
                    ]["mean"],
                    "candidate_pipeline_runtime_ms": method_report[
                        "runtime_and_budget"
                    ]["candidate_pipeline_runtime_ms"]["mean"],
                    "test_evaluation_runtime_ms": method_report[
                        "runtime_and_budget"
                    ]["test_evaluation_runtime_ms"]["mean"],
                    "total_method_runtime_ms": method_report[
                        "runtime_and_budget"
                    ]["total_method_runtime_ms"]["mean"],
                    "test_f1_improvement_per_experiment": (
                        method_report["test_f1_improvement_per_experiment"]["mean"]
                        if method_report["test_f1_improvement_per_experiment"]
                        else ""
                    ),
                    **{key: metrics[key]["mean"] for key in METRIC_KEYS},
                    **{f"{key}_std": metrics[key]["std"] for key in METRIC_KEYS},
                    "f1_ci95_low": metrics["f1"]["ci95_low"],
                    "f1_ci95_high": metrics["f1"]["ci95_high"],
                }
            )
            condition_key = f"{split}_by_condition"
            for condition, condition_metrics in method_report[condition_key].items():
                rows.append(
                    {
                        "record_type": "aggregate_mean",
                        "trial_id": "all",
                        "method": method,
                        "scope": f"{split}_condition",
                        "condition": condition,
                        "trial_count": report["trial_count"],
                        **{key: condition_metrics[key]["mean"] for key in METRIC_KEYS},
                        **{
                            f"{key}_std": condition_metrics[key]["std"]
                            for key in METRIC_KEYS
                        },
                        "f1_ci95_low": condition_metrics["f1"]["ci95_low"],
                        "f1_ci95_high": condition_metrics["f1"]["ci95_high"],
                    }
                )
    return rows


def save_benchmark_report(
    report: dict[str, Any],
    output_dir: str | Path = "generated-artifacts/benchmark",
) -> dict[str, str]:
    """Save new benchmark files without overwriting the existing single-run report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "latest_benchmark.json"
    csv_path = directory / "latest_benchmark.csv"
    report["saved_paths"] = {"json": str(json_path), "csv": str(csv_path)}

    json_temp = json_path.with_suffix(".json.tmp")
    json_temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    json_temp.replace(json_path)

    csv_output = io.StringIO(newline="")
    rows = _csv_rows(report)
    fieldnames = list(dict.fromkeys(field for row in rows for field in row.keys()))
    writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_temp = csv_path.with_suffix(".csv.tmp")
    csv_temp.write_text(csv_output.getvalue(), encoding="utf-8")
    csv_temp.replace(csv_path)
    return report["saved_paths"]
