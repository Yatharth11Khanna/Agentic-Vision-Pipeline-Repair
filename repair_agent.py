"""A bounded, validation-driven parameter search with an auditable held-out run."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from vision_pipeline import (
    BASELINE_CONFIG,
    CONDITION_LABELS,
    CONDITIONS,
    IOU_MATCH_THRESHOLD,
    SPLIT_SEEDS,
    SPLIT_SIZES,
    DetectorConfig,
    SyntheticSample,
    config_as_dict,
    evaluate_dataset,
    metric_changes,
    opencv_version_status,
)

EXPERIMENT_BUDGET = 12
MIN_MEANINGFUL_IMPROVEMENT = 0.002
MAX_NO_IMPROVEMENT_STREAK = 4

ProgressCallback = Callable[[dict[str, Any]], None]


def validation_objective(metrics: dict[str, Any]) -> float:
    # Penalize image-level errors while keeping F1 as the dominant score.
    return float(metrics["f1"]) - 0.10 * float(metrics["failure_rate"])


def diagnose_failure(evaluation: dict[str, Any], condition: str) -> dict[str, Any]:
    """Separate measured condition counts from heuristic action recommendations."""
    condition_metrics = evaluation["by_condition"][condition]
    measured = {
        "condition": condition,
        "images": condition_metrics["images"],
        "false_positives": condition_metrics["fp"],
        "false_negatives": condition_metrics["fn"],
        "precision": condition_metrics["precision"],
        "recall": condition_metrics["recall"],
        "failure_rate": condition_metrics["failure_rate"],
    }
    if condition_metrics["fn"] > 0 and condition_metrics["fp"] > 0:
        mode = "mixed missed-target and false-positive failures"
        heuristic = "Compare segmentation recall and geometry rejection actions; do not widen filters without false-positive control."
    elif condition_metrics["fn"] > 0:
        mode = "missed target failures dominate"
        heuristic = "Prefer recall-oriented segmentation changes such as lower thresholds, wider hue tolerance, or blur/circularity relaxation."
    elif condition_metrics["fp"] > 0:
        mode = "false-positive failures dominate"
        heuristic = "Prefer precision-oriented geometric changes such as larger minimum area or stronger morphology/circularity filtering."
    else:
        mode = "no measured failure count in this condition"
        heuristic = "No failure-guided action is required."
    return {
        "measured_evidence": measured,
        "diagnosis": mode,
        "heuristic_basis": heuristic,
        "evidence_type": "measured metrics; diagnosis classifies dominant failure counts, not causal image analysis",
    }


def _diagnosis_action_priorities(diagnosis: dict[str, Any]) -> tuple[str, ...]:
    """Map measured FP/FN evidence to broad parameter priorities within existing proposals."""
    measured = diagnosis["measured_evidence"]
    false_positives = measured["false_positives"]
    false_negatives = measured["false_negatives"]
    if false_negatives > 0 and false_positives == 0:
        return ("recall", "balanced", "precision")
    if false_positives > 0 and false_negatives == 0:
        return ("precision", "balanced", "recall")
    return ("balanced", "recall", "precision")


def order_proposals_for_diagnosis(
    proposals: list[tuple[str, DetectorConfig]], diagnosis: dict[str, Any]
) -> list[tuple[str, DetectorConfig]]:
    """Deterministically prioritize existing safe proposals for the measured failure mode."""
    precision_terms = ("area", "circularity", "morphology", "reject", "false")
    recall_terms = ("lower", "widen", "relax", "blur", "preserve", "admit")

    def group(reason_config: tuple[str, DetectorConfig]) -> tuple[int, str, DetectorConfig]:
        reason = reason_config[0].lower()
        if any(term in reason for term in precision_terms):
            parameter_group = "precision"
        elif any(term in reason for term in recall_terms):
            parameter_group = "recall"
        else:
            parameter_group = "balanced"
        priorities = _diagnosis_action_priorities(diagnosis)
        return priorities.index(parameter_group), reason, reason_config[1]

    return [(reason, config) for _, reason, config in sorted(
        ((group(item)[0], group(item)[1], group(item)[2]) for item in proposals),
        key=lambda ordered: (ordered[0], ordered[1]),
    )]


def parameter_changes(before: DetectorConfig, after: DetectorConfig) -> dict[str, Any]:
    before_values = config_as_dict(before)
    after_values = config_as_dict(after)
    return {
        key: {"before": before_values[key], "after": after_values[key]}
        for key in before_values
        if before_values[key] != after_values[key]
    }


def _weakest_conditions(evaluation: dict[str, Any]) -> list[str]:
    by_condition = evaluation["by_condition"]
    ranked = sorted(
        (
            condition
            for condition in CONDITIONS
            if condition in by_condition and by_condition[condition]["failure_count"] > 0
        ),
        key=lambda condition: (
            validation_objective(by_condition[condition]),
            -by_condition[condition]["failure_rate"],
            condition,
        ),
    )
    return ranked


def _proposals_for_condition(
    condition: str, config: DetectorConfig
) -> list[tuple[str, DetectorConfig]]:
    proposals: list[tuple[str, DetectorConfig]] = []

    def add(reason: str, **changes: Any) -> None:
        updated = replace(config, **changes)
        if updated != config:
            proposals.append((reason, updated))

    if condition == "color_shift":
        for hue_increase, area_increase in ((3, 20), (3, 40), (5, 20), (7, 40), (3, 60)):
            new_tolerance = min(35, config.hue_tolerance + hue_increase)
            new_area = config.min_area + area_increase
            add(
                (
                    f"widen hue tolerance by {hue_increase} and raise minimum contour area "
                    f"by {area_increase:g} to balance shifted-color recall against false detections"
                ),
                hue_tolerance=new_tolerance,
                min_area=new_area,
            )
        for amount in (5, 9, 13):
            add(
                f"widen the hue interval by {amount} to test shifted target colors",
                hue_tolerance=min(35, config.hue_tolerance + amount),
            )
        for offset in (-8, 8, -16, 16):
            add(
                f"move the hue center by {offset} to test shifted target colors",
                hue_center=(config.hue_center + offset) % 180,
            )
    elif condition == "low_brightness":
        for floor in (max(5, config.min_value - 12), max(5, config.min_value - 24), 8):
            add(f"lower the value floor to {floor} for dark samples", min_value=floor)
        for floor in (max(25, config.min_saturation - 15), max(15, config.min_saturation - 30)):
            add(f"lower the saturation floor to {floor} for dark samples", min_saturation=floor)
    elif condition == "high_brightness":
        for floor in (max(20, config.min_saturation - 15), max(10, config.min_saturation - 30)):
            add(f"lower the saturation floor to {floor} for bright samples", min_saturation=floor)
        add("lower the value floor to admit clipped bright target pixels", min_value=max(5, config.min_value - 12))
    elif condition == "blur":
        for kernel in (3, 5, 7):
            add(f"apply a {kernel} pixel pre-blur before color segmentation", pre_blur=kernel)
        for circularity in (max(0.30, config.min_circularity - 0.12), max(0.25, config.min_circularity - 0.22)):
            add(f"relax the minimum circularity to {circularity:.2f} for blurred edges", min_circularity=circularity)
        add("reduce minimum contour area to preserve small blurred circles", min_area=max(24.0, config.min_area - 18))
    elif condition == "noise":
        for kernel in (5, 7):
            add(f"use a {kernel} pixel morphology kernel to suppress noisy regions", morphology_kernel=kernel)
        for area in (config.min_area + 20, config.min_area + 40):
            add(f"raise minimum contour area to {area:g} to reject noise speckles", min_area=area)
        add("increase circularity filtering to remove irregular noise components", min_circularity=min(0.88, config.min_circularity + 0.10))
    elif condition == "distractors":
        for area in (config.min_area + 20, config.min_area + 40, config.min_area + 60):
            add(f"raise minimum contour area to {area:g} to reject small distractors", min_area=area)
        for circularity in (min(0.90, config.min_circularity + 0.10), min(0.94, config.min_circularity + 0.18)):
            add(f"require circularity of {circularity:.2f} to reject non-circle distractors", min_circularity=circularity)
        add("use stronger morphology to merge edge fragments around the target", morphology_kernel=5)
    else:
        for tolerance in (config.hue_tolerance + 4, config.hue_tolerance + 8):
            add(f"widen the hue interval to {tolerance} for normal-lighting variation", hue_tolerance=min(35, tolerance))
        add("raise the minimum contour area to reject small false detections", min_area=config.min_area + 20)
        add("tighten circularity filtering for round-target precision", min_circularity=min(0.90, config.min_circularity + 0.10))
    return proposals


def compare_same_images(
    baseline: dict[str, Any], repaired: dict[str, Any]
) -> dict[str, Any]:
    baseline_by_id = {row["sample_id"]: row for row in baseline["image_results"]}
    repaired_by_id = {row["sample_id"]: row for row in repaired["image_results"]}
    if baseline_by_id.keys() != repaired_by_id.keys():
        raise ValueError("Baseline and repaired evaluations must use exactly the same images.")

    examples: list[dict[str, Any]] = []
    for sample_id, before in baseline_by_id.items():
        after = repaired_by_id[sample_id]
        if not before["is_success"] and after["is_success"]:
            outcome = "repair_succeeded"
        elif before["is_success"] and not after["is_success"]:
            outcome = "regression"
        elif not before["is_success"] and not after["is_success"]:
            outcome = "still_failing"
        else:
            outcome = "still_successful"
        if outcome != "still_successful":
            examples.append(
                {
                    "sample_id": sample_id,
                    "condition": before["condition"],
                    "outcome": outcome,
                    "baseline": {key: before[key] for key in ("tp", "fp", "fn", "is_success", "predictions")},
                    "repaired": {key: after[key] for key in ("tp", "fp", "fn", "is_success", "predictions")},
                    "ground_truth": before["ground_truth"],
                }
            )
    counts = {
        outcome: sum(example["outcome"] == outcome for example in examples)
        for outcome in ("repair_succeeded", "regression", "still_failing")
    }
    return {
        "same_image_count": len(baseline_by_id),
        "same_sample_ids": True,
        "counts": counts,
        "examples": examples,
    }


def run_adaptive_selection(
    validation_samples: list[SyntheticSample],
    progress_callback: ProgressCallback | None = None,
    budget: int = EXPERIMENT_BUDGET,
) -> dict[str, Any]:
    """Select a configuration from validation images without accepting test data."""
    if not validation_samples:
        raise ValueError("The validation dataset must be non-empty.")
    if any(sample.split != "validation" for sample in validation_samples):
        raise ValueError("Adaptive selection accepts validation samples only.")

    run_started = perf_counter()
    budget = max(0, int(budget))
    trace: list[dict[str, Any]] = []
    if progress_callback:
        progress_callback({"phase": "baseline", "message": "Measuring the baseline on validation images."})
    baseline_validation = evaluate_dataset(validation_samples, BASELINE_CONFIG)
    best_config = BASELINE_CONFIG
    best_validation = baseline_validation
    best_objective = validation_objective(baseline_validation["metrics"])
    tested = {json.dumps(config_as_dict(BASELINE_CONFIG), sort_keys=True)}
    experiments: list[dict[str, Any]] = []
    no_improvement_streak = 0
    stopping_condition = "experiment budget reached"

    trace.append(
        {
            "step": 0,
            "type": "baseline_validation",
            "decision": "Establish baseline metrics before proposing any configuration changes.",
            "config": config_as_dict(BASELINE_CONFIG),
            "metrics": baseline_validation["metrics"],
            "by_condition": baseline_validation["by_condition"],
            "objective": round(best_objective, 6),
        }
    )

    for experiment_number in range(1, budget + 1):
        priorities = _weakest_conditions(best_validation)
        if not priorities:
            stopping_condition = "no validation failures remain to guide another experiment"
            trace.append(
                {
                    "step": len(trace),
                    "type": "stop",
                    "decision": stopping_condition,
                }
            )
            break

        selected_condition: str | None = None
        selected_reason: str | None = None
        candidate: DetectorConfig | None = None
        selected_diagnosis: dict[str, Any] | None = None
        for condition in priorities:
            diagnosis = diagnose_failure(best_validation, condition)
            proposals = order_proposals_for_diagnosis(
                _proposals_for_condition(condition, best_config), diagnosis
            )
            for reason, proposal in proposals:
                key = json.dumps(config_as_dict(proposal), sort_keys=True)
                if key not in tested:
                    selected_condition = condition
                    selected_reason = reason
                    candidate = proposal
                    selected_diagnosis = diagnosis
                    break
            if candidate is not None:
                break
        if candidate is None or selected_condition is None:
            stopping_condition = "bounded candidate space exhausted without another unique configuration"
            trace.append(
                {
                    "step": len(trace),
                    "type": "stop",
                    "decision": stopping_condition,
                    "weakest_conditions": priorities,
                }
            )
            break

        decision = {
            "step": len(trace),
            "type": "decision",
            "decision": (
                f"Investigate {CONDITION_LABELS[selected_condition]} because it has the weakest "
                "current validation objective; measured evidence diagnosed "
                f"{selected_diagnosis['diagnosis']}; {selected_reason}."
            ),
            "selected_condition": selected_condition,
            "diagnosis": selected_diagnosis,
            "current_best_config": config_as_dict(best_config),
            "current_best_validation": best_validation["metrics"],
        }
        trace.append(decision)
        if progress_callback:
            progress_callback(
                {
                    "phase": "experiment",
                    "number": experiment_number,
                    "budget": budget,
                    "condition": CONDITION_LABELS[selected_condition],
                    "message": decision["decision"],
                }
            )

        tested.add(json.dumps(config_as_dict(candidate), sort_keys=True))
        candidate_validation = evaluate_dataset(validation_samples, candidate)
        candidate_objective = validation_objective(candidate_validation["metrics"])
        objective_change = candidate_objective - best_objective
        meaningful = objective_change >= MIN_MEANINGFUL_IMPROVEMENT
        diagnosis = selected_diagnosis or diagnose_failure(best_validation, selected_condition)
        changes = parameter_changes(best_config, candidate)
        accepted = objective_change >= MIN_MEANINGFUL_IMPROVEMENT
        experiment = {
            "number": experiment_number,
            "focus_condition": selected_condition,
            "observation": {
                "metrics_before": best_validation["metrics"],
                "condition_metrics_before": best_validation["by_condition"][selected_condition],
            },
            "diagnosis": diagnosis,
            "action": {
                "reason": selected_reason,
                "parameter_changes": changes,
            },
            "decision": decision["decision"],
            "reason": selected_reason,
            "config": config_as_dict(candidate),
            "validation_metrics": candidate_validation["metrics"],
            "metrics_after": candidate_validation["metrics"],
            "validation_by_condition": candidate_validation["by_condition"],
            "objective": round(candidate_objective, 6),
            "objective_before": round(best_objective, 6),
            "objective_after": round(candidate_objective, 6),
            "objective_change_vs_best": round(objective_change, 6),
            "meaningful_improvement": meaningful,
            "acceptance_rule": (
                f"accept only when validation objective improves by at least {MIN_MEANINGFUL_IMPROVEMENT:.3f}"
            ),
            "accept_reject": "accept" if accepted else "reject",
            "rollback": "candidate already measured and rejected; current best retained" if not accepted else "none",
            "runtime_ms": candidate_validation["runtime_ms"],
        }
        if meaningful:
            best_config = candidate
            best_validation = candidate_validation
            best_objective = candidate_objective
            no_improvement_streak = 0
            experiment["selected_as_best"] = True
        else:
            no_improvement_streak += 1
            experiment["selected_as_best"] = False
        experiments.append(experiment)
        trace.append(
            {
                "step": len(trace),
                "type": "experiment",
                **experiment,
            }
        )

        if progress_callback:
            progress_callback(
                {
                    "phase": "measured",
                    "number": experiment_number,
                    "budget": budget,
                    "condition": CONDITION_LABELS[selected_condition],
                    "metrics": candidate_validation["metrics"],
                    "selected_as_best": experiment["selected_as_best"],
                    "message": (
                        "Validation improved meaningfully; this configuration becomes the new best."
                        if meaningful
                        else "No meaningful validation improvement; retain the current best configuration."
                    ),
                }
            )

        if no_improvement_streak >= MAX_NO_IMPROVEMENT_STREAK:
            stopping_condition = (
                f"stopped after {no_improvement_streak} consecutive experiments below the "
                f"{MIN_MEANINGFUL_IMPROVEMENT:.3f} objective-improvement threshold"
            )
            trace.append(
                {
                    "step": len(trace),
                    "type": "stop",
                    "decision": stopping_condition,
                }
            )
            break

    selected_on_validation = best_validation
    selected_objective = validation_objective(selected_on_validation["metrics"])
    repair_selected = (
        best_config != BASELINE_CONFIG
        and selected_objective - validation_objective(baseline_validation["metrics"])
        >= MIN_MEANINGFUL_IMPROVEMENT
    )
    validation_comparison = compare_same_images(baseline_validation, selected_on_validation)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "opencv": opencv_version_status(),
        "objective_rule": (
            "Validation objective = F1 − 0.10 × image failure rate; an update is meaningful at "
            f"an objective gain of {MIN_MEANINGFUL_IMPROVEMENT:.3f} or more. Precision and recall are reported separately."
        ),
        "baseline_config": config_as_dict(BASELINE_CONFIG),
        "selected_config": config_as_dict(best_config),
        "repair_selected": repair_selected,
        "experiment_budget": budget,
        "experiment_count": len(experiments),
        "validation_evaluation_count": 1 + len(experiments),
        "runtime_ms": round((perf_counter() - run_started) * 1000, 3),
        "validation_pipeline_runtime_ms": round(
            baseline_validation["runtime_ms"] + sum(experiment["runtime_ms"] for experiment in experiments),
            3,
        ),
        "stopping_condition": stopping_condition,
        "validation_seeds": sorted({sample.seed for sample in validation_samples}),
        "validation": {
            "baseline": baseline_validation,
            "selected": selected_on_validation,
            "changes": metric_changes(baseline_validation["metrics"], selected_on_validation["metrics"]),
            "objective_baseline": round(validation_objective(baseline_validation["metrics"]), 6),
            "objective_selected": round(selected_objective, 6),
            "sample_ids": [sample.sample_id for sample in validation_samples],
        },
        "validation_failure_analysis": validation_comparison,
        "experiments": experiments,
        "trace": trace,
    }


def run_repair_agent(
    validation_samples: list[SyntheticSample],
    test_samples: list[SyntheticSample],
    progress_callback: ProgressCallback | None = None,
    budget: int = EXPERIMENT_BUDGET,
) -> dict[str, Any]:
    """Choose on validation only, then compare both configurations on held-out test."""
    if not test_samples:
        raise ValueError("The held-out test dataset must be non-empty.")
    if any(sample.split != "test" for sample in test_samples):
        raise ValueError("Held-out evaluation accepts test samples only.")
    if {sample.sample_id for sample in validation_samples} & {sample.sample_id for sample in test_samples}:
        raise ValueError("Validation and test sample IDs must not overlap.")

    run_started = perf_counter()
    selection = run_adaptive_selection(
        validation_samples,
        progress_callback=progress_callback,
        budget=budget,
    )
    if progress_callback:
        progress_callback(
            {
                "phase": "held_out",
                "message": (
                    "Selection is complete. Evaluating baseline and selected configuration "
                    "on the same held-out test images."
                ),
            }
        )
    baseline_test = evaluate_dataset(test_samples, BASELINE_CONFIG)
    selected_test = evaluate_dataset(test_samples, DetectorConfig(**selection["selected_config"]))
    comparison = compare_same_images(baseline_test, selected_test)
    trace = [
        *selection["trace"],
        {
            "step": len(selection["trace"]),
            "type": "held_out_evaluation",
            "decision": (
                "Configuration selection is complete; compare baseline and selected config "
                "on exactly the same untouched test IDs."
            ),
            "sample_count": comparison["same_image_count"],
            "same_sample_ids": comparison["same_sample_ids"],
            "baseline_metrics": baseline_test["metrics"],
            "selected_metrics": selected_test["metrics"],
        },
    ]
    return {
        **selection,
        "seeds": dict(SPLIT_SEEDS),
        "split_sizes": dict(SPLIT_SIZES),
        "matching_rule": (
            f"Each image has one ground-truth target. Predictions are confidence-sorted; at most "
            f"one can match it, and IoU must be at least {IOU_MATCH_THRESHOLD:.2f}. "
            "Remaining predictions count as false positives. An image fails if it has any false positive or false negative."
        ),
        "trace": trace,
        "test": {
            "baseline": baseline_test,
            "selected": selected_test,
            "changes": metric_changes(baseline_test["metrics"], selected_test["metrics"]),
            "comparison": comparison,
            "sample_ids": [sample.sample_id for sample in test_samples],
            "seeds": sorted({sample.seed for sample in test_samples}),
        },
        "runtime_ms": round((perf_counter() - run_started) * 1000, 3),
        "measured_pipeline_runtime_ms": round(
            selection["validation_pipeline_runtime_ms"]
            + baseline_test["runtime_ms"]
            + selected_test["runtime_ms"],
            3,
        ),
    }


def save_run_report(report: dict[str, Any], output_dir: str | Path = "generated-artifacts") -> dict[str, str]:
    """Persist the chosen configuration and full JSON trace for later inspection."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "selected_config.json"
    trace_path = directory / "latest_run.json"
    config_path.write_text(
        json.dumps(
            {
                "selected_config": report["selected_config"],
                "repair_selected": report["repair_selected"],
                "validation_metrics": report["validation"]["selected"]["metrics"],
                "selection_rule": report["objective_rule"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    trace_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"config": str(config_path), "trace": str(trace_path)}
