"""Reproducible three-way benchmark for adaptive and non-adaptive vision repair."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

from benchmark_reporting import (
    METHOD_LABELS,
    aggregate_trials,
    benchmark_conclusion,
    metric_difference,
    paired_image_outcomes,
    save_benchmark_report,
)
from repair_agent import (
    EXPERIMENT_BUDGET,
    MIN_MEANINGFUL_IMPROVEMENT,
    ProgressCallback,
    compare_same_images,
    run_adaptive_selection,
    validation_objective,
)
from vision_pipeline import (
    BASELINE_CONFIG,
    CONDITIONS,
    DetectorConfig,
    SyntheticSample,
    config_as_dict,
    evaluate_dataset,
    generate_split,
    metric_changes,
    opencv_version_status,
)

MIN_TRIALS = 5
MAX_TRIALS = 20
DEFAULT_TRIAL_COUNT = 5
DEFAULT_BASE_SEED = 20261013
SEED_STRIDE = 10_000
ProgressEvent = dict[str, Any]
BenchmarkProgressCallback = Callable[[ProgressEvent], None]


def trial_seed_sets(trial_count: int, base_seed: int) -> list[dict[str, int]]:
    """Create distinct, reproducible seeds for all partitions and random search."""
    trial_count = int(trial_count)
    base_seed = int(base_seed)
    if trial_count < MIN_TRIALS or trial_count > MAX_TRIALS:
        raise ValueError(f"Trial count must be between {MIN_TRIALS} and {MAX_TRIALS}.")
    if base_seed < 0:
        raise ValueError("Base seed must be non-negative.")

    result: list[dict[str, int]] = []
    for index in range(trial_count):
        origin = base_seed + index * SEED_STRIDE
        seeds = {
            "train": origin + 11,
            "validation": origin + 23,
            "random_search": origin + 37,
            "test": origin + 53,
        }
        if max(seeds.values()) > 2**32 - 1:
            raise ValueError("Trial seed plan exceeds NumPy's supported 32-bit seed range.")
        result.append(seeds)

    dataset_seeds = [
        result[index][split]
        for index in range(trial_count)
        for split in ("train", "validation", "test")
    ]
    if len(dataset_seeds) != len(set(dataset_seeds)):
        raise ValueError("Generated train, validation, and test seeds must all be distinct.")
    all_seeds = [seed for seed_set in result for seed in seed_set.values()]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Dataset and search seeds must all be distinct.")
    return result


def candidate_space_description() -> dict[str, Any]:
    """The shared legal parameter bounds available to both searches."""
    return {
        "hue_center": {"min": 0, "max": 179, "step": 1},
        "hue_tolerance": {"min": 13, "max": 35, "step": 1},
        "min_saturation": {"min": 10, "max": 72, "step": 1},
        "min_value": {"min": 5, "max": 24, "step": 1},
        "morphology_kernel": [3, 5, 7],
        "min_area": {
            "min": 24,
            "max": int(BASELINE_CONFIG.min_area + EXPERIMENT_BUDGET * 60),
            "step": 1,
        },
        "min_circularity": {"min": 0.25, "max": 0.94, "step": 0.01},
        "pre_blur": [0, 3, 5, 7],
    }


def _random_config(rng: random.Random) -> DetectorConfig:
    area_bounds = candidate_space_description()["min_area"]
    return DetectorConfig(
        hue_center=rng.randint(0, 179),
        hue_tolerance=rng.randint(13, 35),
        min_saturation=rng.randint(10, 72),
        min_value=rng.randint(5, 24),
        morphology_kernel=rng.choice((3, 5, 7)),
        min_area=float(rng.randint(area_bounds["min"], area_bounds["max"])),
        min_circularity=round(rng.randint(25, 94) / 100, 2),
        pre_blur=rng.choice((0, 3, 5, 7)),
    )


def generate_random_candidates(search_seed: int, budget: int) -> list[DetectorConfig]:
    """Draw a fixed, seeded candidate list before measuring any of its members."""
    budget = max(0, int(budget))
    rng = random.Random(int(search_seed))
    seen = {BASELINE_CONFIG}
    candidates: list[DetectorConfig] = []
    attempt_limit = max(100, budget * 100)
    attempts = 0
    while len(candidates) < budget and attempts < attempt_limit:
        attempts += 1
        candidate = _random_config(rng)
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    if len(candidates) != budget:
        raise RuntimeError("Unable to draw the requested number of unique random candidates.")
    return candidates


def _config_is_in_candidate_space(config: dict[str, Any]) -> bool:
    for parameter, bounds in candidate_space_description().items():
        value = config[parameter]
        if isinstance(bounds, dict):
            if not bounds["min"] <= value <= bounds["max"]:
                return False
        elif value not in bounds:
            return False
    return True


def run_nonadaptive_search(
    validation_samples: list[SyntheticSample],
    experiment_budget: int,
    search_seed: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Measure a fixed seeded random candidate list; decisions never alter its order."""
    if not validation_samples:
        raise ValueError("The validation dataset must be non-empty.")
    if any(sample.split != "validation" for sample in validation_samples):
        raise ValueError("Random search accepts validation samples only.")
    budget = max(0, int(experiment_budget))
    candidates = generate_random_candidates(search_seed, budget)
    run_started = perf_counter()

    if progress_callback:
        progress_callback(
            {
                "phase": "random_baseline",
                "message": "Measuring the baseline before the fixed random candidate list.",
            }
        )
    baseline_validation = evaluate_dataset(validation_samples, BASELINE_CONFIG)
    best_config = BASELINE_CONFIG
    best_validation = baseline_validation
    best_objective = validation_objective(baseline_validation["metrics"])
    experiments: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "type": "baseline_validation",
            "decision": "Measure the shared baseline objective before scoring random candidates.",
            "config": config_as_dict(BASELINE_CONFIG),
            "metrics": baseline_validation["metrics"],
            "objective": round(best_objective, 6),
            "search_seed": int(search_seed),
        }
    ]

    for experiment_number, candidate in enumerate(candidates, start=1):
        if progress_callback:
            progress_callback(
                {
                    "phase": "random_candidate",
                    "number": experiment_number,
                    "budget": budget,
                    "message": (
                        f"Scoring fixed random candidate {experiment_number} of {budget} "
                        "on validation images."
                    ),
                }
            )
        candidate_validation = evaluate_dataset(validation_samples, candidate)
        candidate_objective = validation_objective(candidate_validation["metrics"])
        objective_change = candidate_objective - best_objective
        meaningful = objective_change >= MIN_MEANINGFUL_IMPROVEMENT
        experiment = {
            "number": experiment_number,
            "config": config_as_dict(candidate),
            "validation_metrics": candidate_validation["metrics"],
            "validation_by_condition": candidate_validation["by_condition"],
            "objective": round(candidate_objective, 6),
            "objective_change_vs_best": round(objective_change, 6),
            "meaningful_improvement": meaningful,
            "selected_as_best": meaningful,
            "runtime_ms": candidate_validation["runtime_ms"],
        }
        if meaningful:
            best_config = candidate
            best_validation = candidate_validation
            best_objective = candidate_objective
        experiments.append(experiment)
        trace.append(
            {
                "step": experiment_number,
                "type": "random_candidate_evaluation",
                "decision": (
                    "Keep this configuration as the best validation result."
                    if meaningful
                    else "Retain the current best; this candidate did not clear the shared improvement threshold."
                ),
                **experiment,
            }
        )

    selected_objective = validation_objective(best_validation["metrics"])
    repair_selected = (
        best_config != BASELINE_CONFIG
        and selected_objective - validation_objective(baseline_validation["metrics"])
        >= MIN_MEANINGFUL_IMPROVEMENT
    )
    return {
        "method": "budget_matched_random_search",
        "search_seed": int(search_seed),
        "candidate_generation_rule": (
            "All candidate configurations are generated from the recorded shared parameter bounds "
            "using Python's seeded random.Random before any candidate is evaluated. The list is "
            "independent of measured results."
        ),
        "candidate_space": candidate_space_description(),
        "selected_config": config_as_dict(best_config),
        "repair_selected": repair_selected,
        "experiment_budget": budget,
        "experiment_count": len(experiments),
        "validation_evaluation_count": 1 + len(experiments),
        "runtime_ms": round((perf_counter() - run_started) * 1000, 3),
        "validation_pipeline_runtime_ms": round(
            baseline_validation["runtime_ms"]
            + sum(experiment["runtime_ms"] for experiment in experiments),
            3,
        ),
        "stopping_condition": "fixed non-adaptive candidate budget exhausted",
        "validation": {
            "baseline": baseline_validation,
            "selected": best_validation,
            "changes": metric_changes(
                baseline_validation["metrics"], best_validation["metrics"]
            ),
            "objective_baseline": round(
                validation_objective(baseline_validation["metrics"]), 6
            ),
            "objective_selected": round(selected_objective, 6),
            "sample_ids": [sample.sample_id for sample in validation_samples],
        },
        "experiments": experiments,
        "trace": trace,
    }


def _method_record(
    method: str,
    config: dict[str, Any],
    validation: dict[str, Any],
    test: dict[str, Any],
    experiment_count: int,
    experiment_budget: int,
    selection_runtime_ms: float,
    candidate_pipeline_runtime_ms: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "config": config,
        "validation": validation,
        "test": test,
        "experiment_count": int(experiment_count),
        "experiment_budget": int(experiment_budget),
        "selection_runtime_ms": round(float(selection_runtime_ms), 3),
        "candidate_pipeline_runtime_ms": round(
            float(candidate_pipeline_runtime_ms), 3
        ),
        "test_evaluation_runtime_ms": test["runtime_ms"],
        "total_method_runtime_ms": round(
            float(selection_runtime_ms) + float(test["runtime_ms"]), 3
        ),
    }


def _trial_progress(
    callback: BenchmarkProgressCallback | None,
    trial_index: int,
    trial_count: int,
    progress_in_trial: float,
    message: str,
) -> None:
    if callback:
        callback(
            {
                "trial": trial_index + 1,
                "trial_count": trial_count,
                "progress": min(
                    1.0, max(0.0, (trial_index + progress_in_trial) / trial_count)
                ),
                "message": message,
            }
        )


def run_benchmark(
    trial_count: int = DEFAULT_TRIAL_COUNT,
    base_seed: int = DEFAULT_BASE_SEED,
    progress_callback: BenchmarkProgressCallback | None = None,
) -> dict[str, Any]:
    """Run adaptive and fixed random searches, then compare all methods on held-out trials."""
    if not opencv_version_status()["is_opencv5"]:
        raise RuntimeError(opencv_version_status()["message"])
    seeds_by_trial = trial_seed_sets(trial_count, base_seed)
    benchmark_started = perf_counter()
    trials: list[dict[str, Any]] = []

    for trial_index, seeds in enumerate(seeds_by_trial):
        trial_id = f"trial-{trial_index + 1:02d}"
        _trial_progress(
            progress_callback,
            trial_index,
            trial_count,
            0.0,
            f"{trial_id}: generating training and validation samples.",
        )
        train_samples = generate_split("train", seeds["train"])
        validation_samples = generate_split("validation", seeds["validation"])

        def adaptive_progress(event: ProgressEvent) -> None:
            number = int(event.get("number", 0))
            budget = max(1, int(event.get("budget", EXPERIMENT_BUDGET)))
            fraction = min(1.0, number / budget)
            _trial_progress(
                progress_callback,
                trial_index,
                trial_count,
                0.05 + 0.35 * fraction,
                f"{trial_id}: adaptive search — {event.get('message', event.get('phase', 'running'))}",
            )

        adaptive = run_adaptive_selection(
            validation_samples,
            progress_callback=adaptive_progress,
            budget=EXPERIMENT_BUDGET,
        )
        adaptive_configs = [adaptive["selected_config"]] + [
            experiment["config"] for experiment in adaptive["experiments"]
        ]
        if not all(_config_is_in_candidate_space(config) for config in adaptive_configs):
            raise RuntimeError(
                "Adaptive repair proposed a configuration outside the declared shared search space."
            )
        random_budget = adaptive["experiment_count"]

        def random_progress(event: ProgressEvent) -> None:
            number = int(event.get("number", 0))
            budget = max(1, int(event.get("budget", random_budget)))
            fraction = min(1.0, number / budget)
            _trial_progress(
                progress_callback,
                trial_index,
                trial_count,
                0.42 + 0.35 * fraction,
                f"{trial_id}: fixed random search — {event.get('message', event.get('phase', 'running'))}",
            )

        random_search = run_nonadaptive_search(
            validation_samples,
            experiment_budget=random_budget,
            search_seed=seeds["random_search"],
            progress_callback=random_progress,
        )
        random_configs = [random_search["selected_config"]] + [
            experiment["config"] for experiment in random_search["experiments"]
        ]
        if not all(_config_is_in_candidate_space(config) for config in random_configs):
            raise RuntimeError(
                "Random search evaluated a configuration outside the declared shared search space."
            )
        if random_search["experiment_count"] != adaptive["experiment_count"]:
            raise RuntimeError(
                f"{trial_id} violated matched candidate budget: adaptive "
                f"{adaptive['experiment_count']} vs random {random_search['experiment_count']}."
            )

        # The held-out split is generated only after both validation searches finish.
        _trial_progress(
            progress_callback,
            trial_index,
            trial_count,
            0.80,
            f"{trial_id}: searches complete; generating held-out test images.",
        )
        test_samples = generate_split("test", seeds["test"])
        baseline_test = evaluate_dataset(test_samples, BASELINE_CONFIG)
        adaptive_test = evaluate_dataset(
            test_samples, DetectorConfig(**adaptive["selected_config"])
        )
        random_test = evaluate_dataset(
            test_samples, DetectorConfig(**random_search["selected_config"])
        )
        expected_test_ids = [sample.sample_id for sample in test_samples]
        for method_name, evaluation in (
            ("baseline", baseline_test),
            ("adaptive", adaptive_test),
            ("random_search", random_test),
        ):
            actual_ids = [row["sample_id"] for row in evaluation["image_results"]]
            if actual_ids != expected_test_ids:
                raise RuntimeError(
                    f"{method_name} did not evaluate the exact held-out sample sequence."
                )

        baseline_validation = adaptive["validation"]["baseline"]
        methods = {
            "baseline": _method_record(
                "baseline",
                config_as_dict(BASELINE_CONFIG),
                baseline_validation,
                baseline_test,
                experiment_count=0,
                experiment_budget=0,
                selection_runtime_ms=0.0,
                candidate_pipeline_runtime_ms=0.0,
            ),
            "adaptive": _method_record(
                "adaptive",
                adaptive["selected_config"],
                adaptive["validation"]["selected"],
                adaptive_test,
                experiment_count=adaptive["experiment_count"],
                experiment_budget=adaptive["experiment_budget"],
                selection_runtime_ms=adaptive["runtime_ms"],
                candidate_pipeline_runtime_ms=sum(
                    experiment["runtime_ms"] for experiment in adaptive["experiments"]
                ),
            ),
            "random_search": _method_record(
                "random_search",
                random_search["selected_config"],
                random_search["validation"]["selected"],
                random_test,
                experiment_count=random_search["experiment_count"],
                experiment_budget=random_search["experiment_budget"],
                selection_runtime_ms=random_search["runtime_ms"],
                candidate_pipeline_runtime_ms=sum(
                    experiment["runtime_ms"] for experiment in random_search["experiments"]
                ),
            ),
        }
        paired_differences = {
            "adaptive_vs_baseline": metric_difference(
                adaptive_test["metrics"], baseline_test["metrics"]
            ),
            "random_search_vs_baseline": metric_difference(
                random_test["metrics"], baseline_test["metrics"]
            ),
            "adaptive_vs_random_search": metric_difference(
                adaptive_test["metrics"], random_test["metrics"]
            ),
        }
        comparisons = {
            "adaptive_vs_baseline": compare_same_images(baseline_test, adaptive_test),
            "random_search_vs_baseline": compare_same_images(
                baseline_test, random_test
            ),
            "adaptive_vs_random_search": paired_image_outcomes(
                adaptive_test, random_test, "adaptive", "random_search"
            ),
        }
        trials.append(
            {
                "trial_id": trial_id,
                "seeds": dict(seeds),
                "split_sizes": {
                    "train": len(train_samples),
                    "validation": len(validation_samples),
                    "test": len(test_samples),
                },
                "sample_ids": {
                    "train": [sample.sample_id for sample in train_samples],
                    "validation": [sample.sample_id for sample in validation_samples],
                    "test": expected_test_ids,
                },
                "test_generation_order": (
                    "generated after adaptive and random validation searches completed"
                ),
                "methods": methods,
                "paired_differences": paired_differences,
                "comparisons": comparisons,
                "selection": {
                    "adaptive": adaptive,
                    "random_search": random_search,
                },
                "validation_gain_over_baseline": {
                    "adaptive_f1": round(
                        adaptive["validation"]["selected"]["metrics"]["f1"]
                        - baseline_validation["metrics"]["f1"],
                        6,
                    ),
                    "random_search_f1": round(
                        random_search["validation"]["selected"]["metrics"]["f1"]
                        - baseline_validation["metrics"]["f1"],
                        6,
                    ),
                },
            }
        )
        _trial_progress(
            progress_callback,
            trial_index,
            trial_count,
            1.0,
            f"{trial_id}: all methods evaluated on the same held-out images.",
        )

    aggregates = aggregate_trials(trials)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "opencv": opencv_version_status(),
        "trial_count": len(trials),
        "base_seed": int(base_seed),
        "seeds": seeds_by_trial,
        "candidate_space": candidate_space_description(),
        "experiment_budget_rule": (
            "For each trial, fixed random search evaluates exactly the number of unique candidates "
            "actually evaluated by adaptive repair, including when adaptive repair stops early."
        ),
        "selection_protocol": (
            "Each search receives only that trial's validation samples. Test samples are generated "
            "after both searches select configurations, then the baseline and both selected configs "
            "are evaluated on exactly the same test IDs."
        ),
        "training_split_note": (
            "A separately seeded training split is generated and preserved for every trial. "
            "The detector is a classical parameterized OpenCV pipeline and does not fit a learned model."
        ),
        "matching_rule": (
            f"One target per image; confidence-sorted predictions; IoU >= 0.30 for a match; "
            "remaining detections are false positives; image failure means any false positive or missed target."
        ),
        "objective_rule": (
            "Both searches use validation objective F1 - 0.10 × image failure rate. "
            f"A selected improvement must be at least {MIN_MEANINGFUL_IMPROVEMENT:.3f}."
        ),
        "nonadaptive_rule": (
            "Candidate configurations are sampled uniformly and independently by parameter from the "
            "recorded shared bounds with a per-trial seeded Python random.Random instance. The complete "
            "candidate list is fixed before any candidate is scored."
        ),
        "runtime_ms": round((perf_counter() - benchmark_started) * 1000, 3),
        "aggregates": aggregates,
        "conclusion": benchmark_conclusion(aggregates, trials),
        "trials": trials,
    }
    return report


