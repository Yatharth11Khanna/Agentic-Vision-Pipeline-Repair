from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import benchmark
from benchmark_reporting import save_benchmark_report, summarize_trials
from vision_pipeline import (
    BASELINE_CONFIG,
    CONDITIONS,
    GroundTruth,
    SyntheticSample,
    config_as_dict,
    generate_split,
)


def test_seed_plan_is_repeatable_and_partitions_are_independent():
    first = benchmark.trial_seed_sets(5, 1234)
    assert first == benchmark.trial_seed_sets(5, 1234)
    assert len(first) == 5
    seeds = [seed for trial in first for seed in trial.values()]
    assert len(seeds) == len(set(seeds))
    assert all(set(trial) == {"train", "validation", "random_search", "test"} for trial in first)
    with pytest.raises(ValueError):
        benchmark.trial_seed_sets(4, 1234)


def test_explicit_split_seed_reproduces_images_without_changing_default_ids():
    first = generate_split("validation", seed=881)
    repeated = generate_split("validation", seed=881)
    different = generate_split("validation", seed=882)
    default = generate_split("validation")

    assert [sample.sample_id for sample in first] == [
        sample.sample_id for sample in repeated
    ]
    assert all((left.image == right.image).all() for left, right in zip(first, repeated))
    assert first[0].sample_id.startswith("validation-881-")
    assert not (first[0].image == different[0].image).all()
    assert default[0].sample_id == "validation-000"
    with pytest.raises(ValueError):
        generate_split("validation", seed=2**32)


def test_random_search_candidates_are_seeded_unique_and_within_declared_bounds():
    first = benchmark.generate_random_candidates(99, 8)
    repeated = benchmark.generate_random_candidates(99, 8)
    assert first == repeated
    assert len(set(first)) == 8
    assert BASELINE_CONFIG not in first
    for candidate in first:
        assert benchmark._config_is_in_candidate_space(config_as_dict(candidate))


def test_trial_summary_uses_sample_std_and_t_interval_for_five_trials():
    summary = summarize_trials([0.1, 0.2, 0.3, 0.4, 0.5])
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["std"] == pytest.approx(0.158114, abs=1e-6)
    assert summary["ci95_low"] == pytest.approx(0.103707, abs=1e-6)
    assert summary["ci95_high"] == pytest.approx(0.496293, abs=1e-6)


def test_benchmark_searches_finish_before_test_generation_and_share_test_ids(
    monkeypatch, tmp_path
):
    events: list[tuple[str, str]] = []

    def representative_split(split: str, seed: int):
        events.append((split, str(seed)))
        conditions = CONDITIONS[:2] if split == "train" else CONDITIONS
        return [
            SyntheticSample(
                sample_id=f"{split}-{seed}-{index:03d}",
                split=split,
                condition=condition,
                seed=seed,
                image=np.zeros((32, 32, 3), dtype=np.uint8),
                ground_truth=GroundTruth(16, 16, 5, (11, 11, 11, 11)),
            )
            for index, condition in enumerate(conditions)
        ]

    def fake_evaluate(samples, config):
        by_condition = {}
        for condition in CONDITIONS:
            image_count = sum(sample.condition == condition for sample in samples)
            if image_count:
                by_condition[condition] = {
                    "tp": image_count,
                    "fp": 0,
                    "fn": 0,
                    "images": image_count,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "failure_count": 0,
                    "failure_rate": 0.0,
                }
        metrics = {
            "tp": len(samples),
            "fp": 0,
            "fn": 0,
            "images": len(samples),
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "failure_count": 0,
            "failure_rate": 0.0,
        }
        return {
            "split": samples[0].split,
            "metrics": metrics,
            "by_condition": by_condition,
            "runtime_ms": 1.0,
            "config": config_as_dict(config),
            "image_results": [
                {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "condition": sample.condition,
                    "predictions": [],
                    "ground_truth": {"bbox": list(sample.ground_truth.bbox)},
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "is_success": True,
                }
                for sample in samples
            ],
        }

    monkeypatch.setattr(benchmark, "generate_split", representative_split)
    monkeypatch.setattr(benchmark, "evaluate_dataset", fake_evaluate)

    def adaptive_selection(validation_samples, progress_callback=None, budget=12):
        assert validation_samples and all(sample.split == "validation" for sample in validation_samples)
        seed = validation_samples[0].sample_id
        events.append(("adaptive", seed))
        evaluation = fake_evaluate(validation_samples, BASELINE_CONFIG)
        experiment = {"config": config_as_dict(BASELINE_CONFIG), "runtime_ms": 1.0}
        return {
            "selected_config": config_as_dict(BASELINE_CONFIG),
            "experiment_count": 2,
            "experiment_budget": budget,
            "runtime_ms": 2.0,
            "experiments": [experiment, experiment],
            "validation": {"baseline": evaluation, "selected": evaluation},
        }

    def random_search(validation_samples, experiment_budget, search_seed, progress_callback=None):
        assert validation_samples and all(sample.split == "validation" for sample in validation_samples)
        assert experiment_budget == 2
        events.append(("random", str(search_seed)))
        evaluation = fake_evaluate(validation_samples, BASELINE_CONFIG)
        experiment = {"config": config_as_dict(BASELINE_CONFIG), "runtime_ms": 1.0}
        return {
            "selected_config": config_as_dict(BASELINE_CONFIG),
            "experiment_count": experiment_budget,
            "experiment_budget": experiment_budget,
            "runtime_ms": 2.0,
            "experiments": [experiment, experiment],
            "validation": {"selected": evaluation},
        }

    monkeypatch.setattr(benchmark, "run_adaptive_selection", adaptive_selection)
    monkeypatch.setattr(benchmark, "run_nonadaptive_search", random_search)
    report = benchmark.run_benchmark(trial_count=5, base_seed=7123)

    assert report["trial_count"] == 5
    assert len(report["trials"]) == 5
    for trial in report["trials"]:
        assert (
            trial["methods"]["adaptive"]["experiment_count"]
            == trial["methods"]["random_search"]["experiment_count"]
            == 2
        )
        test_ids = trial["sample_ids"]["test"]
        for method in ("baseline", "adaptive", "random_search"):
            evaluated_ids = [
                item["sample_id"]
                for item in trial["methods"][method]["test"]["image_results"]
            ]
            assert evaluated_ids == test_ids
        assert trial["test_generation_order"].startswith("generated after")
        assert all(condition in trial["methods"]["adaptive"]["test"]["by_condition"] for condition in CONDITIONS)

    for trial in report["trials"]:
        validation_seed = str(trial["seeds"]["validation"])
        random_seed = str(trial["seeds"]["random_search"])
        test_seed = str(trial["seeds"]["test"])
        adaptive_event = events.index(("adaptive", f"validation-{validation_seed}-000"))
        random_event = events.index(("random", random_seed))
        test_event = events.index(("test", test_seed))
        assert adaptive_event < random_event < test_event

    paths = save_benchmark_report(report, tmp_path)
    saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    with Path(paths["csv"]).open(newline="", encoding="utf-8") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert saved["trial_count"] == 5
    assert saved["saved_paths"]["csv"] == paths["csv"]
    assert any(row["scope"] == "validation_overall" for row in csv_rows)
    assert any(row["scope"] == "test_condition" for row in csv_rows)
    assert any(row["record_type"] == "aggregate_mean" for row in csv_rows)

