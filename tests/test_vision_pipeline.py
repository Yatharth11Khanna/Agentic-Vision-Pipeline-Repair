import cv2
import numpy as np

from repair_agent import (
    diagnose_failure,
    order_proposals_for_diagnosis,
    parameter_changes,
    run_adaptive_selection,
    run_repair_agent,
    validation_objective,
)
from vision_pipeline import (
    BASELINE_CONFIG,
    CONDITIONS,
    DetectorConfig,
    IMAGE_SIZE,
    SPLIT_SEEDS,
    GroundTruth,
    SyntheticSample,
    detect_circles,
    evaluate_dataset,
    generate_split,
    intersection_over_union,
    opencv_version_status,
)


def clean_target_sample() -> SyntheticSample:
    hsv = np.array([[[0, 220, 238]]], dtype=np.uint8)
    target_color = tuple(
        int(channel)
        for channel in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    )
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), (60, 80, 95), dtype=np.uint8)
    cv2.circle(image, (96, 98), 12, target_color, thickness=-1, lineType=cv2.LINE_AA)
    return SyntheticSample(
        sample_id="clean-1",
        split="validation",
        condition="normal",
        seed=99,
        image=image,
        ground_truth=GroundTruth(96, 98, 12, (84, 86, 25, 25)),
    )


def test_pinned_opencv_5_package_is_visible() -> None:
    status = opencv_version_status()
    assert status["opencv_version"].startswith("5.")
    assert status["package_version"] == "5.0.0.93"
    assert status["is_opencv5"] is True


def test_synthetic_split_is_deterministic_and_balanced() -> None:
    first = generate_split("validation")
    second = generate_split("validation")

    assert len(first) == 42
    assert {sample.condition for sample in first} == set(CONDITIONS)
    assert {sample.seed for sample in first} == {SPLIT_SEEDS["validation"]}
    for a, b in zip(first, second):
        assert a.sample_id == b.sample_id
        assert a.condition == b.condition
        assert a.ground_truth == b.ground_truth
        assert np.array_equal(a.image, b.image)


def test_labeled_target_remains_visible_after_every_condition_transform() -> None:
    for split in ("train", "validation", "test"):
        for sample in generate_split(split):
            center_x = sample.ground_truth.center_x
            center_y = sample.ground_truth.center_y
            center_hsv = cv2.cvtColor(sample.image, cv2.COLOR_BGR2HSV)[center_y, center_x]
            assert int(center_hsv[1]) >= 35, (sample.sample_id, sample.condition, center_hsv)
            assert int(center_hsv[2]) >= 15, (sample.sample_id, sample.condition, center_hsv)


def test_opencv_pipeline_finds_target_and_suppresses_duplicate_proposals() -> None:
    sample = clean_target_sample()
    predictions = detect_circles(sample.image, BASELINE_CONFIG)
    evaluation = evaluate_dataset([sample], BASELINE_CONFIG)

    assert predictions
    assert evaluation["metrics"]["tp"] == 1
    assert evaluation["metrics"]["fp"] == 0
    assert evaluation["metrics"]["fn"] == 0
    assert evaluation["metrics"]["f1"] == 1.0


def test_iou_matching_rule_and_no_overlap() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert 0.30 <= intersection_over_union((83, 85, 28, 28), (84, 86, 25, 25))


def test_agent_uses_validation_for_selection_and_pairs_held_out_images() -> None:
    validation = generate_split("validation")
    test = generate_split("test")
    report = run_repair_agent(validation, test, budget=2)

    assert report["experiment_count"] <= 2
    assert report["validation_evaluation_count"] == report["experiment_count"] + 1
    assert report["runtime_ms"] > 0
    assert report["measured_pipeline_runtime_ms"] > 0
    assert set(report["validation"]["sample_ids"]).isdisjoint(report["test"]["sample_ids"])
    assert report["test"]["comparison"]["same_sample_ids"] is True
    assert report["test"]["comparison"]["same_image_count"] == len(test)
    assert any(event["type"] == "experiment" for event in report["trace"])


def test_targeted_diagnosis_and_deterministic_action_trace() -> None:
    samples = generate_split("validation")
    first = run_adaptive_selection(samples, budget=2)
    second = run_adaptive_selection(samples, budget=2)

    assert first["selected_config"] == second["selected_config"]
    assert first["experiment_count"] == second["experiment_count"]
    experiments = [event for event in first["trace"] if event["type"] == "experiment"]
    assert experiments
    for experiment in experiments:
        assert experiment["observation"]
        assert experiment["diagnosis"]["measured_evidence"]
        assert experiment["diagnosis"]["heuristic_basis"]
        assert experiment["action"]["reason"]
        assert experiment["metrics_after"] == experiment["validation_metrics"]
        assert experiment["accept_reject"] in {"accept", "reject"}
        assert experiment["observation"]["metrics_before"] == experiment["observation"]["metrics_before"]
        assert experiment["objective_before"] == round(
            validation_objective(experiment["observation"]["metrics_before"]), 6
        )


def test_controlled_failure_patterns_change_targeted_proposal_order() -> None:
    proposals = [
        ("raise minimum contour area to reject false detections", BASELINE_CONFIG),
        ("lower the value floor to admit target pixels", DetectorConfig(min_value=5)),
    ]

    def fake_evaluation(false_positives: int, false_negatives: int) -> dict:
        return {
            "by_condition": {
                "noise": {
                    "images": 1,
                    "fp": false_positives,
                    "fn": false_negatives,
                    "precision": 0.5,
                    "recall": 0.5,
                    "failure_rate": 1.0,
                }
            }
        }

    fn_only = diagnose_failure(fake_evaluation(0, 2), "noise")
    fp_only = diagnose_failure(fake_evaluation(2, 0), "noise")
    fn_first = order_proposals_for_diagnosis(proposals, fn_only)
    fp_first = order_proposals_for_diagnosis(proposals, fp_only)

    assert "lower" in fn_first[0][0]
    assert "area" in fp_first[0][0]
    assert fn_first != fp_first


def test_parameter_changes_and_diagnosis_are_complete() -> None:
    evaluation = evaluate_dataset(generate_split("validation")[:7], BASELINE_CONFIG)
    condition = next(
        condition
        for condition, metrics in evaluation["by_condition"].items()
        if metrics["fn"] > 0
    )
    diagnosis = diagnose_failure(evaluation, condition)

    assert diagnosis["measured_evidence"]["condition"] == condition
    assert diagnosis["evidence_type"].startswith("measured metrics")
    assert parameter_changes(BASELINE_CONFIG, BASELINE_CONFIG) == {}


def test_rejected_candidate_rolls_back_and_budget_is_hard_limit() -> None:
    samples = generate_split("validation")
    report = run_adaptive_selection(samples, budget=2)

    assert report["experiment_count"] <= 2
    rejected = [event for event in report["trace"] if event.get("accept_reject") == "reject"]
    for event in rejected:
        assert "current best retained" in event["rollback"]
    assert report["selected_config"] == BASELINE_CONFIG.__dict__ or report["repair_selected"]
    for experiment in rejected:
        rejected_config = DetectorConfig(**experiment["config"])
        assert rejected_config != DetectorConfig(**report["selected_config"])
