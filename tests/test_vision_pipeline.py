import cv2
import numpy as np

from repair_agent import run_repair_agent
from vision_pipeline import (
    BASELINE_CONFIG,
    CONDITIONS,
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
