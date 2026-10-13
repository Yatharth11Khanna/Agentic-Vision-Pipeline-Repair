"""Synthetic target data and a measurable OpenCV circle-detection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any

import cv2
import numpy as np

IMAGE_SIZE = 192
CONDITIONS = (
    "normal",
    "low_brightness",
    "high_brightness",
    "blur",
    "color_shift",
    "noise",
    "distractors",
)
CONDITION_LABELS = {
    "normal": "Normal lighting",
    "low_brightness": "Low brightness",
    "high_brightness": "High brightness",
    "blur": "Blur",
    "color_shift": "Color shift",
    "noise": "Noise",
    "distractors": "Distractors",
}
SPLIT_SEEDS = {"train": 1103, "validation": 2207, "test": 3301}
SPLIT_SIZES = {"train": 84, "validation": 42, "test": 49}
IOU_MATCH_THRESHOLD = 0.30


@dataclass(frozen=True)
class DetectorConfig:
    """Parameters exposed to the bounded validation-only repair search."""

    hue_center: int = 0
    hue_tolerance: int = 13
    min_saturation: int = 72
    min_value: int = 24
    morphology_kernel: int = 3
    min_area: float = 55.0
    min_circularity: float = 0.48
    pre_blur: int = 0


@dataclass(frozen=True)
class GroundTruth:
    center_x: int
    center_y: int
    radius: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    split: str
    condition: str
    seed: int
    image: np.ndarray
    ground_truth: GroundTruth


BASELINE_CONFIG = DetectorConfig()


def opencv_version_status() -> dict[str, Any]:
    """Return an explicit compliance status; OpenCV 4 is never treated as 5."""
    cv_version = getattr(cv2, "__version__", "unknown")
    try:
        package_version = version("opencv-python-headless")
    except PackageNotFoundError:
        package_version = "package metadata unavailable"
    is_v5 = cv_version.split(".", maxsplit=1)[0] == "5" and package_version.startswith("5.")
    return {
        "opencv_version": cv_version,
        "package_version": package_version,
        "is_opencv5": is_v5,
        "message": (
            f"OpenCV {cv_version} ({package_version}) is installed."
            if is_v5
            else (
                "OpenCV 5 is required. "
                f"Detected cv2 {cv_version} and opencv-python-headless {package_version}; "
                "the app will not silently use or claim OpenCV 4."
            )
        ),
    }


def require_opencv5() -> None:
    status = opencv_version_status()
    if not status["is_opencv5"]:
        raise RuntimeError(status["message"])


def _target_color() -> tuple[int, int, int]:
    hsv_pixel = np.array([[[0, 220, 238]]], dtype=np.uint8)
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr_pixel)


def _background(rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(35, 150, size=3, dtype=np.uint8)
    image = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[:] = base
    # A modest smooth gradient makes backgrounds less uniform without obscuring targets.
    horizontal = np.linspace(-12, 12, IMAGE_SIZE, dtype=np.float32)[None, :, None]
    vertical = np.linspace(8, -8, IMAGE_SIZE, dtype=np.float32)[:, None, None]
    noise = rng.normal(0, 4, size=(IMAGE_SIZE, IMAGE_SIZE, 1))
    image = np.clip(image.astype(np.float32) + horizontal + vertical + noise, 0, 255)
    return image.astype(np.uint8)


def _draw_distractors(
    image: np.ndarray,
    rng: np.random.Generator,
    target_x: int,
    target_y: int,
    target_radius: int,
    count: int,
) -> None:
    palette = [(218, 170, 48), (55, 190, 205), (80, 205, 90), (160, 90, 205)]
    for index in range(count):
        x = int(rng.integers(12, IMAGE_SIZE - 12))
        y = int(rng.integers(12, IMAGE_SIZE - 12))
        radius = int(rng.integers(3, 9))
        if (x - target_x) ** 2 + (y - target_y) ** 2 < (target_radius + radius + 8) ** 2:
            continue
        color = palette[int(rng.integers(0, len(palette)))]
        # Some distractor circles deliberately share the target's hue but are smaller.
        if index == 0 and count >= 5:
            color = _target_color()
            radius = int(rng.integers(3, 6))
        if index % 3 == 0:
            cv2.circle(image, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        elif index % 3 == 1:
            cv2.rectangle(image, (x - radius, y - radius), (x + radius, y + radius), color, -1)
        else:
            points = np.array(
                [[x, y - radius], [x - radius, y + radius], [x + radius, y + radius]],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [points], color)


def _apply_condition(
    image: np.ndarray, condition: str, rng: np.random.Generator
) -> np.ndarray:
    output = image
    if condition == "low_brightness":
        output = np.clip(output.astype(np.float32) * float(rng.uniform(0.28, 0.42)), 0, 255)
    elif condition == "high_brightness":
        output = np.clip(output.astype(np.float32) * float(rng.uniform(1.35, 1.62)), 0, 255)
    elif condition == "blur":
        kernel = int(rng.choice([5, 7]))
        output = cv2.GaussianBlur(output, (kernel, kernel), 1.1)
    elif condition == "color_shift":
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        shift = int(rng.choice([-19, -15, 15, 19]))
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 180
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif condition == "noise":
        noise = rng.normal(0, 24, size=output.shape)
        output = np.clip(output.astype(np.float32) + noise, 0, 255)
    return output.astype(np.uint8)


def generate_split(split: str) -> list[SyntheticSample]:
    """Generate a deterministic split with independent, documented split seeds."""
    require_opencv5()
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(SPLIT_SEEDS)}.")

    seed = SPLIT_SEEDS[split]
    count = SPLIT_SIZES[split]
    rng = np.random.default_rng(seed)
    conditions = np.resize(np.asarray(CONDITIONS, dtype=object), count)
    rng.shuffle(conditions)
    samples: list[SyntheticSample] = []

    for index, condition_value in enumerate(conditions):
        condition = str(condition_value)
        image = _background(rng)
        radius = int(rng.integers(7, 17))
        center_x = int(rng.integers(radius + 14, IMAGE_SIZE - radius - 14))
        center_y = int(rng.integers(radius + 14, IMAGE_SIZE - radius - 14))

        shape_count = int(rng.integers(1, 4))
        if condition == "distractors":
            shape_count += 4
        _draw_distractors(image, rng, center_x, center_y, radius, shape_count)
        cv2.circle(
            image,
            (center_x, center_y),
            radius,
            _target_color(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        image = _apply_condition(image, condition, rng)
        bbox = (center_x - radius, center_y - radius, 2 * radius + 1, 2 * radius + 1)
        samples.append(
            SyntheticSample(
                sample_id=f"{split}-{index:03d}",
                split=split,
                condition=condition,
                seed=seed,
                image=image,
                ground_truth=GroundTruth(center_x, center_y, radius, bbox),
            )
        )
    return samples


def make_hue_mask(hsv: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Threshold hue with wraparound, plus configurable saturation/value floors."""
    center = config.hue_center % 180
    tolerance = min(max(config.hue_tolerance, 1), 89)
    low, high = center - tolerance, center + tolerance
    lower_sv = (max(0, config.min_saturation), max(0, config.min_value))
    if low < 0:
        first = cv2.inRange(
            hsv,
            np.array([0, *lower_sv], dtype=np.uint8),
            np.array([high, 255, 255], dtype=np.uint8),
        )
        second = cv2.inRange(
            hsv,
            np.array([180 + low, *lower_sv], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        return cv2.bitwise_or(first, second)
    if high >= 180:
        first = cv2.inRange(
            hsv,
            np.array([low, *lower_sv], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        second = cv2.inRange(
            hsv,
            np.array([0, *lower_sv], dtype=np.uint8),
            np.array([high - 180, 255, 255], dtype=np.uint8),
        )
        return cv2.bitwise_or(first, second)
    return cv2.inRange(
        hsv,
        np.array([low, *lower_sv], dtype=np.uint8),
        np.array([high, 255, 255], dtype=np.uint8),
    )


def intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _box_from_circle(x: float, y: float, radius: float) -> tuple[int, int, int, int]:
    left, top = int(round(x - radius)), int(round(y - radius))
    diameter = max(1, int(round(radius * 2 + 1)))
    return (left, top, diameter, diameter)


def _non_maximum_suppression(
    candidates: list[dict[str, Any]], overlap_threshold: float = 0.35
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if all(
            intersection_over_union(candidate["bbox"], existing["bbox"]) < overlap_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def detect_circles(
    image: np.ndarray, config: DetectorConfig = BASELINE_CONFIG
) -> list[dict[str, Any]]:
    """Run HSV, morphology, contours and Hough circle proposals with NMS."""
    require_opencv5()
    working = image
    if config.pre_blur >= 3:
        kernel = config.pre_blur if config.pre_blur % 2 else config.pre_blur + 1
        working = cv2.GaussianBlur(working, (kernel, kernel), 0)

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    mask = make_hue_mask(hsv, config)
    kernel_size = max(1, int(config.morphology_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    candidates: list[dict[str, Any]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_area or area > IMAGE_SIZE * IMAGE_SIZE * 0.25:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(4 * np.pi * area / (perimeter * perimeter))
        if circularity < config.min_circularity:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if radius < 3:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect = min(width, height) / max(width, height)
        if aspect < 0.55:
            continue
        solidity = min(1.0, area / (np.pi * radius * radius))
        confidence = float(min(0.99, 0.72 * circularity + 0.28 * solidity))
        candidates.append(
            {
                "bbox": _box_from_circle(center_x, center_y, radius),
                "center": [round(float(center_x), 2), round(float(center_y), 2)],
                "radius": round(float(radius), 2),
                "confidence": round(confidence, 4),
                "source": "contour",
            }
        )

    # Hough proposals provide an independent circle cue; NMS prevents duplicate credit.
    hough_input = cv2.medianBlur(mask, 5)
    hough = cv2.HoughCircles(
        hough_input,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=16,
        param1=80,
        param2=18,
        minRadius=4,
        maxRadius=24,
    )
    if hough is not None:
        for center_x, center_y, radius in np.round(hough[0]).astype(int):
            candidates.append(
                {
                    "bbox": _box_from_circle(float(center_x), float(center_y), float(radius)),
                    "center": [int(center_x), int(center_y)],
                    "radius": int(radius),
                    "confidence": 0.58,
                    "source": "hough",
                }
            )
    return _non_maximum_suppression(candidates)


def _image_metrics(
    predictions: list[dict[str, Any]], ground_truth: GroundTruth
) -> dict[str, Any]:
    gt_box = ground_truth.bbox
    matched_index: int | None = None
    matched_iou = 0.0
    for index, prediction in enumerate(predictions):
        overlap = intersection_over_union(prediction["bbox"], gt_box)
        if overlap > matched_iou:
            matched_iou, matched_index = overlap, index
    true_positive = int(matched_index is not None and matched_iou >= IOU_MATCH_THRESHOLD)
    false_positive = len(predictions) - true_positive
    false_negative = 1 - true_positive
    matched = bool(true_positive)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "matched_iou": round(matched_iou, 4) if matched else None,
        "is_success": matched and false_positive == 0,
    }


def summarize_results(image_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(row["tp"] for row in image_results)
    total_fp = sum(row["fp"] for row in image_results)
    total_fn = sum(row["fn"] for row in image_results)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = sum(not row["is_success"] for row in image_results)
    overlaps = [row["matched_iou"] for row in image_results if row["matched_iou"] is not None]
    return {
        "images": len(image_results),
        "targets": len(image_results),
        "predictions": total_tp + total_fp,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "failure_count": errors,
        "failure_rate": round(errors / len(image_results), 6) if image_results else 0.0,
        "mean_matched_iou": round(float(np.mean(overlaps)), 6) if overlaps else 0.0,
    }


def evaluate_dataset(
    samples: list[SyntheticSample], config: DetectorConfig
) -> dict[str, Any]:
    """Evaluate every sample once and preserve per-image predictions and errors."""
    started = perf_counter()
    image_results: list[dict[str, Any]] = []
    for sample in samples:
        predictions = detect_circles(sample.image, config)
        counts = _image_metrics(predictions, sample.ground_truth)
        image_results.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "condition": sample.condition,
                "predictions": predictions,
                "ground_truth": {
                    "bbox": list(sample.ground_truth.bbox),
                    "center": [sample.ground_truth.center_x, sample.ground_truth.center_y],
                    "radius": sample.ground_truth.radius,
                },
                **counts,
            }
        )
    elapsed_ms = (perf_counter() - started) * 1000

    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for row in image_results if row["condition"] == condition]
        if rows:
            by_condition[condition] = summarize_results(rows)
    return {
        "split": samples[0].split if samples else "empty",
        "metrics": summarize_results(image_results),
        "by_condition": by_condition,
        "runtime_ms": round(elapsed_ms, 3),
        "config": asdict(config),
        "image_results": image_results,
    }


def metric_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "failure_rate"):
        baseline_value = float(before.get(key, 0.0))
        new_value = float(after.get(key, 0.0))
        absolute = new_value - baseline_value
        relative = absolute / abs(baseline_value) if baseline_value else None
        changes[key] = {
            "absolute": round(absolute, 6),
            "relative": round(relative, 6) if relative is not None else None,
        }
    return changes


def config_as_dict(config: DetectorConfig) -> dict[str, Any]:
    return asdict(config)
