from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from real_image_evaluation import (
    evaluate_real_images,
    load_real_image_manifest,
)


def write_labeled_image(path, center_x: int, center_y: int, radius: int) -> None:
    image = np.full((120, 120, 3), (60, 80, 95), dtype=np.uint8)
    hsv = np.array([[[0, 220, 238]]], dtype=np.uint8)
    target_color = tuple(int(channel) for channel in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(image, (center_x, center_y), radius, target_color, thickness=-1)
    assert cv2.imwrite(str(path), image)


def test_real_manifest_images_use_existing_detector(tmp_path) -> None:
    write_labeled_image(tmp_path / "sample.png", 60, 60, 12)
    (tmp_path / "manifest.json").write_text(
        json.dumps([
            {
                "filename": "sample.png",
                "center_x": 60,
                "center_y": 60,
                "radius": 12,
                "condition": "indoor",
            }
        ]),
        encoding="utf-8",
    )

    samples = load_real_image_manifest(tmp_path / "manifest.json", tmp_path)
    result = evaluate_real_images(samples)

    assert result["metrics"]["images"] == 1
    assert result["metrics"]["f1"] == 1.0
    assert result["image_results"][0]["split"] == "real"


def test_real_manifest_rejects_duplicate_filename(tmp_path) -> None:
    write_labeled_image(tmp_path / "sample.png", 60, 60, 12)
    (tmp_path / "manifest.json").write_text(
        json.dumps([
            {"filename": "sample.png", "center_x": 60, "center_y": 60, "radius": 12},
            {"filename": "sample.png", "center_x": 60, "center_y": 60, "radius": 12},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate filename"):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)


def test_real_manifest_rejects_missing_image(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"filename": "missing.png", "center_x": 1, "center_y": 1, "radius": 4}]),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)


def test_real_manifest_rejects_center_outside_image(tmp_path) -> None:
    write_labeled_image(tmp_path / "sample.png", 60, 60, 12)
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"filename": "sample.png", "center_x": 121, "center_y": 60, "radius": 12}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lies outside"):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)


def test_real_manifest_rejects_traversal(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"filename": "../outside.png", "center_x": 1, "center_y": 1, "radius": 4}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="directly inside"):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)


def test_real_manifest_rejects_corrupt_image(tmp_path) -> None:
    (tmp_path / "corrupt.png").write_bytes(b"not an image")
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"filename": "corrupt.png", "center_x": 1, "center_y": 1, "radius": 4}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported or corrupt"):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)


def test_real_manifest_rejects_unsupported_extension(tmp_path) -> None:
    (tmp_path / "sample.gif").write_bytes(b"GIF89a")
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"filename": "sample.gif", "center_x": 1, "center_y": 1, "radius": 4}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported image extension"):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)


def test_real_manifest_rejects_malformed_annotation(tmp_path) -> None:
    write_labeled_image(tmp_path / "sample.png", 60, 60, 12)
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"filename": "sample.png", "center_x": 60.5, "center_y": 60, "radius": 12}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="center_x must be an integer"):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)


def test_real_manifest_rejects_invalid_json(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("[{", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_real_image_manifest(tmp_path / "manifest.json", tmp_path)
