"""Real-image loading and evaluation helpers without synthetic selection leakage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_pipeline import (
    BASELINE_CONFIG,
    DetectorConfig,
    GroundTruth,
    SyntheticSample,
    evaluate_dataset,
)


def _integer(value: Any, field: str, record_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{record_id}: {field} must be an integer")
    integer = int(value)
    if isinstance(value, float) and value != integer:
        raise ValueError(f"{record_id}: {field} must be an integer")
    return integer


def _manifest_filename(value: Any, record_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_id}: filename is required.")
    filename = value.strip().replace("\\", "/")
    if filename != Path(filename).name or filename in (".", ".."):
        raise ValueError(f"{record_id}: filename must name a file directly inside the image directory")
    return filename


def load_real_image_manifest(
    manifest_path: str | Path,
    image_directory: str | Path,
) -> list[SyntheticSample]:
    """Load real images and one target circle per image from a JSON manifest."""
    manifest_path = Path(manifest_path)
    image_directory = Path(image_directory)
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("Manifest is not valid UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest contains invalid JSON at line {error.lineno}, column {error.colno}") from error
    if isinstance(records, dict):
        records = records.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a non-empty JSON array or an object with an 'images' array.")

    samples: list[SyntheticSample] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest item {index} must be a JSON object.")
        filename = _manifest_filename(record.get("filename"), f"image-{index + 1}")
        record_id = filename
        if record_id in seen_ids:
            raise ValueError(f"Duplicate filename in manifest: {filename}")
        seen_ids.add(record_id)

        center = record.get("center")
        bbox = record.get("bbox")
        circle = record.get("circle") or {}
        center_x = _integer(record.get("center_x", center[0] if isinstance(center, list) else None), "center_x", record_id)
        center_y = _integer(record.get("center_y", center[1] if isinstance(center, list) else None), "center_y", record_id)
        radius = _integer(record.get("radius", circle.get("radius") if isinstance(circle, dict) else None), "radius", record_id)
        if radius <= 0:
            raise ValueError(f"{record_id}: radius must be positive.")

        if isinstance(bbox, list) and len(bbox) == 4:
            box = tuple(_integer(value, "bbox", record_id) for value in bbox)
        else:
            box = (
                center_x - radius,
                center_y - radius,
                2 * radius + 1,
                2 * radius + 1,
            )

        image_path = image_directory / filename
        root = image_directory.resolve()
        resolved_image = image_path.resolve()
        if resolved_image.parent != root:
            raise ValueError(f"{record_id}: image path escapes the upload workspace")
        if not image_path.is_file():
            raise FileNotFoundError(f"{record_id}: image file does not exist")
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            raise ValueError(f"{record_id}: unsupported image extension {image_path.suffix or '(none)'}")
        encoded = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{record_id}: unsupported or corrupt image")
        height, width = image.shape[:2]
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise ValueError(f"{record_id}: target center lies outside the {width}x{height} image")

        samples.append(
            SyntheticSample(
                sample_id=f"real-{filename}",
                split="real",
                condition=str(record.get("condition", "real_image")),
                seed=0,
                image=image,
                ground_truth=GroundTruth(center_x, center_y, radius, box),
            )
        )
    return samples


def evaluate_real_images(
    samples: list[SyntheticSample],
    config: DetectorConfig = BASELINE_CONFIG,
) -> dict[str, object]:
    """Evaluate real samples with the same detector and matching rules as synthetic data."""
    if any(sample.split != "real" for sample in samples):
        raise ValueError("Real-image evaluation requires samples loaded with split='real'.")
    return evaluate_dataset(samples, config)
