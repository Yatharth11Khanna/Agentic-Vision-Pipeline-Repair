"""Streamlit interface for the OpenCV pipeline repair experiment."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import cv2

    from benchmark_ui import load_saved_benchmark, render_benchmark_page
    from real_image_evaluation import evaluate_real_images
    from repair_agent import run_repair_agent, save_run_report
    from vision_pipeline import (
        BASELINE_CONFIG,
        CONDITION_LABELS,
        CONDITIONS,
        SPLIT_SEEDS,
        SPLIT_SIZES,
        DetectorConfig,
        SyntheticSample,
        evaluate_dataset,
        opencv_version_status,
    )
except Exception as import_error:  # Keep an unavailable/incompatible OpenCV install explicit in the UI.
    st.set_page_config(page_title="Agentic Vision Pipeline Repair", layout="wide")
    st.title("Agentic Vision Pipeline Repair")
    st.error(
        "The vision pipeline could not start. OpenCV 5.0.0.93 and its compatible runtime are required."
    )
    st.code(f"{type(import_error).__name__}: {import_error}", language="text")
    st.stop()

st.set_page_config(page_title="Agentic Vision Pipeline Repair", layout="wide")


@st.cache_data(show_spinner=False)
def load_split(split: str, seed: int | None = None) -> list[SyntheticSample]:
    # Omitting seed preserves the documented default split used by existing pages.
    from vision_pipeline import generate_split

    return generate_split(split, seed)


def load_saved_report() -> dict[str, Any] | None:
    path = Path("generated-artifacts/latest_run.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_card(label: str, value: float, change: dict[str, Any] | None = None, inverse: bool = False) -> None:
    delta = None
    if change:
        absolute = change["absolute"]
        sign = "+" if absolute > 0 else ""
        delta = f"{sign}{absolute * 100:.1f} pp"
        if inverse:
            delta += " (lower is better)"
    st.metric(label, fmt_percent(value), delta=delta)


def render_metrics(metrics: dict[str, Any], changes: dict[str, Any] | None = None) -> None:
    columns = st.columns(4)
    for column, key, label, inverse in zip(
        columns,
        ("precision", "recall", "f1", "failure_rate"),
        ("Precision", "Recall", "F1 score", "Failure rate"),
        (False, False, False, True),
    ):
        with column:
            metric_card(label, metrics[key], changes.get(key) if changes else None, inverse)
    st.caption(
        f"{metrics['tp']} true positives · {metrics['fp']} false positives · "
        f"{metrics['fn']} missed targets · {metrics['images']} images"
    )


def render_condition_table(
    by_condition: dict[str, Any], comparison: dict[str, Any] | None = None
) -> None:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition not in by_condition:
            continue
        metrics = by_condition[condition]
        row: dict[str, Any] = {
            "Condition": CONDITION_LABELS[condition],
            "Images": metrics["images"],
            "Precision": fmt_percent(metrics["precision"]),
            "Recall": fmt_percent(metrics["recall"]),
            "F1": fmt_percent(metrics["f1"]),
            "Failure rate": fmt_percent(metrics["failure_rate"]),
            "FP": metrics["fp"],
            "Missed": metrics["fn"],
        }
        if comparison and condition in comparison:
            row["F1 change"] = f"{comparison[condition]['f1'] * 100:+.1f} pp"
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")


def render_condition_comparison(
    baseline: dict[str, Any], selected: dict[str, Any]
) -> None:
    rows = []
    for condition in CONDITIONS:
        before = baseline.get(condition)
        after = selected.get(condition)
        if not before or not after:
            continue
        rows.append(
            {
                "Condition": CONDITION_LABELS[condition],
                "Images": after["images"],
                "Baseline P / R / F1": " / ".join(
                    fmt_percent(before[key]) for key in ("precision", "recall", "f1")
                ),
                "Baseline failure": fmt_percent(before["failure_rate"]),
                "Selected P / R / F1": " / ".join(
                    fmt_percent(after[key]) for key in ("precision", "recall", "f1")
                ),
                "Selected failure": fmt_percent(after["failure_rate"]),
                "F1 change": f"{(after['f1'] - before['f1']) * 100:+.1f} pp",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")


def render_change_table(changes: dict[str, Any]) -> None:
    st.dataframe(
        [
            {
                "Metric": key.replace("_", " ").title(),
                "Absolute change": f"{change['absolute']:+.4f}",
                "Relative change": (
                    f"{change['relative']:+.1%}"
                    if change["relative"] is not None
                    else "undefined (baseline is zero)"
                ),
            }
            for key, change in changes.items()
        ],
        hide_index=True,
        width="stretch",
    )


def annotate_sample(
    sample: SyntheticSample,
    predictions: list[dict[str, Any]] | None = None,
) -> Any:
    image = sample.image.copy()
    x, y, width, height = sample.ground_truth.bbox
    cv2.rectangle(image, (x, y), (x + width, y + height), (80, 220, 90), 2)
    cv2.putText(
        image,
        "ground truth",
        (max(4, x), max(13, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (80, 220, 90),
        1,
        cv2.LINE_AA,
    )
    for index, prediction in enumerate(predictions or []):
        px, py, pw, ph = prediction["bbox"]
        color = (245, 155, 35) if index == 0 else (210, 80, 200)
        cv2.rectangle(image, (px, py), (px + pw, py + ph), color, 2)
        label = f"{prediction['source']} {prediction['confidence']:.2f}"
        cv2.putText(
            image,
            label,
            (max(4, px), min(IMAGE_HEIGHT - 4, py + ph + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


IMAGE_HEIGHT = 192
MAX_REAL_IMAGE_COUNT = 20
MAX_REAL_IMAGE_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_REAL_IMAGE_MANIFEST_BYTES = 1 * 1024 * 1024


def show_evaluation_failures(evaluation: dict[str, Any], samples: list[SyntheticSample], limit: int = 8) -> None:
    by_id = {sample.sample_id: sample for sample in samples}
    failures = [row for row in evaluation["image_results"] if not row["is_success"]]
    if not failures:
        st.success("No image-level failures in this evaluation.")
        return
    st.write(f"{len(failures)} failing images.")
    for row in failures[:limit]:
        sample = by_id.get(row["sample_id"])
        if sample is None:
            continue
        title = (
            f"{sample.sample_id} · {CONDITION_LABELS[row['condition']]} · "
            f"{row['fp']} FP · {row['fn']} missed"
        )
        with st.expander(title):
            st.image(annotate_sample(sample, row["predictions"]), width="stretch")
            st.json(
                {
                    "ground_truth": row["ground_truth"],
                    "predictions": row["predictions"],
                    "IoU_match_threshold": 0.30,
                }
            )


def overview_page() -> None:
    st.title("Agentic Vision Pipeline Repair")
    st.write(
        "A reproducible OpenCV experiment that measures where a circle detector fails, "
        "tests bounded configuration changes on validation images, and checks the selected "
        "configuration once against a separate held-out test set."
    )
    status = opencv_version_status()
    if status["is_opencv5"]:
        st.success(status["message"])
    else:
        st.error(status["message"])
    st.caption(
        "This is a technical proof of concept. It does not use an LLM, paid API, or fabricated metrics."
    )

    if not status["is_opencv5"]:
        return

    report = st.session_state.get("run_report")
    if report:
        st.subheader("Latest measured run")
        st.caption(
            f"{report['experiment_count']} candidate experiments · "
            f"{'repair selected' if report['repair_selected'] else 'baseline retained'}"
        )
        render_metrics(
            report["test"]["selected"]["metrics"],
            report["test"]["changes"],
        )
    else:
        st.info("No repair run has been saved yet. Run the agent to produce a trace and held-out comparison.")


def dataset_page() -> None:
    st.title("Dataset")
    st.write(
        "Synthetic 192 × 192 images contain one labeled, fully visible target. "
        "The target is drawn after background objects and only photometric changes, blur, or noise "
        "are applied, so its geometric labels remain valid."
    )
    st.markdown(
        "**Fixed seeds:** "
        f"training `{SPLIT_SEEDS['train']}`, validation `{SPLIT_SEEDS['validation']}`, "
        f"held-out test `{SPLIT_SEEDS['test']}`. "
        f"Split sizes: {SPLIT_SIZES['train']} / {SPLIT_SIZES['validation']} / {SPLIT_SIZES['test']}."
    )
    counts = st.columns(3)
    for column, split, title in zip(
        counts,
        ("train", "validation", "test"),
        ("Training", "Validation", "Held-out test"),
    ):
        column.metric(title, f"{SPLIT_SIZES[split]} images", f"seed {SPLIT_SEEDS[split]}")
    st.info(
        "The held-out test images are not previewed or used during tuning. They are only evaluated "
        "after the repair agent has stopped."
    )
    split = st.radio(
        "Preview split",
        ("train", "validation"),
        format_func=lambda key: "Training" if key == "train" else "Validation",
        horizontal=True,
    )
    samples = load_split(split)
    distribution = Counter(sample.condition for sample in samples)
    st.caption(
        "Condition counts: "
        + " · ".join(
            f"{CONDITION_LABELS[condition]} {distribution.get(condition, 0)}"
            for condition in CONDITIONS
        )
    )
    selected = [
        next(sample for sample in samples if sample.condition == condition)
        for condition in CONDITIONS
    ]
    columns = st.columns(3)
    for index, sample in enumerate(selected):
        with columns[index % 3]:
            st.image(annotate_sample(sample), width="stretch")
            st.caption(
                f"{sample.sample_id} · {CONDITION_LABELS[sample.condition]} · "
                f"center=({sample.ground_truth.center_x}, {sample.ground_truth.center_y}), "
                f"radius={sample.ground_truth.radius}px"
            )


def baseline_page() -> None:
    st.title("Baseline")
    st.write(
        "The baseline converts BGR images to HSV, thresholds a wraparound red hue interval, "
        "applies morphological open/close, extracts contours, and adds Hough circle proposals. "
        "Non-maximum suppression removes duplicate circle proposals before scoring."
    )
    st.markdown(
        "A prediction matches the single target only when its bounding-box IoU is at least **0.30**. "
        "At most one prediction can match; remaining predictions are false positives. "
        "An image-level failure has any false positive or missed target."
    )
    st.code(json.dumps(BASELINE_CONFIG.__dict__, indent=2), language="json")

    if st.button("Run baseline on validation set", type="primary"):
        with st.spinner("Running the baseline on validation images…"):
            validation = load_split("validation")
            result = evaluate_dataset(validation, BASELINE_CONFIG)
        st.session_state["baseline_validation"] = result

    result = st.session_state.get("baseline_validation")
    report = st.session_state.get("run_report")
    if result is None and report:
        result = report["validation"]["baseline"]
    if result:
        st.subheader("Validation metrics")
        render_metrics(result["metrics"])
        st.caption(f"Pipeline runtime: {result['runtime_ms']:.1f} ms")
        st.subheader("By image condition")
        render_condition_table(result["by_condition"])
        st.subheader("Failures")
        show_evaluation_failures(result, load_split("validation"))


def agent_run_page() -> None:
    st.title("Agent run")
    st.write(
        "The agent measures a validation baseline, finds the weakest condition group, proposes a "
        "parameter change for that condition, and uses the measured validation result to decide "
        "whether that configuration becomes the next best starting point."
    )
    st.markdown(
        f"Budget: **12 candidate experiments**. Search stops after four consecutive trials below "
        "the meaningful-improvement threshold or when its bounded candidates are exhausted. "
        "The objective is **F1 − 0.10 × image failure rate**; a gain of **0.002** is considered meaningful."
    )
    if not opencv_version_status()["is_opencv5"]:
        st.error(opencv_version_status()["message"])
        return

    if st.button("Run validation-only repair and held-out evaluation", type="primary"):
        progress = st.progress(0.0)
        with st.status("Starting measured validation experiments…", expanded=True) as status:

            def on_progress(event: dict[str, Any]) -> None:
                message = event.get("message", "")
                if message:
                    status.write(message)
                if event.get("phase") == "measured":
                    progress.progress(min(1.0, event["number"] / max(1, event["budget"])))
                elif event.get("phase") == "held_out":
                    progress.progress(1.0)
                    status.update(label="Validation search complete; running held-out test.", state="running")

            try:
                validation_samples = load_split("validation")
                test_samples = load_split("test")
                report = run_repair_agent(
                    validation_samples,
                    test_samples,
                    progress_callback=on_progress,
                )
                paths = save_run_report(report)
                report["saved_paths"] = paths
                # Save the paths in the report too so the downloadable trace includes them.
                Path(paths["trace"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
                st.session_state["run_report"] = report
                status.update(label="Experiment and held-out evaluation complete.", state="complete", expanded=False)
                st.success(
                    f"Completed {report['experiment_count']} candidate experiments. "
                    f"{'A validation improvement was selected.' if report['repair_selected'] else 'The baseline was retained.'}"
                )
            except Exception as error:
                status.update(label="Experiment stopped with an error.", state="error", expanded=True)
                st.error(f"{type(error).__name__}: {error}")

    report = st.session_state.get("run_report")
    if report:
        st.subheader("Current run")
        st.write(
            f"**Selected outcome:** {'repaired configuration' if report['repair_selected'] else 'baseline retained'}  \n"
            f"**Candidate experiments:** {report['experiment_count']} / {report['experiment_budget']}  \n"
            f"**Stopping condition:** {report['stopping_condition']}"
        )
        st.json(report["selected_config"])
        st.download_button(
            "Download complete run trace",
            data=json.dumps(report, indent=2),
            file_name="agentic-vision-run.json",
            mime="application/json",
        )


def results_page() -> None:
    st.title("Results")
    report = st.session_state.get("run_report")
    if not report:
        st.info("Run the repair agent first. Results will be based on its measured experiments.")
        return

    st.subheader("Held-out test comparison")
    st.warning(
        "The agent chose its configuration using validation only. Baseline and selected pipelines "
        "were then run on the same untouched held-out test image IDs."
    )
    test = report["test"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Baseline — test**")
        render_metrics(test["baseline"]["metrics"])
    with right:
        st.markdown("**Selected — test**")
        render_metrics(test["selected"]["metrics"], test["changes"])
    st.caption(
        f"Paired held-out samples: {test['comparison']['same_image_count']} · "
        f"same IDs: {test['comparison']['same_sample_ids']}"
    )

    tab_test, tab_validation, tab_runtime = st.tabs(
        ("Held-out by condition", "Validation selection", "Runtime and configuration")
    )
    with tab_test:
        render_condition_comparison(
            test["baseline"]["by_condition"],
            test["selected"]["by_condition"],
        )
        st.caption(
            "Changes are selected minus baseline. F1 change is in percentage points; "
            "the absolute and relative overall changes are shown below."
        )
    with tab_validation:
        validation = report["validation"]
        st.markdown("**Baseline validation**")
        render_metrics(validation["baseline"]["metrics"])
        st.markdown("**Selected validation**")
        render_metrics(validation["selected"]["metrics"], validation["changes"])
        render_condition_comparison(
            validation["baseline"]["by_condition"],
            validation["selected"]["by_condition"],
        )
        st.markdown("**Validation metric changes**")
        render_change_table(validation["changes"])
        st.caption(
            "Validation drove parameter selection. Test metrics did not participate in the search."
        )
    with tab_runtime:
        st.write(
            f"OpenCV: **{report['opencv']['opencv_version']}** "
            f"(package {report['opencv']['package_version']})"
        )
        st.write(
            f"Validation runs: {report['validation_evaluation_count']} "
            f"({report['experiment_count']} candidate experiments plus baseline)."
        )
        st.write(
            f"End-to-end agent run: {report['runtime_ms']:.1f} ms · "
            f"measured detector calls: {report['measured_pipeline_runtime_ms']:.1f} ms."
        )
        st.write(
            f"Runtime — baseline validation {validation['baseline']['runtime_ms']:.1f} ms, "
            f"selected validation {validation['selected']['runtime_ms']:.1f} ms, "
            f"baseline test {test['baseline']['runtime_ms']:.1f} ms, "
            f"selected test {test['selected']['runtime_ms']:.1f} ms."
        )
        st.write(f"Stopped because: {report['stopping_condition']}")
        st.markdown("**Baseline configuration**")
        st.json(report["baseline_config"])
        st.markdown("**Selected configuration**")
        st.json(report["selected_config"])
        st.markdown("**Metric-change rule**")
        st.caption(
            "Absolute change is selected minus baseline. Relative change is absolute change divided "
            "by the baseline metric; if a baseline metric is zero, relative change is undefined."
        )
        changes = test["changes"]
        render_change_table(changes)

    st.download_button(
        "Download selected config",
        data=json.dumps(
            {
                "selected_config": report["selected_config"],
                "repair_selected": report["repair_selected"],
                "validation_metrics": validation["selected"]["metrics"],
                "selection_rule": report["objective_rule"],
            },
            indent=2,
        ),
        file_name="selected-config.json",
        mime="application/json",
    )
    if report.get("saved_paths"):
        st.caption(
            "Saved locally: "
            f"`{report['saved_paths']['config']}` and `{report['saved_paths']['trace']}`."
        )


def failure_analysis_page() -> None:
    st.title("Failure analysis")
    report = st.session_state.get("run_report")
    if not report:
        st.info("A completed agent run is needed to compare repaired and baseline failures.")
        return

    scope = st.radio(
        "Compare split",
        ("test", "validation"),
        format_func=lambda key: "Held-out test" if key == "test" else "Validation",
        horizontal=True,
    )
    if scope == "test":
        comparison = report["test"]["comparison"]
        split_samples = load_split("test")
    else:
        comparison = report["validation_failure_analysis"]
        split_samples = load_split("validation")
    counts = comparison["counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Same images", comparison["same_image_count"])
    c2.metric("Fixed by selected config", counts["repair_succeeded"])
    c3.metric("Regressions", counts["regression"])
    c4.metric("Still failing", counts["still_failing"])

    examples = comparison["examples"]
    if not examples:
        st.success("Baseline and selected pipeline both handled every image without extra detections.")
        return
    sample_ids = [row["sample_id"] for row in examples]
    selected_id = st.selectbox("Inspect an example", sample_ids)
    example = next(row for row in examples if row["sample_id"] == selected_id)
    sample = next(item for item in split_samples if item.sample_id == selected_id)
    st.write(
        f"**Outcome:** {example['outcome'].replace('_', ' ').title()} · "
        f"**Condition:** {CONDITION_LABELS[example['condition']]}"
    )
    baseline_col, repaired_col = st.columns(2)
    with baseline_col:
        st.markdown("**Baseline**")
        st.image(annotate_sample(sample, example["baseline"]["predictions"]), width="stretch")
        st.json({key: example["baseline"][key] for key in ("tp", "fp", "fn", "is_success")})
    with repaired_col:
        st.markdown("**Selected configuration**")
        st.image(annotate_sample(sample, example["repaired"]["predictions"]), width="stretch")
        st.json({key: example["repaired"][key] for key in ("tp", "fp", "fn", "is_success")})


def real_image_page() -> None:
    st.title("Real-image evaluation")
    st.warning(
        "Real-world validation requires independently collected and labeled images. "
        "Synthetic images, labels created by the app, or tunable real-image sets are not real-world evidence."
    )
    st.write(
        "Upload matching images and a JSON manifest with one target circle per image. "
        "The baseline configuration is evaluated using the same detector and IoU rules as the synthetic benchmark."
    )
    manifest_format = {
        "filename": "example.jpg",
        "center_x": 320,
        "center_y": 240,
        "radius": 18,
        "condition": "outdoor",
    }
    st.code(json.dumps([manifest_format], indent=2), language="json")
    st.caption(
        "Uploads are limited to 20 images, 20 MB per file, 50 MB total, and a 1 MB manifest."
    )

    uploaded_images = st.file_uploader(
        "Labeled images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )
    manifest_upload = st.file_uploader("Annotation manifest JSON", type=["json"])
    if uploaded_images and manifest_upload:
        if st.button("Evaluate uploaded real images", type="primary"):
            if len(uploaded_images) > MAX_REAL_IMAGE_COUNT:
                st.error(f"Upload no more than {MAX_REAL_IMAGE_COUNT} images at a time.")
                return
            if manifest_upload.size > MAX_REAL_IMAGE_MANIFEST_BYTES:
                st.error("The annotation manifest must be no larger than 1 MB.")
                return
            total_upload_bytes = manifest_upload.size + sum(
                uploaded_image.size for uploaded_image in uploaded_images
            )
            if total_upload_bytes > MAX_REAL_IMAGE_UPLOAD_BYTES:
                st.error("The manifest and images together must be no larger than 50 MB.")
                return

            import tempfile
            from pathlib import Path

            from real_image_evaluation import load_real_image_manifest

            with tempfile.TemporaryDirectory() as temporary_directory:
                temporary_path = Path(temporary_directory)
                manifest_path = temporary_path / "manifest.json"
                manifest_path.write_bytes(manifest_upload.getvalue())
                uploaded_names = set()
                for uploaded_image in uploaded_images:
                    filename = Path(uploaded_image.name).name
                    if filename in uploaded_names:
                        st.error(f"Duplicate uploaded filename: {filename}")
                        return
                    uploaded_names.add(filename)
                    (temporary_path / filename).write_bytes(uploaded_image.getvalue())
                try:
                    manifest_records = json.loads(manifest_upload.getvalue())
                    if isinstance(manifest_records, dict):
                        manifest_records = manifest_records.get("images")
                    manifest_names = {
                        record["filename"] for record in manifest_records
                        if isinstance(record, dict) and isinstance(record.get("filename"), str)
                    }
                    if uploaded_names != manifest_names:
                        missing = sorted(manifest_names - uploaded_names)
                        extra = sorted(uploaded_names - manifest_names)
                        raise ValueError(
                            "Uploaded images must exactly match manifest filenames"
                            + (f"; missing: {', '.join(missing)}" if missing else "")
                            + (f"; extra: {', '.join(extra)}" if extra else "")
                        )
                    samples = load_real_image_manifest(manifest_path, temporary_path)
                    evaluation = evaluate_real_images(samples)
                except Exception as error:
                    st.error(f"{type(error).__name__}: {error}")
                    return

            st.subheader("Real-image results")
            st.caption("These are actual uploaded-image results only; no real-image data affected configuration selection.")
            render_metrics(evaluation["metrics"])
            rows = []
            for image_result in evaluation["image_results"]:
                rows.append({
                    "Image": image_result["sample_id"].removeprefix("real-"),
                    "Condition": image_result["condition"],
                    "TP": image_result["tp"],
                    "FP": image_result["fp"],
                    "FN": image_result["fn"],
                    "Matched IoU": image_result["matched_iou"],
                    "Success": image_result["is_success"],
                })
            st.dataframe(rows, hide_index=True, width="stretch")
            st.download_button(
                "Download real-image evaluation JSON",
                data=json.dumps(evaluation, indent=2),
                file_name="real-image-evaluation.json",
                mime="application/json",
            )


def trace_page() -> None:
    st.title("Agent trace")
    report = st.session_state.get("run_report")
    if not report:
        st.info("The trace is created by the agent run. Each step includes its decision and measured results.")
        return
    st.write(
        f"**Candidate experiments:** {report['experiment_count']} · "
        f"**Validation evaluations:** {report['validation_evaluation_count']} · "
        f"**Outcome:** {'repair selected' if report['repair_selected'] else 'baseline retained'}"
    )
    st.caption(report["objective_rule"])
    for event in report["trace"]:
        title = f"Step {event['step']} · {event['type'].replace('_', ' ').title()}"
        with st.expander(title, expanded=event["type"] in ("decision", "experiment")):
            if event.get("decision"):
                st.write(event["decision"])
            if event.get("config"):
                st.markdown("**Configuration**")
                st.json(event["config"])
            if event.get("metrics"):
                st.markdown("**Measured validation metrics**")
                render_metrics(event["metrics"])
            if event.get("observation"):
                st.markdown("**Observation — measured evidence**")
                st.json(event["observation"])
            if event.get("diagnosis"):
                st.markdown("**Diagnosis**")
                st.write(event["diagnosis"]["diagnosis"])
                st.caption(event["diagnosis"]["heuristic_basis"])
            if event.get("action"):
                st.markdown("**Chosen action**")
                st.write(event["action"]["reason"])
                st.json(event["action"]["parameter_changes"])
            if event.get("validation_metrics"):
                st.markdown("**Measured candidate metrics**")
                render_metrics(event["validation_metrics"])
                st.caption(
                    f"Objective: {event['objective']:.4f} · "
                    f"change vs best: {event['objective_change_vs_best']:+.4f} · "
                    f"Decision: {event.get('accept_reject', 'unknown')}"
                )
                if event.get("rollback"):
                    st.caption(event["rollback"])
            if event.get("by_condition"):
                render_condition_table(event["by_condition"])
            if event.get("validation_by_condition"):
                render_condition_table(event["validation_by_condition"])
            if event.get("baseline_metrics"):
                st.write("Held-out baseline and selected metrics:")
                st.json(
                    {
                        "baseline": event["baseline_metrics"],
                        "selected": event["selected_metrics"],
                        "same_sample_ids": event["same_sample_ids"],
                    }
                )
    st.download_button(
        "Download auditable trace JSON",
        data=json.dumps(report, indent=2),
        file_name="agentic-vision-run.json",
        mime="application/json",
    )


if "run_report" not in st.session_state:
    st.session_state["run_report"] = load_saved_report()
if "benchmark_report" not in st.session_state:
    st.session_state["benchmark_report"] = load_saved_benchmark()

page = st.sidebar.radio(
    "Experiment",
    (
        "Overview",
        "Dataset",
        "Baseline",
        "Agent run",
        "Benchmark",
        "Real-image evaluation",
        "Results",
        "Failure analysis",
        "Agent trace",
    ),
)
st.sidebar.caption("OpenCV AI Competition 2026 · MVP")

if page == "Overview":
    overview_page()
elif page == "Dataset":
    dataset_page()
elif page == "Baseline":
    baseline_page()
elif page == "Agent run":
    agent_run_page()
elif page == "Benchmark":
    render_benchmark_page(load_split, annotate_sample)
elif page == "Real-image evaluation":
    real_image_page()
elif page == "Results":
    results_page()
elif page == "Failure analysis":
    failure_analysis_page()
elif page == "Agent trace":
    trace_page()
