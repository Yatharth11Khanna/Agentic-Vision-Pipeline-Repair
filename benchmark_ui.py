"""Streamlit components for running and inspecting reproducible benchmark trials."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from benchmark import (
    DEFAULT_BASE_SEED,
    DEFAULT_TRIAL_COUNT,
    MAX_TRIALS,
    MIN_TRIALS,
    run_benchmark,
)
from benchmark_reporting import METHOD_LABELS
from vision_pipeline import CONDITION_LABELS, CONDITIONS


def load_saved_benchmark() -> dict[str, Any] | None:
    path = Path("generated-artifacts/benchmark/latest_benchmark.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _display_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _render_method_summary(report: dict[str, Any]) -> None:
    st.subheader("Held-out test results")
    st.caption(
        "Each method was evaluated on the same test images within a trial. "
        "Intervals describe variation across independent trials."
    )
    aggregates = report["aggregates"]["methods"]
    columns = st.columns(len(METHOD_LABELS))
    for column, method in zip(columns, METHOD_LABELS):
        f1 = aggregates[method]["test_metrics"]["f1"]
        with column:
            st.metric(
                f"{METHOD_LABELS[method]} · mean F1",
                _display_percent(f1["mean"]),
            )
            st.caption(
                f"95% CI {_display_percent(f1['ci95_low'])}–"
                f"{_display_percent(f1['ci95_high'])} · SD "
                f"{_display_percent(f1['std'])}"
            )

    rows: list[dict[str, Any]] = []
    for method, method_report in aggregates.items():
        row: dict[str, Any] = {"Method": METHOD_LABELS[method]}
        for metric in ("precision", "recall", "f1", "failure_rate"):
            summary = method_report["test_metrics"][metric]
            label = metric.replace("_", " ").title()
            row[f"{label} mean"] = _display_percent(summary["mean"])
            row[f"{label} SD"] = _display_percent(summary["std"])
            row[f"{label} 95% CI"] = (
                f"{_display_percent(summary['ci95_low'])}–"
                f"{_display_percent(summary['ci95_high'])}"
            )
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")

    chart_rows = [
        {"Method": METHOD_LABELS[method], "F1": aggregates[method]["test_metrics"]["f1"]["mean"]}
        for method in METHOD_LABELS
    ]
    st.markdown("**Mean held-out F1 by method**")
    st.bar_chart(
        pd.DataFrame(chart_rows),
        x="Method",
        y="F1",
        color="Method",
        y_min=0,
        y_max=1,
    )


def _render_condition_results(report: dict[str, Any]) -> None:
    st.subheader("Held-out results by condition")
    aggregates = report["aggregates"]["methods"]
    condition_rows: list[dict[str, Any]] = []
    f1_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        label = CONDITION_LABELS[condition]
        chart_row: dict[str, Any] = {"Condition": label}
        for method in METHOD_LABELS:
            metrics = aggregates[method]["test_by_condition"][condition]
            chart_row[METHOD_LABELS[method]] = metrics["f1"]["mean"]
            row: dict[str, Any] = {
                "Condition": label,
                "Method": METHOD_LABELS[method],
            }
            for metric in ("precision", "recall", "f1", "failure_rate"):
                summary = metrics[metric]
                metric_label = metric.replace("_", " ").title()
                row[f"{metric_label} mean"] = _display_percent(summary["mean"])
                row[f"{metric_label} SD"] = _display_percent(summary["std"])
                row[f"{metric_label} 95% CI"] = (
                    f"{_display_percent(summary['ci95_low'])}–"
                    f"{_display_percent(summary['ci95_high'])}"
                )
            condition_rows.append(row)
        f1_rows.append(chart_row)

    st.line_chart(
        pd.DataFrame(f1_rows).set_index("Condition"),
        y=[METHOD_LABELS[method] for method in METHOD_LABELS],
        y_min=0,
        y_max=1,
    )
    st.dataframe(condition_rows, hide_index=True, width="stretch")


def _render_paired_differences(report: dict[str, Any]) -> None:
    st.subheader("Paired method differences")
    rows: list[dict[str, Any]] = []
    for relation, metric_summaries in report["aggregates"]["paired_differences"].items():
        left, _, right = relation.partition("_vs_")
        label = f"{METHOD_LABELS[left]} − {METHOD_LABELS[right]}"
        for metric, summary in metric_summaries.items():
            rows.append(
                {
                    "Comparison": label,
                    "Metric": metric.replace("_", " ").title(),
                    "Mean difference": f"{summary['mean']:+.4f}",
                    "SD": f"{summary['std']:.4f}",
                    "95% CI": f"{summary['ci95_low']:+.4f} to {summary['ci95_high']:+.4f}",
                }
            )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Positive precision, recall, and F1 differences favor the left-hand method. "
        "A negative failure-rate difference favors it."
    )


def _render_trial_details(report: dict[str, Any]) -> None:
    st.subheader("Per-trial results and compute")
    rows = []
    for trial in report["trials"]:
        row: dict[str, Any] = {
            "Trial": trial["trial_id"],
            "Train seed": trial["seeds"]["train"],
            "Validation seed": trial["seeds"]["validation"],
            "Random-search seed": trial["seeds"]["random_search"],
            "Held-out seed": trial["seeds"]["test"],
        }
        for method in METHOD_LABELS:
            method_data = trial["methods"][method]
            prefix = METHOD_LABELS[method]
            row[f"{prefix} F1"] = _display_percent(method_data["test"]["metrics"]["f1"])
            row[f"{prefix} failures"] = (
                f"{method_data['test']['metrics']['failure_count']}/"
                f"{method_data['test']['metrics']['images']}"
            )
            row[f"{prefix} experiments"] = method_data["experiment_count"]
            row[f"{prefix} total ms"] = method_data["total_method_runtime_ms"]
        row["Matched search budgets"] = (
            trial["methods"]["adaptive"]["experiment_count"]
            == trial["methods"]["random_search"]["experiment_count"]
        )
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")

    method_rows = []
    for method, method_report in report["aggregates"]["methods"].items():
        runtime = method_report["runtime_and_budget"]
        method_rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Mean candidates": f"{runtime['experiment_count']['mean']:.2f}",
                "Candidate SD": f"{runtime['experiment_count']['std']:.2f}",
                "Mean search ms": f"{runtime['selection_runtime_ms']['mean']:.1f}",
                "Mean candidate pipeline ms": f"{runtime['candidate_pipeline_runtime_ms']['mean']:.1f}",
                "Mean held-out evaluation ms": f"{runtime['test_evaluation_runtime_ms']['mean']:.1f}",
                "Mean total method ms": f"{runtime['total_method_runtime_ms']['mean']:.1f}",
            }
        )
    st.markdown("**Aggregate experiment counts and runtime**")
    st.dataframe(method_rows, hide_index=True, width="stretch")

    st.markdown("**F1 by trial**")
    trial_chart_rows = []
    for trial in report["trials"]:
        chart_row: dict[str, Any] = {"Trial": trial["trial_id"]}
        chart_row.update(
            {
                METHOD_LABELS[method]: trial["methods"][method]["test"]["metrics"]["f1"]
                for method in METHOD_LABELS
            }
        )
        trial_chart_rows.append(chart_row)
    st.line_chart(pd.DataFrame(trial_chart_rows).set_index("Trial"))


def _render_failure_analysis(
    report: dict[str, Any],
    load_split: Callable[[str, int | None], list[Any]],
    annotate_sample: Callable[[Any, list[dict[str, Any]] | None], Any],
) -> None:
    st.subheader("Repairs, regressions, and remaining failures")
    count_rows = []
    for relation, counts in report["aggregates"]["paired_image_outcome_counts"].items():
        left, _, right = relation.partition("_vs_")
        row: dict[str, Any] = {
            "Comparison": f"{METHOD_LABELS[left]} vs {METHOD_LABELS[right]}"
        }
        row.update(counts)
        count_rows.append(row)
    st.dataframe(count_rows, hide_index=True, width="stretch")

    trial_id = st.selectbox(
        "Trial for image-level examples",
        [trial["trial_id"] for trial in report["trials"]],
        key="benchmark_failure_trial",
    )
    selected_trial = next(trial for trial in report["trials"] if trial["trial_id"] == trial_id)
    relation_labels = {
        "adaptive_vs_baseline": "Adaptive repair vs baseline",
        "random_search_vs_baseline": "Random search vs baseline",
        "adaptive_vs_random_search": "Adaptive repair vs random search",
    }
    relation = st.selectbox(
        "Method comparison",
        list(relation_labels),
        format_func=lambda item: relation_labels[item],
        key="benchmark_failure_relation",
    )
    comparison = selected_trial["comparisons"][relation]
    st.write(f"Image comparison counts: {comparison['counts']}")
    examples = comparison["examples"]
    if not examples:
        st.success("No failures or regressions were observed for this method comparison.")
        return

    example_id = st.selectbox(
        "Failure or regression example",
        [example["sample_id"] for example in examples],
        key="benchmark_failure_example",
    )
    example = next(row for row in examples if row["sample_id"] == example_id)
    sample = next(
        item
        for item in load_split("test", selected_trial["seeds"]["test"])
        if item.sample_id == example_id
    )
    st.caption(
        f"{example['outcome'].replace('_', ' ').title()} · "
        f"{CONDITION_LABELS[example['condition']]}"
    )

    if relation == "adaptive_vs_random_search":
        left, right = example["adaptive"], example["random_search"]
    else:
        left, right = example["baseline"], example["repaired"]
    if relation == "adaptive_vs_baseline":
        left_label, right_label = METHOD_LABELS["baseline"], METHOD_LABELS["adaptive"]
    elif relation == "random_search_vs_baseline":
        left_label, right_label = METHOD_LABELS["baseline"], METHOD_LABELS["random_search"]
    else:
        left_label, right_label = (
            METHOD_LABELS["adaptive"],
            METHOD_LABELS["random_search"],
        )
    columns = st.columns(2)
    for column, label, outcome in zip(columns, (left_label, right_label), (left, right)):
        with column:
            st.markdown(f"**{label}**")
            st.image(annotate_sample(sample, outcome["predictions"]), width="stretch")
            st.json(
                {
                    key: outcome[key]
                    for key in ("tp", "fp", "fn", "is_success")
                }
            )


def render_benchmark_page(
    load_split: Callable[[str, int | None], list[Any]],
    annotate_sample: Callable[[Any, list[dict[str, Any]] | None], Any],
) -> None:
    st.title("Reproducible benchmark")
    st.write(
        "Compare the unchanged baseline, validation-driven adaptive repair, and a seeded "
        "non-adaptive random search with the same candidate-evaluation budget."
    )
    st.info(
        "Each trial uses new, separately seeded training, validation, random-search, and held-out "
        "splits. Both searches finish using validation images before the test split is generated. "
        "All three methods then run on the same held-out images for that trial."
    )

    with st.form("benchmark-run-form"):
        first, second = st.columns(2)
        with first:
            trial_count = st.number_input(
                "Independent trials",
                min_value=MIN_TRIALS,
                max_value=MAX_TRIALS,
                value=DEFAULT_TRIAL_COUNT,
                step=1,
            )
        with second:
            base_seed = st.number_input(
                "Base seed",
                min_value=0,
                max_value=2**32 - 1 - MAX_TRIALS * 10_000 - 100,
                value=DEFAULT_BASE_SEED,
                step=1,
            )
        run = st.form_submit_button("Run benchmark", type="primary")

    if run:
        progress_bar = st.progress(0.0)
        status = st.empty()

        def update_progress(event: dict[str, Any]) -> None:
            progress_bar.progress(float(event["progress"]))
            status.caption(event["message"])

        try:
            with st.spinner("Running the trial searches and held-out evaluations…"):
                result = run_benchmark(
                    trial_count=int(trial_count),
                    base_seed=int(base_seed),
                    progress_callback=update_progress,
                )
                paths = save_benchmark_report(result)
            st.session_state["benchmark_report"] = result
            status.success(
                f"Completed {result['trial_count']} trials. "
                f"JSON and CSV saved under {paths['json']} and {paths['csv']}."
            )
        except Exception as error:  # Surface failure without replacing the last valid report.
            status.empty()
            st.error(f"Benchmark did not complete: {type(error).__name__}: {error}")

    report = st.session_state.get("benchmark_report")
    if report is None:
        report = load_saved_benchmark()
        if report:
            st.session_state["benchmark_report"] = report
    if not report:
        st.warning("Run the benchmark to create a report. No prior benchmark report was found.")
        return

    st.caption(
        f"{report['trial_count']} trials · base seed {report['base_seed']} · "
        f"OpenCV {report['opencv']['opencv_version']} · started {report['created_at']}"
    )
    st.success(report["conclusion"])
    _render_method_summary(report)

    st.subheader("Validation-selected results")
    st.caption("This section is separate from held-out test results and is not used for the final comparison.")
    validation_rows = []
    for method, method_report in report["aggregates"]["methods"].items():
        row = {"Method": METHOD_LABELS[method]}
        for metric in ("precision", "recall", "f1", "failure_rate"):
            summary = method_report["validation_metrics"][metric]
            label = metric.replace("_", " ").title()
            row[f"{label} mean"] = _display_percent(summary["mean"])
            row[f"{label} SD"] = _display_percent(summary["std"])
            row[f"{label} 95% CI"] = (
                f"{_display_percent(summary['ci95_low'])}–"
                f"{_display_percent(summary['ci95_high'])}"
            )
        validation_rows.append(row)
    st.dataframe(validation_rows, hide_index=True, width="stretch")

    _render_condition_results(report)
    _render_paired_differences(report)
    _render_trial_details(report)
    _render_failure_analysis(report, load_split, annotate_sample)

    st.subheader("Benchmark protocol and reproducibility")
    with st.expander("Search bounds and evaluation rules"):
        st.json(
            {
                "candidate_space": report["candidate_space"],
                "experiment_budget_rule": report["experiment_budget_rule"],
                "selection_protocol": report["selection_protocol"],
                "training_split_note": report["training_split_note"],
                "objective_rule": report["objective_rule"],
                "nonadaptive_rule": report["nonadaptive_rule"],
                "matching_rule": report["matching_rule"],
                "seed_plan": report["seeds"],
            }
        )
    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download full benchmark JSON",
            data=json.dumps(report, indent=2),
            file_name="agentic-vision-benchmark.json",
            mime="application/json",
            width="stretch",
        )
    csv_path = Path(report.get("saved_paths", {}).get("csv", ""))
    if csv_path.exists():
        with download_columns[1]:
            st.download_button(
                "Download trial and aggregate CSV",
                data=csv_path.read_bytes(),
                file_name="agentic-vision-benchmark.csv",
                mime="text/csv",
                width="stretch",
            )
    else:
        with download_columns[1]:
            st.warning("The saved CSV report is unavailable.")
