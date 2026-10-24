# Agentic Vision Pipeline Repair

A reproducible proof of concept for an autonomous OpenCV pipeline tester. It creates synthetic
colored-circle images, measures a real HSV/morphology/contour/Hough detector, searches bounded
parameter changes using validation metrics, and compares the baseline and selected configurations
on the same held-out test images.

## Run

The project is pinned to Python 3.11+ and `opencv-python-headless==5.0.0.93`. OpenCV 4 is not
accepted as a fallback. The verified development environment uses Python 3.12 with the project
virtual environment at `.venv`.

```bash
.venv/bin/python -m streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

To run the complete test suite:

```bash
.venv/bin/python -m pytest -q
```

The app checks both `cv2.__version__` and the installed package metadata at startup. A missing
or incompatible OpenCV 5 installation is reported as an error and is never silently replaced by
OpenCV 4.

## Deterministic agent demo

With the fixed validation seed `2207`, a six-candidate demonstration using Python 3.12.3 produced
this reproducible sequence in the validation-only agent run:

1. Experiment 1 rejected `hue_center=164`; objective changed from `0.783334` to `0.116421`.
2. Experiment 2 rejected `hue_center=172`; objective changed from `0.783334` to `0.766884`.
3. Experiment 3 rejected `hue_center=16`; objective changed from `0.783334` to `0.059273`.
4. Experiment 4 accepted `hue_center=8`; objective increased from `0.783334` to `0.839036`.
5. Experiment 5 rejected `hue_center=24`, measured from `0.839036` to `0.065368`.
6. Experiment 6 rejected `hue_tolerance=26`, measured from `0.839036` to `0.667633`; the accepted
   `hue_center=8` configuration remained selected.

Execute and inspect the trace directly:

```bash
.venv/bin/python - <<'PY'
from repair_agent import run_adaptive_selection
from vision_pipeline import generate_split

report = run_adaptive_selection(generate_split("validation"), budget=6)
print("selected config:", report["selected_config"])
for event in report["trace"]:
    if event["type"] == "experiment":
        print(event["number"], event["diagnosis"]["diagnosis"],
              event["action"]["parameter_changes"], event["accept_reject"])
PY
```

This short trace is illustrative of the loop, not a claim that the repair is better than random
search. The six-budget sequence is separate from the five-trial benchmark's existing stopping rules.

## Production container

Build and run the OpenCV 5 application locally:

```bash
docker build -t agentic-vision-pipeline-repair .
docker run --rm -p 8080:8080 agentic-vision-pipeline-repair
```

The service binds to `0.0.0.0:${PORT:-8080}` and exposes Streamlit health at
`/_stcore/health`. Uploaded images and manifests are processed in temporary container storage;
they are not persisted when a container restarts or is replaced. Do not use ephemeral storage
as a dataset archive. Supply independently labeled data again when a new container instance starts.

At startup, the app checks both `cv2.__version__` and the installed package version. A missing,
incompatible, or non-5 installation is shown as an error; it is never described as OpenCV 5.

## Experiment design

- Each image is 192 × 192 with one visible circle and a bounding-box, center, and radius label.
- The target is drawn after background distractors. Brightness, blur, hue, and noise transforms do
  not alter its geometry.
- The seven conditions are normal lighting, low brightness, high brightness, blur, color shift,
  noise, and distractors.
- The fixed split seeds are training `1103`, validation `2207`, and held-out test `3301`.
- Split sizes are 84 training, 42 validation, and 49 test images. Conditions are balanced within
  each split.
- The baseline uses HSV color conversion and wraparound hue thresholding, morphological open and
  close, contour geometry, Hough circle candidates, and confidence-based non-maximum suppression.
- A prediction matches the sole target at IoU ≥ 0.30. Predictions are sorted by confidence, at most
  one can match the target, and all unmatched predictions are false positives. An image fails if
  it has any false positive or false negative.
- The agent evaluates its initial configuration and every candidate on validation images only. It
  prioritizes the weakest condition, diagnoses its measured FP/FN pattern, and reorders existing
  bounded condition-specific proposals before execution. The diagnosis is a heuristic based on
  aggregate counts, not proof of an image-level causal mechanism.
- The search budget is 12 candidates. It stops after four consecutive experiments below the
  meaningful-improvement threshold, if no validation failures remain to guide a search, or if the
  unique candidate space is exhausted.
- The selection objective is `F1 - 0.10 × image failure rate`. An objective gain of at least 0.002
  is considered meaningful. The agent is allowed to retain the baseline.
- After selection stops, baseline and selected configurations are each run once on the exact same
  49 held-out test images. Test measurements do not influence parameter selection.

Precision, recall, F1, image failure rate, per-condition metrics, runtime, configuration changes,
false-positive/false-negative counts, and regressions are retained in the run report. The latest
report and selected configuration are written to `generated-artifacts/` and can be downloaded from
the app.

## Multi-trial benchmark

Open **Benchmark** in the sidebar to compare three methods over 5–20 independent trials:

1. The unchanged baseline.
2. The adaptive validation-driven repair agent.
3. A seeded non-adaptive random search over the same declared parameter bounds.

Each trial uses independent train, validation, random-search, and test seeds. Both searches receive
validation samples only. The test split is generated only after both searches have selected a
configuration, and all three methods are evaluated on the same held-out images. Random search
evaluates exactly as many candidates as adaptive repair actually evaluated in that trial, including
when adaptive repair stops early. Its full candidate sequence is generated before candidate scoring
and cannot react to results.

The report includes per-trial and aggregate precision, recall, F1, image failure rate, condition-level
results, paired method differences, mean and sample standard deviation, two-sided 95% Student-t
intervals, experiment counts, runtime, and per-image repair/regression/failure comparisons. The
app's summary calls out when the held-out data do not show a clear adaptive advantage.

Benchmark files are separate from the existing single-run report:

- `generated-artifacts/benchmark/latest_benchmark.json` — full seeds, results, comparisons, and
  adaptive/random-search traces.
- `generated-artifacts/benchmark/latest_benchmark.csv` — per-trial and aggregate overall and
  per-condition metric rows for validation and held-out test data.

Use the app's download buttons to export either file. The default is five trials with base seed
`20261013`; the base seed and trial count can be changed in the Benchmark page.

## Submission status and limitations

- The benchmark and all performance evidence currently use synthetic, deterministically generated
  images with independently generated labels.
- The real-image upload workflow evaluates independently uploaded, labeled images with the same
  detector and IoU rules, but no genuine real-world dataset evaluation has been verified.
- AWS deployment has not been completed. There is no public AWS URL or deployment claim.
- The fresh five-trial causal-diagnosis benchmark produced mean held-out F1 values of 0.619403
  baseline, 0.698988 adaptive, and 0.678928 random search. Adaptive minus random was +0.0201 with
  95% CI [-0.0357, +0.0759], so adaptive superiority is not established.
- Official competition deadline and submission/judging requirements must be checked against the
  current competition rules; this repository does not assert that AWS hosting is mandatory.

