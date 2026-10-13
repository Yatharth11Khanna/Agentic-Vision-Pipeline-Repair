# Agentic Vision Pipeline Repair

A reproducible proof of concept for an autonomous OpenCV pipeline tester. It creates synthetic
colored-circle images, measures a real HSV/morphology/contour/Hough detector, searches bounded
parameter changes using validation metrics, and compares the baseline and selected configurations
on the same held-out test images.

## Run

The project is pinned to Python 3.11+ and `opencv-python-headless==5.0.0.93`. OpenCV 4 is not
accepted as a fallback.

```bash
.pythonlibs/bin/streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

The Replit workflow starts the same command. To run the test suite:

```bash
.pythonlibs/bin/python -m pytest -q
```

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
  prioritizes the weakest condition, proposes condition-relevant bounded parameter changes, and
  uses measured results to update its best configuration and next decision.
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

