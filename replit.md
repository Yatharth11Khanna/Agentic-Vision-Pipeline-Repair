# Agentic Vision Pipeline Repair

An OpenCV 5 proof of concept that measures synthetic circle-detection failures, tests bounded validation-driven repairs, and reports untouched held-out test results.

## Run & Operate

- `.pythonlibs/bin/streamlit run app.py --server.port 5000 --server.address 0.0.0.0` — run the Streamlit app
- `.pythonlibs/bin/python -m pytest -q` — run Python tests
- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required environment variables: none

## Stack

- Python 3.11
- Streamlit, NumPy, Matplotlib, pytest
- `opencv-python-headless==5.0.0.93` (OpenCV 5 is verified at startup; no OpenCV 4 fallback)
- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `app.py` — Streamlit experiment interface
- `vision_pipeline.py` — deterministic synthetic data, OpenCV detector, matching, and metrics
- `repair_agent.py` — bounded validation search, held-out evaluation, and JSON trace persistence
- `tests/` — pytest coverage
- `README.md` — experiment methodology, seeds, matching rule, and run instructions

## Architecture decisions

- Configuration selection uses validation only; held-out images are evaluated after the search stops.
- A prediction must reach IoU 0.30 to match the single labeled target; duplicate detections count as false positives.
- The repair objective is F1 minus 0.10 times image failure rate, with a 0.002 threshold for meaningful improvement.
- The fixed train, validation, and test seeds are 1103, 2207, and 3301.

## Product

Users can inspect deterministic synthetic images, run the baseline, execute a measured parameter search, compare held-out metrics, inspect regressions and failures, and download the chosen configuration and full agent trace.

## User preferences

- Treat this as a technical proof of concept: prioritize a reproducible end-to-end experiment over visual polish.
- Do not use a paid API or present fabricated performance numbers.

## Gotchas

- Keep `opencv-python-headless==5.0.0.93` pinned. If OpenCV 5 is unavailable or fails its runtime check, report the exact failure rather than claiming OpenCV 4 compliance.
- Do not add test-set evaluation to the selection loop.
- Run reports are written under `generated-artifacts/` and excluded from deploy/check-in output.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
