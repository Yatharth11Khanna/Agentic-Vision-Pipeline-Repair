FROM python:3.12.3-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.13.5

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY app.py benchmark.py benchmark_reporting.py benchmark_ui.py ./
COPY main.py repair_agent.py vision_pipeline.py real_image_evaluation.py ./
COPY README.md ./

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home appuser \
    && mkdir -p /app/generated-artifacts /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser/.streamlit

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://127.0.0.1:${PORT:-8080}/_stcore/health || exit 1

CMD ["sh", "-c", "python -c \"import cv2; from importlib.metadata import version; runtime=cv2.__version__; package=version('opencv-python-headless'); print(f'OpenCV runtime={runtime}; package={package}'); assert runtime.startswith('5.'); assert package.startswith('5.'); import streamlit; print(f'Streamlit={streamlit.__version__}')\" && exec python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true --browser.gatherUsageStats=false"]
