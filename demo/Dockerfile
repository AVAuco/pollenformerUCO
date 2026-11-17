# syntax=docker/dockerfile:1
#
# PollenFormerUCO demo image. The build context is the repository root:
#   docker build -f demo/Dockerfile -t pollenformer-uco-demo .
# The Hugging Face Space receives the same tree with this file at its root (scripts/stage_space.py)
# and builds the LAST stage, so `app` must stay last.
#
# Model artifacts are not baked in: the container downloads and verifies them on start
# (`pollenformer fetch`), so the image builds without them and stays small.

# Keep in sync with .python-version and with uv in mise.toml.
FROM python:3.13.18-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# libgomp1: OpenMP runtime for torch. curl: HEALTHCHECK. ca-certificates: TLS downloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# ---------------------------------------------------------------------------------------------
# deps: third-party dependencies only, cached until pyproject.toml or uv.lock change
# ---------------------------------------------------------------------------------------------
FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra demo --no-install-project

# ---------------------------------------------------------------------------------------------
# app: the pollenformer package and the Streamlit demo (Hugging Face Spaces target)
# ---------------------------------------------------------------------------------------------
FROM base AS app
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra demo --no-editable
COPY demo ./demo

# Non-root user with a writable HOME: Hugging Face runs as uid 1000, and the torch hub code,
# the model artifacts and Streamlit all write under it.
ENV HOME=/home/user \
    TORCH_HOME=/home/user/.cache/torch \
    POLLENFORMER_HOME=/home/user/.cache/pollenformer \
    XDG_CACHE_HOME=/home/user/.cache \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN useradd --create-home --uid 1000 user \
    && chown -R 1000:0 /app /home/user \
    && chmod -R g=u /app /home/user
USER 1000

# Pre-fetch the pinned DINOv2 hub code (architecture only: the checkpoint holds every weight).
RUN python -c "from pollenformer.model import build_classifier; build_classifier()"

# Streamlit reads .streamlit/config.toml from the working directory.
WORKDIR /app/demo
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "pollenformer fetch && exec streamlit run main.py"]
