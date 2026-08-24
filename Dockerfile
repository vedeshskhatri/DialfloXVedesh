# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Dialflo Voice Attribute Inference Service
# ---------------------------------------------------------------------------
# Build: docker build -t dialflo-voice-attributes .
# Run:   docker compose up   (see docker-compose.yml)
#
# CPU-only target — GPU is an optimisation, not a requirement.
# The model weights (~1.2 GB) are baked into the image at build time so the
# running container has zero cold-start network dependency.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# System deps:
#   libsndfile1  → soundfile (audio decode)
#   ffmpeg       → fallback decode for exotic codecs (Opus, MP3, AMR)
#   git, curl    → download support & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Download model weights at BUILD TIME so the running container is air-gapped.
# "No external deps beyond publicly available model weights" — the weights are
# fetched once during build, not on every cold start.
# ---------------------------------------------------------------------------
COPY scripts/ ./scripts/
RUN python scripts/download_model.py

# ---------------------------------------------------------------------------
# Copy application source
# ---------------------------------------------------------------------------
COPY app/ ./app/

# Tell transformers to use the baked-in weights, not fetch from HF at runtime
ENV MODEL_PATH=/models/age-gender
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models
ENV HUGGINGFACE_HUB_VERBOSITY=warning

# Non-root user for security
RUN useradd -m -u 1000 dialflo && chown -R dialflo /app /models
USER dialflo

EXPOSE 8000

# Uvicorn: 1 worker per container (scale horizontally via compose replicas or k8s).
# --no-access-log: access logging is handled by our structured logger in
# logging_config.py — not the Uvicorn default which doesn't emit request_id.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
