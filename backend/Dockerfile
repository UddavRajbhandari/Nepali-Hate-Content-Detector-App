# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency installation
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements correctly (from repo root context)
COPY backend/requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        fontconfig \
        fonts-noto-core \
        fonts-noto-extra \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy backend app
COPY backend/app/ ./app/

# Copy scripts (important for your preprocessing + XAI)
COPY scripts/ ./scripts/

# Create data directory
RUN mkdir -p /app/data

# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration
# ─────────────────────────────────────────────────────────────────────────────
ENV HF_MODEL_ID=UDHOV/xlm-roberta-large-nepali-hate-classification \
    HISTORY_FILE=/app/data/prediction_history.jsonl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# Expose port
EXPOSE 8000

# Healthcheck (important for Railway)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]