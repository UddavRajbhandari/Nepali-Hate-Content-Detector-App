# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency installation
# Using a separate stage keeps the final image clean: build tools and pip cache
# are discarded, only the installed packages are copied forward.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile some Python deps (e.g. tokenizers Rust ext)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an explicit prefix so we can COPY just this directory in stage 2
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# ── System runtime dependencies ───────────────────────────────────────────────
# libgomp1        — required by PyTorch CPU backend (OpenMP)
# fonts-noto-core — base Noto fonts
# fonts-noto-extra — includes NotoSansDevanagari needed for Devanagari rendering
#                    in matplotlib (explainability charts)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        fonts-noto-core \
        fonts-noto-extra \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# ── Copy installed Python packages from builder ───────────────────────────────
COPY --from=builder /install /usr/local

# ── Copy application source ───────────────────────────────────────────────────
COPY app/ ./app/

# ── Copy your preprocessing scripts directory ─────────────────────────────────
# This is the scripts/ folder from your original Streamlit project containing:
#   transformer_data_preprocessing.py  (HateSpeechPreprocessor, EMOJI_TO_NEPALI)
#   explainability.py                  (LIMEExplainer, SHAPExplainer wrappers)
#   captum_explainer.py               (CaptumExplainer wrapper)
# Place this directory at backend/scripts/ before building.
COPY scripts/ ./scripts/

# ── Persistent data directory ─────────────────────────────────────────────────
# History is written here. On Railway/HF Spaces mount a volume at /app/data
# so predictions survive container restarts.
RUN mkdir -p data

# ── Pre-download the model at build time ──────────────────────────────────────
# Baking the model into the image means:
#   - Cold starts are instant (no 2.2 GB download on first request)
#   - The container works fully offline after deployment
#   - Image size increases by ~2.2 GB (acceptable for a dedicated backend)
#
# Set HF_MODEL_ID build arg to switch models without changing the Dockerfile.
# Set HF_TOKEN build arg if the model repo is private.
#
# To SKIP pre-downloading (e.g. for faster iteration during development),
# build with: docker build --build-arg SKIP_MODEL_DOWNLOAD=1 ...
ARG HF_MODEL_ID=UDHOV/xlm-roberta-large-nepali-hate-classification
ARG HF_TOKEN=""
ARG SKIP_MODEL_DOWNLOAD=""

RUN if [ -z "$SKIP_MODEL_DOWNLOAD" ]; then \
        python -c " \
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
import os; \
token = os.environ.get('HF_TOKEN') or None; \
AutoTokenizer.from_pretrained('${HF_MODEL_ID}', token=token); \
AutoModelForSequenceClassification.from_pretrained('${HF_MODEL_ID}', token=token); \
print('Model cached successfully.') \
        "; \
    else \
        echo "Skipping model download (SKIP_MODEL_DOWNLOAD is set)."; \
    fi

# ── Environment defaults ──────────────────────────────────────────────────────
# These can all be overridden at runtime via docker run -e or Railway variables.
ENV HF_MODEL_ID=${HF_MODEL_ID} \
    HISTORY_FILE=/app/data/prediction_history.jsonl \
    # Prevents Python from writing .pyc files into the image layers
    PYTHONDONTWRITEBYTECODE=1 \
    # Ensures print() and logging output appears immediately in docker logs
    PYTHONUNBUFFERED=1 \
    # Stops HuggingFace from showing progress bars that clutter docker logs
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Docker (and Railway) will mark the container as unhealthy if /health returns
# non-200 for 3 consecutive checks. The 60s start_period gives the model time
# to load before health checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# workers=1 is mandatory — the model is loaded into a single Python process.
# Multiple workers would each load their own 2.2 GB model copy, exhausting RAM.
# Horizontal scaling is done at the infrastructure level (multiple containers),
# not within a single container.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
