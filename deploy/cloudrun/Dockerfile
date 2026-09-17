# Build ONLY from the extracted --target cloud-run bundle, not the raw repository.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080 \
    SCOUT_DEVICE=cpu \
    SPACES_ZERO_GPU=0 \
    SCOUT_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    SCOUT_MAX_INPUT_TOKENS=8192 \
    SCOUT_MAX_NEW_TOKENS=192 \
    LOL_UI_BACKEND=local \
    LOL_TWO_TOWER_DEVICE=cpu \
    RAG_EMBEDDING_DEVICE=cpu \
    RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    RAG_EMBEDDING_BATCH_SIZE=16 \
    LOL_CHROMA_DIR=/tmp/lol-chroma \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/hf-cache && chown appuser:appuser /opt/hf-cache

COPY requirements.txt ./
# Installing the matching CPU wheel first prevents pip pulling CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install -r requirements.txt \
    && pip check \
    && python -c "import torch; assert torch.version.cuda is None, 'CPU-only PyTorch required'"

COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser assets/ ./assets/
COPY --chown=appuser:appuser artifacts/ ./artifacts/
COPY --chown=appuser:appuser deployment/ ./deployment/
COPY --chown=appuser:appuser scripts/preload_models.py ./scripts/preload_models.py

USER appuser
RUN python -c "from pathlib import Path; from src.deployment import validate_bundle; validate_bundle(Path('/app'))"
# Downloads files only, during Cloud Build. No training or LLM inference.
RUN python scripts/preload_models.py

# Runtime must use the baked cache, not download models on visitors' requests.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8080
CMD ["python", "app.py"]
