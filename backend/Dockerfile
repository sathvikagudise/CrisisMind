FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

COPY . .

# Materialise the Git-LFS model checkpoints (no-op if already checked out).
# Fails the build if any deployed checkpoint is still an LFS pointer / missing.
RUN git config --global --add safe.directory /app \
    && (git -C /app lfs pull || echo "git lfs pull skipped") \
    && for f in \
         backend/outputs/checkpoints/best_model.pth \
         backend/outputs/checkpoints/best_text_model.pth \
         backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth ; do \
         size=$(stat -c%s "$f"); \
         echo "$f -> $size bytes"; \
         test "$size" -gt 1000000 || (echo "ERROR: $f is an LFS pointer / missing" && exit 1); \
       done

# CPU-only PyTorch wheels keep the image lean for Render.
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
         -r backend/requirements.txt

EXPOSE 8000
CMD uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
