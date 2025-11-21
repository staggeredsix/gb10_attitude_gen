FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://download.pytorch.org/whl/nightly/cu130 \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --upgrade pip && \
    pip install . && \
    python3 - <<'PY'
from huggingface_hub import snapshot_download

models = [
    ("emotion", "Qwen/Qwen2-VL-2B-Instruct"),
    ("diffusion", "black-forest-labs/FLUX.1-schnell"),
    ("controlnet", "InstantX/FLUX.1-dev-Controlnet-Union"),
    ("face-segmentation", "briaai/RMBG-1.4"),
]

for label, repo in models:
    print(f"[preload] downloading {label}: {repo}")
    snapshot_download(repo_id=repo, local_files_only=False)
PY

CMD ["ai-mood-mirror-web", "--host", "0.0.0.0", "--port", "8000"]
