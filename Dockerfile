FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.org/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --upgrade pip && \
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --extra-index-url https://pypi.org/simple && \
    pip install .

CMD ["ai-mood-mirror-web", "--host", "0.0.0.0", "--port", "8000"]
