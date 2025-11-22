FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://download.pytorch.org/whl/nightly/cu130 \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    HF_HOME=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface/hub

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        openssl \ 
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models
WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts/container_entrypoint.sh /usr/local/bin/container_entrypoint.sh

RUN chmod +x /usr/local/bin/container_entrypoint.sh \
    && python3 -m pip install --upgrade pip \
    && pip install .

VOLUME ["/models"]

ENTRYPOINT ["/usr/local/bin/container_entrypoint.sh"]
CMD ["ai-mood-mirror-web", "--host", "0.0.0.0", "--port", "8000"]
