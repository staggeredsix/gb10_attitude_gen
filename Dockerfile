FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface/hub

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models
WORKDIR /app

COPY requirements.txt README.md app.py ltx2_backend.py ./
COPY static ./static

RUN python3 -m pip install --upgrade pip \
    && pip install -r requirements.txt

RUN mkdir -p /app/packages \
    && git clone https://github.com/Lightricks/ltx-core.git /app/packages/ltx-core \
    && git clone https://github.com/Lightricks/ltx-pipelines.git /app/packages/ltx-pipelines \
    && python3 -m pip install -e /app/packages/ltx-core -e /app/packages/ltx-pipelines

VOLUME ["/models"]

CMD ["python3", "-m", "app", "--host", "0.0.0.0", "--port", "8000"]
