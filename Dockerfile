FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface/hub

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models
WORKDIR /app

COPY requirements.txt README.md app.py ltx2_backend.py ./
COPY static ./static

# Install your app deps (should NOT include torch; if it does, remove it)
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

# Clone LTX-2 monorepo and install subpackages WITHOUT deps (prevents torch downgrade/replacement)
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && python3 -m pip install --no-deps -e /app/LTX-2/packages/ltx-core \
    && python3 -m pip install --no-deps -e /app/LTX-2/packages/ltx-pipelines \
    && python3 - <<'PY'
import torch
print("Torch kept:", torch.__version__)
PY

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

VOLUME ["/models"]

CMD ["python3", "-m", "app", "--host", "0.0.0.0", "--port", "8000"]
