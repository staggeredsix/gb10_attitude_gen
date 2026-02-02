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
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models
WORKDIR /app

COPY requirements.txt README.md app.py ltx2_backend.py settings_loader.py settings.conf ./
COPY scripts/patch_ltx2_cache_models.py /app/patch_ltx2_cache_models.py
COPY scripts/patch_ltx2_distilled_cfg.py /app/patch_ltx2_distilled_cfg.py
COPY static ./static

# Install your app deps (should NOT include torch; if it does, remove it)
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

# Clone LTX-2 monorepo and install subpackages WITHOUT deps (prevents torch downgrade/replacement)
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && python3 /app/patch_ltx2_cache_models.py /app/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py \
    && python3 /app/patch_ltx2_distilled_cfg.py /app/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py \
    && python3 -m pip install -e /app/LTX-2/packages/ltx-core \
    && python3 -m pip install -e /app/LTX-2/packages/ltx-pipelines 

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

VOLUME ["/models"]

CMD ["python3", "-m", "app", "--host", "0.0.0.0", "--port", "8000"]
