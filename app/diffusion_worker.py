"""Diffusion-only worker for dual-Spark deployments."""
from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import AppConfig, ENV_PREFIX, load_config, parse_args
from .image_generator import ImageGenerator

LOGGER = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    mood: Optional[str] = None
    reference_image_b64: Optional[str] = None
    control_image_b64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


def _decode_image_b64(value: Optional[str]) -> Optional[np.ndarray]:
    if not value:
        return None
    try:
        data = base64.b64decode(value)
        np_data = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to decode base64 image: %s", exc)
        return None


def _encode_image_b64(image: np.ndarray) -> Optional[str]:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        return None
    return base64.b64encode(buffer).decode("ascii")


def _select_diffusion_device(config: AppConfig) -> str:
    return (
        os.getenv(f"{ENV_PREFIX}DIFFUSION_DEVICE")
        or os.getenv("AI_MOOD_MIRROR_DIFFUSION_DEVICE")
        or config.diffusion_device
        or config.device
        or "cuda:0"
    )


def create_worker_app(config: AppConfig) -> FastAPI:
    app = FastAPI()
    diffusion_device = _select_diffusion_device(config)
    generator = ImageGenerator(config.diffusion_model, config.controlnet_model, diffusion_device)
    LOGGER.info("Diffusion worker ready on %s", diffusion_device)

    @app.post("/generate")
    async def generate(request: GenerationRequest) -> dict[str, str]:
        if request.seed is not None:
            torch.manual_seed(request.seed)

        init_image = _decode_image_b64(request.reference_image_b64)
        control_image = _decode_image_b64(request.control_image_b64)

        if init_image is None and control_image is None:
            raise HTTPException(status_code=400, detail="reference_image_b64 or control_image_b64 required")

        if request.width and request.height:
            generator.output_size = (int(request.width), int(request.height))
            if init_image is not None:
                init_image = cv2.resize(init_image, (int(request.width), int(request.height)), interpolation=cv2.INTER_AREA)
            if control_image is not None:
                control_image = cv2.resize(
                    control_image, (int(request.width), int(request.height)), interpolation=cv2.INTER_AREA
                )

        generated, _reference_control = generator.generate(
            request.prompt,
            init_image,
            previous_output=None,
            reference_control=control_image,
        )
        if generated is None:
            raise HTTPException(status_code=500, detail="generation_failed")

        encoded = _encode_image_b64(generated)
        if encoded is None:
            raise HTTPException(status_code=500, detail="encode_failed")

        return {"image_b64": encoded}

    return app


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args(argv)
    config = load_config(args)
    app = create_worker_app(config)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
