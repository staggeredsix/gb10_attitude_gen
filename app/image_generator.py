"""Diffusion-based image generator."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

LOGGER = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images from text prompts using a diffusion pipeline."""

    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        LOGGER.info("Loading diffusion pipeline: %s on %s", model_name, device)
        kwargs = {"dtype": dtype}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_name,
            **kwargs,
        ).to(self.device)

    def generate(self, prompt: str, init_image: np.ndarray | None) -> Optional[np.ndarray]:
        """Generate an image for the given prompt and return a BGR numpy array."""

        if init_image is None:
            LOGGER.debug("No init image provided for generation")
            return None

        LOGGER.info("Generating image for prompt: %s", prompt)
        try:
            autocast_ctx = torch.autocast(device_type=self.device, dtype=torch.float16) if self.device == "cuda" else nullcontext()
            with autocast_ctx:
                init_pil = Image.fromarray(cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB))
                image = self.pipe(
                    prompt=prompt,
                    image=init_pil,
                    strength=0.6,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                ).images[0]
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Image generation failed: %s", exc)
            return None


__all__ = ["ImageGenerator"]
