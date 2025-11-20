"""Diffusion-based image generator."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image

LOGGER = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images from text prompts using a diffusion pipeline."""

    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        LOGGER.info("Loading diffusion pipeline: %s on %s", model_name, device)
        kwargs = {"torch_dtype": dtype}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_name,
            **kwargs,
        ).to(self.device)

    def generate(self, prompt: str) -> Optional[np.ndarray]:
        """Generate an image for the given prompt and return a BGR numpy array."""
        LOGGER.info("Generating image for prompt: %s", prompt)
        try:
            autocast_ctx = torch.autocast(device_type=self.device, dtype=torch.float16) if self.device == "cuda" else nullcontext()
            with autocast_ctx:
                image = self.pipe(
                    prompt,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                ).images[0]
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Image generation failed: %s", exc)
            return None


__all__ = ["ImageGenerator"]
