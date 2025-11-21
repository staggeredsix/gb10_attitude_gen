"""Diffusion-based image generator."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class _PipelineBundle:
    pipeline: StableDiffusionControlNetImg2ImgPipeline
    dtype: torch.dtype


class ImageGenerator:
    """Generate images from text prompts using a diffusion pipeline conditioned on webcam input."""

    def __init__(self, model_name: str, controlnet_name: str, device: str) -> None:
        if device not in {"cuda", "mps"}:
            LOGGER.error("Image generation requires a GPU; device '%s' is unsupported", device)
            raise RuntimeError("Image generation requires a GPU")

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.error("CUDA requested but not available; cannot initialize diffusion pipeline")
            raise RuntimeError("CUDA device not available")

        if device == "mps" and not torch.backends.mps.is_available():
            LOGGER.error("MPS requested but not available; cannot initialize diffusion pipeline")
            raise RuntimeError("MPS device not available")

        self.device = device
        dtype = torch.float16
        self.bundle = self._load_pipeline(model_name, controlnet_name, dtype)

    def _load_pipeline(self, model_name: str, controlnet_name: str, dtype: torch.dtype) -> _PipelineBundle:
        LOGGER.info("Loading ControlNet model: %s", controlnet_name)
        controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=dtype)

        LOGGER.info("Loading diffusion pipeline: %s on %s", model_name, self.device)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_name,
            controlnet=controlnet,
            torch_dtype=dtype,
        )


        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:  # noqa: BLE001
            LOGGER.debug("xFormers attention could not be enabled", exc_info=True)
        pipe.to(device=self.device, dtype=dtype)

        return _PipelineBundle(pipeline=pipe, dtype=dtype)

    @staticmethod
    def _prepare_control_image(init_image: np.ndarray) -> Image.Image:
        blurred = cv2.GaussianBlur(init_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)
        control = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(control)

    def generate(self, prompt: str, init_image: np.ndarray | None) -> Optional[np.ndarray]:
        """Generate an image for the given prompt and return a BGR numpy array."""

        if init_image is None:
            LOGGER.debug("No init image provided for generation")
            return None

        LOGGER.info("Generating image for prompt: %s", prompt)
        try:
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.bundle.dtype)
                if self.device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                init_pil = Image.fromarray(cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB))
                control_pil = self._prepare_control_image(init_image)
                result = self.bundle.pipeline(
                    prompt=prompt,
                    image=init_pil,
                    control_image=control_pil,
                    strength=0.5,
                    num_inference_steps=10,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=1.0,
                    output_type="np",
                )
                image = result.images[0]

            if not np.isfinite(image).all():
                LOGGER.warning("Generated image contained non-finite values; sanitizing output")
                image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

            image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            return cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Image generation failed: %s", exc)
            return None


__all__ = ["ImageGenerator"]
