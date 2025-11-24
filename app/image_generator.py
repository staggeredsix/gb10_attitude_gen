"""Diffusion-based image generator."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import FluxControlNetModel, FluxControlNetPipeline, FluxTransformer2DModel
from diffusers.quantizers.auto import DiffusersAutoQuantizer
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class _PipelineBundle:
    pipeline: FluxControlNetPipeline
    dtype: torch.dtype


class ImageGenerator:
    """Generate images from text prompts using a diffusion pipeline conditioned on webcam input."""

    def __init__(self, model_name: str, controlnet_name: str, device: str) -> None:
        resolved = torch.device(device)

        self._install_quantization_guard()

        if resolved.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; falling back to CPU for diffusion")
            resolved = torch.device("cpu")

        if resolved.type == "cuda" and not self._is_cuda_usable(resolved):
            LOGGER.warning("CUDA device not usable for diffusion; falling back to CPU")
            resolved = torch.device("cpu")

        if resolved.type == "mps" and not torch.backends.mps.is_available():
            LOGGER.warning("MPS requested but not available; falling back to CPU for diffusion")
            resolved = torch.device("cpu")

        if resolved.type not in {"cuda", "mps", "cpu"}:
            LOGGER.error("Diffusion requires a valid device; got %s", device)
            raise RuntimeError("Unsupported device for diffusion")

        self.device = resolved
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            LOGGER.info("BF16 not supported on this CUDA device; using float16 for diffusion")
            dtype = torch.float16
        elif self.device.type == "cpu":
            LOGGER.warning("Running diffusion pipeline on CPU; performance will be significantly degraded")
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float16
        self.num_inference_steps = 4
        self.guidance_scale = 3.0
        self.conditioning_scale = 0.85
        self.bundle = self._load_pipeline(model_name, controlnet_name, dtype)

        controlnet_config = self.bundle.pipeline.controlnet.config
        is_union_model = bool(getattr(controlnet_config, "union", False)) or "union" in controlnet_name.lower()
        self.control_mode = 0 if is_union_model else None
        if self.control_mode is not None:
            LOGGER.info("ControlNet-Union detected; defaulting control_mode to canny (0)")

    def _load_pipeline(self, model_name: str, controlnet_name: str, dtype: torch.dtype) -> _PipelineBundle:
        LOGGER.info("Loading ControlNet model: %s", controlnet_name)
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_name, torch_dtype=dtype, trust_remote_code=True
        )

        try:
            import bitsandbytes as bnb  # noqa: F401
        except ImportError:  # pragma: no cover - runtime dependency check
            LOGGER.warning("bitsandbytes is not installed; quantized FLUX models may fail to load")

        LOGGER.info("Loading FLUX transformer weights from %s", model_name)
        transformer = FluxTransformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        LOGGER.info("Loading diffusion pipeline: %s on %s", model_name, self.device)
        pipe = FluxControlNetPipeline.from_pretrained(
            model_name,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=dtype,
            trust_remote_code=True,
        )


        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:  # noqa: BLE001
            LOGGER.debug("xFormers attention could not be enabled", exc_info=True)
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        else:
            LOGGER.debug("Pipeline does not support VAE slicing; skipping enable_vae_slicing")
        pipe.set_progress_bar_config(disable=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        LOGGER.info("Pushing FLUX pipeline to %s with dtype=%s", self.device, dtype)
        pipe.to(device=self.device, dtype=dtype)
        LOGGER.info("FLUX ControlNet pipeline is ready on %s", self.device)

        return _PipelineBundle(pipeline=pipe, dtype=dtype)

    def _install_quantization_guard(self) -> None:
        original_merge = DiffusersAutoQuantizer.merge_quantization_configs
        original_from_config = DiffusersAutoQuantizer.from_config

        def _safe_merge(cls, quantization_config, quantization_config_from_args):
            try:
                return original_merge(quantization_config, quantization_config_from_args)
            except ValueError as err:
                message = str(err)
                if "Unknown quantization type" in message:
                    LOGGER.warning(
                        "Ignoring unsupported quantization config '%s'; loading model without quantization",
                        getattr(quantization_config, "get", lambda key, default=None: None)(
                            "quant_method", None
                        ),
                    )
                    return None
                raise

        DiffusersAutoQuantizer.merge_quantization_configs = classmethod(_safe_merge)

        def _safe_from_config(cls, quantization_config, **kwargs):
            if quantization_config is None:
                LOGGER.warning(
                    "Quantization config missing or incomplete; loading model without quantization"
                )
                return None

            quant_method = (
                quantization_config.get("quant_method")
                if isinstance(quantization_config, dict)
                else getattr(quantization_config, "quant_method", None)
            )
            if quant_method is None:
                LOGGER.warning(
                    "Quantization config missing or incomplete; loading model without quantization"
                )
                return None
            return original_from_config.__func__(cls, quantization_config, **kwargs)

        DiffusersAutoQuantizer.from_config = classmethod(_safe_from_config)

    @staticmethod
    def _is_cuda_usable(device: torch.device) -> bool:
        try:
            torch.empty(1, device=device).mul_(1)
            return True
        except RuntimeError as err:
            message = str(err).lower()
            if "no kernel image is available" in message or "cuda error" in message:
                LOGGER.debug("CUDA usability check failed: %s", err)
                return False
            raise

    def _move_to_cpu(self) -> None:
        LOGGER.warning("Retrying diffusion on CPU after GPU failure")
        self.device = torch.device("cpu")
        self.bundle.dtype = torch.float32
        self.bundle.pipeline.to(device=self.device, dtype=self.bundle.dtype)

    @staticmethod
    def _prepare_control_image(init_image: np.ndarray) -> Image.Image:
        blurred = cv2.GaussianBlur(init_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)
        control = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(control)

    def _blend_frames(self, new_image: np.ndarray, previous_image: np.ndarray | None) -> np.ndarray:
        if previous_image is None:
            return new_image

        if previous_image.shape != new_image.shape:
            previous_image = cv2.resize(previous_image, (new_image.shape[1], new_image.shape[0]))

        blend_ratio = 0.35  # prioritize prior frame to maintain visual continuity
        return cv2.addWeighted(new_image, blend_ratio, previous_image, 1 - blend_ratio, 0)

    def _should_fallback_to_cpu(self, exc: Exception) -> bool:
        if self.device.type == "cpu":
            return False
        message = str(exc).lower()
        return "no kernel image is available" in message or "cuda error" in message

    def _generate_once(
        self, prompt: str, init_image: np.ndarray, previous_output: np.ndarray | None = None
    ) -> np.ndarray:
        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.bundle.dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            control_pil = self._prepare_control_image(init_image)
            result = self.bundle.pipeline(
                prompt=prompt,
                control_image=control_pil,
                control_mode=self.control_mode,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.conditioning_scale,
                output_type="np",
            )
            image = result.images[0]

        if not np.isfinite(image).all():
            LOGGER.warning("Generated image contained non-finite values; sanitizing output")
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        return self._blend_frames(cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR), previous_output)

    def generate(
        self, prompt: str, init_image: np.ndarray | None, previous_output: np.ndarray | None = None
    ) -> Optional[np.ndarray]:
        """Generate an image for the given prompt and return a BGR numpy array."""

        if init_image is None:
            LOGGER.debug("No init image provided for generation")
            return None

        LOGGER.info("Generating image for prompt: %s", prompt)
        try:
            return self._generate_once(prompt, init_image, previous_output)
        except RuntimeError as exc:
            if self._should_fallback_to_cpu(exc):
                self._move_to_cpu()
                try:
                    return self._generate_once(prompt, init_image, previous_output)
                except Exception as retry_exc:  # noqa: BLE001
                    LOGGER.exception(
                        "Image generation failed after CPU fallback: %s", retry_exc
                    )
                    return None
            LOGGER.exception("Image generation failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Image generation failed: %s", exc)
            return None


__all__ = ["ImageGenerator"]
