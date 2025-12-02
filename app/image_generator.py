"""Diffusion-based image generator."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import sdnq  # noqa: F401  # Registers SDNQ quantization with diffusers and transformers
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
        dtypes = self._preferred_dtypes()
        self.num_inference_steps = 6
        self.guidance_scale = 3.5
        self.conditioning_scale = 0.9
        self.bundle = self._load_pipeline(model_name, controlnet_name, dtypes)

        self.output_size = self._determine_output_size()
        self.reference_control: np.ndarray | None = None

        controlnet_config = self.bundle.pipeline.controlnet.config
        is_union_model = bool(getattr(controlnet_config, "union", False)) or "union" in controlnet_name.lower()
        self.control_mode = 0 if is_union_model else None
        if self.control_mode is not None:
            LOGGER.info("ControlNet-Union detected; defaulting control_mode to canny (0)")

    def _preferred_dtypes(self) -> list[torch.dtype]:
        """Return a list of preferred dtypes from fastest to safest for the device."""

        if self.device.type == "cpu":
            LOGGER.warning(
                "Running diffusion pipeline on CPU; performance will be significantly degraded"
            )
            return [torch.float32]

        dtypes: list[torch.dtype] = []
        if self.device.type == "cuda":
            # Prefer the most aggressive precision available for throughput; fall back gracefully.
            if hasattr(torch.cuda, "is_fp8_available") and torch.cuda.is_fp8_available():
                dtypes.append(torch.float8_e4m3fn)
            dtypes.append(torch.float16)
            if torch.cuda.is_bf16_supported():
                dtypes.append(torch.bfloat16)
            return dtypes

        # MPS generally prefers float16 for performance.
        if self.device.type == "mps":
            dtypes.append(torch.float16)
        return dtypes or [torch.float16]

    def _load_pipeline(
        self, model_name: str, controlnet_name: str, dtypes: list[torch.dtype]
    ) -> _PipelineBundle:
        LOGGER.info("Loading ControlNet model: %s", controlnet_name)

        try:
            import bitsandbytes as bnb  # noqa: F401
        except ImportError:  # pragma: no cover - runtime dependency check
            LOGGER.warning("bitsandbytes is not installed; quantized FLUX models may fail to load")

        controlnet: FluxControlNetModel | None = None

        def _load_controlnet(dtype: torch.dtype) -> FluxControlNetModel:
            LOGGER.info("Loading ControlNet weights with dtype=%s", dtype)
            return FluxControlNetModel.from_pretrained(
                controlnet_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                # Community checkpoints may have mismatched shapes; allow loading with
                # random initialization for incompatible tensors to keep the app running.
                ignore_mismatched_sizes=True,
            )

        def _is_gguf_model(name: str) -> bool:
            lowered = name.lower().rstrip("/")
            return "gguf" in lowered

        def _build_pipeline(target_model: str, dtype: torch.dtype) -> _PipelineBundle:
            nonlocal controlnet
            if controlnet is None or controlnet.dtype != dtype:
                controlnet = _load_controlnet(dtype)

            if _is_gguf_model(target_model):
                raise ValueError(
                    "GGUF checkpoints are not compatible with the diffusers FLUX ControlNet pipeline"
                )
            LOGGER.info("Loading FLUX transformer weights from %s", target_model)
            transformer = FluxTransformer2DModel.from_pretrained(
                target_model,
                subfolder="transformer",
                torch_dtype=dtype,
                trust_remote_code=True,
                # Some FLUX community weights ship mismatched shapes; allow loading with
                # random init for incompatible tensors to keep the app running.
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

            LOGGER.info("Loading diffusion pipeline: %s on %s", target_model, self.device)
            pipe = FluxControlNetPipeline.from_pretrained(
                target_model,
                controlnet=controlnet,
                transformer=transformer,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:  # noqa: BLE001
                LOGGER.debug("xFormers attention could not be enabled", exc_info=True)
            if self.device.type == "cpu":
                pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                else:
                    LOGGER.debug("Pipeline does not support VAE slicing; skipping enable_vae_slicing")
            else:
                LOGGER.debug("Skipping attention/vae slicing to maximize GPU throughput")
            pipe.set_progress_bar_config(disable=True)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True

            LOGGER.info("Pushing FLUX pipeline to %s with dtype=%s", self.device, dtype)
            pipe.to(device=self.device, dtype=dtype)
            LOGGER.info("FLUX ControlNet pipeline is ready on %s", self.device)

            return _PipelineBundle(pipeline=pipe, dtype=dtype)

        last_err: Exception | None = None
        for dtype in dtypes:
            try:
                return _build_pipeline(model_name, dtype)
            except (RuntimeError, ValueError) as err:
                last_err = err
                message = str(err).lower()
                fallback_model = "black-forest-labs/FLUX.1-schnell"
                should_retry = (
                    ("size mismatch" in message or "mismatched" in message or "gguf" in message)
                    and model_name != fallback_model
                )
                if should_retry:
                    LOGGER.warning(
                        "Failed to load diffusion model '%s' due to incompatible or unsupported weights; retrying with fallback '%s'",
                        model_name,
                        fallback_model,
                    )
                    try:
                        return _build_pipeline(fallback_model, dtype)
                    except Exception as retry_err:  # noqa: BLE001
                        last_err = retry_err
                        LOGGER.debug("Fallback model load failed with dtype %s", dtype, exc_info=True)
                LOGGER.warning(
                    "Pipeline load failed with dtype %s; trying next precision option if available",
                    dtype,
                )

        if last_err:
            raise last_err
        raise RuntimeError("Unable to initialize diffusion pipeline")

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

    def _determine_output_size(self) -> tuple[int, int]:
        """Infer the pipeline's output resolution for control image resizing."""

        processor = getattr(self.bundle.pipeline, "image_processor", None)
        size = getattr(processor, "size", None)
        if isinstance(size, dict):
            width = size.get("width") or size.get("shortest_edge")
            height = size.get("height") or size.get("shortest_edge")
            if width and height:
                return int(width), int(height)

        transformer = getattr(self.bundle.pipeline, "transformer", None)
        sample_size = getattr(getattr(transformer, "config", None), "sample_size", None)
        if sample_size:
            return int(sample_size), int(sample_size)

        LOGGER.warning("Unable to determine pipeline output size; defaulting to 1024x1024")
        return 1024, 1024

    def resize_to_output(self, frame: np.ndarray) -> np.ndarray:
        """Resize an input frame to match the diffusion output resolution."""

        width, height = self.output_size
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _prepare_control_image(
        self, init_image: np.ndarray, previous_output: np.ndarray | None
    ) -> Image.Image:
        if self.reference_control is None:
            self.reference_control = init_image.copy()

        blurred = cv2.GaussianBlur(self.reference_control, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        control = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(control)

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
            control_pil = self._prepare_control_image(init_image, previous_output)
            pipeline_kwargs = dict(
                prompt=prompt,
                control_image=control_pil,
                control_mode=self.control_mode,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.conditioning_scale,
                output_type="np",
            )

            result = self.bundle.pipeline(**pipeline_kwargs)
            image = result.images[0]

        if not np.isfinite(image).all():
            LOGGER.warning("Generated image contained non-finite values; sanitizing output")
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

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
