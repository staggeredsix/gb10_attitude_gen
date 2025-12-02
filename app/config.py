"""Configuration management for AI Mood Mirror application."""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch


ENV_PREFIX = "AI_MOOD_MIRROR_"
LOGGER = logging.getLogger(__name__)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    """Runtime configuration for the application."""

    camera_index: int = 0
    emotion_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    diffusion_model: str = "city96/FLUX.2-dev-gguf/flux2-dev-Q4_K_S.gguf"
    controlnet_model: str = "InstantX/FLUX.1-dev-Controlnet-Union"
    face_segmentation_model: str = "briaai/RMBG-1.4"
    segmentation_min_area: float = 0.01
    generation_interval: float = 7.0
    diffusion_device: Optional[str] = None
    use_cuda: bool = True
    show_windows: bool = True
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    default_mode: str = "single"
    enable_https: bool = True
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    @property
    def device(self) -> str:
        """Return the torch device string based on availability and config.

        The application requires a GPU-backed device. If initialization fails,
        the caller must handle the RuntimeError and exit gracefully.
        """
        if not self.use_cuda:
            LOGGER.warning("CUDA usage disabled via configuration; using CPU")
            return "cpu"

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            LOGGER.info("CUDA available: %s (compute capability %s.%s)", name, *capability)
            return "cuda"

        if torch.backends.mps.is_available():
            LOGGER.info("Metal Performance Shaders backend detected; using MPS")
            return "mps"

        LOGGER.warning("No GPU detected, get into container and troubleshoot")
        return "cpu"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the application."""

    parser = argparse.ArgumentParser(description="AI Mood Mirror")
    parser.add_argument("--camera-index", type=int, help="Index of the webcam to use")
    parser.add_argument("--emotion-model", type=str, help="Hugging Face model id for emotion classification")
    parser.add_argument("--diffusion-model", type=str, help="Model id for diffusion image generation")
    parser.add_argument(
        "--controlnet-model",
        type=str,
        help="Model id for the ControlNet used to condition generation on the webcam image",
    )
    parser.add_argument(
        "--diffusion-device",
        type=str,
        help="Torch device string for running FLUX diffusion (e.g., cuda:1 for dedicated GPU)",
    )
    parser.add_argument("--face-segmentation-model", type=str, help="Model id for face segmentation")
    parser.add_argument(
        "--segmentation-min-area",
        type=float,
        help="Minimum area ratio (0-1) for accepting a segmentation mask",
    )
    parser.add_argument("--generation-interval", type=float, help="Seconds between portrait generations")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", help="Force use of CUDA if available")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", help="Disable CUDA usage")
    parser.add_argument("--no-ui", dest="show_windows", action="store_false", help="Disable OpenCV windows (headless)")
    parser.add_argument("--host", type=str, help="Host for the web UI server")
    parser.add_argument("--port", type=int, help="Port for the web UI server")
    parser.add_argument(
        "--default-mode",
        type=str,
        choices=["single", "dual"],
        help="Default inference mode to present in the web UI",
    )
    parser.add_argument(
        "--https",
        dest="enable_https",
        action="store_true",
        help="Serve the web UI over HTTPS (self-signed certs generated if missing)",
    )
    parser.add_argument(
        "--no-https",
        dest="enable_https",
        action="store_false",
        help="Disable HTTPS and serve over HTTP",
    )
    parser.add_argument("--ssl-certfile", type=str, help="Path to an SSL certificate file")
    parser.add_argument("--ssl-keyfile", type=str, help="Path to an SSL private key file")
    parser.set_defaults(use_cuda=None, show_windows=True, enable_https=None)
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> AppConfig:
    """Create an AppConfig from CLI args and environment variables."""

    env_camera = os.getenv(f"{ENV_PREFIX}CAMERA_INDEX")
    env_emotion_model = os.getenv(f"{ENV_PREFIX}EMOTION_MODEL")
    env_diffusion_model = os.getenv(f"{ENV_PREFIX}DIFFUSION_MODEL")
    env_controlnet_model = os.getenv(f"{ENV_PREFIX}CONTROLNET_MODEL")
    env_face_seg_model = os.getenv(f"{ENV_PREFIX}FACE_SEGMENTATION_MODEL")
    env_seg_min_area = os.getenv(f"{ENV_PREFIX}SEGMENTATION_MIN_AREA")
    env_gen_interval = os.getenv(f"{ENV_PREFIX}GENERATION_INTERVAL")
    env_diffusion_device = os.getenv(f"{ENV_PREFIX}DIFFUSION_DEVICE")
    env_use_cuda = _bool_env(f"{ENV_PREFIX}USE_CUDA", True)
    env_host = os.getenv(f"{ENV_PREFIX}HOST")
    env_port = os.getenv(f"{ENV_PREFIX}PORT")
    env_default_mode = os.getenv(f"{ENV_PREFIX}DEFAULT_MODE") or os.getenv("DEFAULT_MODE")
    env_enable_https = _bool_env(f"{ENV_PREFIX}ENABLE_HTTPS", True)
    env_ssl_certfile = os.getenv(f"{ENV_PREFIX}SSL_CERTFILE")
    env_ssl_keyfile = os.getenv(f"{ENV_PREFIX}SSL_KEYFILE")
    role_hint = os.getenv("ROLE")

    camera_index = args.camera_index if args.camera_index is not None else int(env_camera) if env_camera else 0
    emotion_model = args.emotion_model if args.emotion_model else env_emotion_model or "Qwen/Qwen2-VL-2B-Instruct"
    diffusion_model = (
        args.diffusion_model
        if args.diffusion_model
        else env_diffusion_model
        or "city96/FLUX.2-dev-gguf/flux2-dev-Q4_K_S.gguf"
    )
    controlnet_model = args.controlnet_model if args.controlnet_model else env_controlnet_model or "InstantX/FLUX.1-dev-Controlnet-Union"
    face_seg_model = (
        args.face_segmentation_model
        if args.face_segmentation_model
        else env_face_seg_model
        if env_face_seg_model
        else "briaai/RMBG-1.4"
    )
    segmentation_min_area = (
        args.segmentation_min_area
        if args.segmentation_min_area is not None
        else float(env_seg_min_area)
        if env_seg_min_area
        else 0.01
    )
    generation_interval = (
        args.generation_interval
        if args.generation_interval is not None
        else float(env_gen_interval) if env_gen_interval else 7.0
    )
    diffusion_device = args.diffusion_device if args.diffusion_device else env_diffusion_device

    use_cuda = env_use_cuda if args.use_cuda is None else args.use_cuda
    show_windows = args.show_windows
    server_host = args.host if args.host else env_host or "0.0.0.0"
    server_port = args.port if args.port is not None else int(env_port) if env_port else 8000
    default_mode = args.default_mode or env_default_mode or ("dual" if role_hint in {"vision", "diffusion", "dual"} else "single")
    enable_https = env_enable_https if args.enable_https is None else args.enable_https
    ssl_certfile = args.ssl_certfile if args.ssl_certfile else env_ssl_certfile
    ssl_keyfile = args.ssl_keyfile if args.ssl_keyfile else env_ssl_keyfile

    return AppConfig(
        camera_index=camera_index,
        emotion_model=emotion_model,
        diffusion_model=diffusion_model,
        controlnet_model=controlnet_model,
        face_segmentation_model=face_seg_model,
        segmentation_min_area=segmentation_min_area,
        generation_interval=generation_interval,
        diffusion_device=diffusion_device,
        use_cuda=use_cuda,
        show_windows=show_windows,
        server_host=server_host,
        server_port=server_port,
        default_mode=default_mode,
        enable_https=enable_https,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


__all__ = ["AppConfig", "parse_args", "load_config"]
