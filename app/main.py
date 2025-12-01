"""Entry point for the AI Mood Mirror application."""
from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np

from .camera import Camera
from .config import AppConfig, load_config, parse_args
from .generation_scheduler import AdaptiveGenerationScheduler
from .image_generator import ImageGenerator
from .prompt_builder import build_whimsical_prompt
from .ui import show_window

LOGGER = logging.getLogger(__name__)


def run(config: AppConfig) -> None:
    """Run the AI Mood Mirror application."""

    LOGGER.info("Starting AI Mood Mirror on %s", config.device)

    try:
        diffusion_device = config.diffusion_device or config.device
        generator = ImageGenerator(
            config.diffusion_model, config.controlnet_model, diffusion_device
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to initialize models: %s", exc)
        return

    generated_img: Optional[np.ndarray] = None

    try:
        with Camera(config.camera_index) as camera:
            scheduler = AdaptiveGenerationScheduler(
                initial_interval=max(config.generation_interval, 0.1)
            )
            while True:
                ret, frame = camera.read()
                if not ret:
                    LOGGER.warning("Failed to read frame from camera")
                    time.sleep(0.05)
                    continue

                if scheduler.should_generate():
                    resized_frame = generator.resize_to_output(frame)
                    prompt = build_whimsical_prompt()
                    gen_start = time.time()
                    generated = generator.generate(
                        prompt, resized_frame, previous_output=generated_img
                    )
                    scheduler.record_latency(time.time() - gen_start)
                    if generated is not None:
                        generated_img = generator.upscale_for_display(
                            generated, frame.shape[:2]
                        )
                elif generated_img is not None:
                    generated_img = generator.upscale_for_display(
                        generated_img, frame.shape[:2]
                    )

                if config.show_windows:
                    show_window("Webcam", frame)
                    if generated_img is not None:
                        show_window("AI Mood Portrait", generated_img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        if config.show_windows:
            cv2.destroyAllWindows()
        LOGGER.info("Shutting down")


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args(argv)
    config = load_config(args)
    run(config)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
