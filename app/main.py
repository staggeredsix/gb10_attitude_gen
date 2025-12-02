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
from .emotion_classifier import EmotionClassifier
from .face_segmentation import FaceSegmenter, apply_subject_mask
from .generation_scheduler import AdaptiveGenerationScheduler
from .image_generator import ImageGenerator
from .prompt_builder import MoodStyleController
from .ui import show_window

LOGGER = logging.getLogger(__name__)


def run(config: AppConfig) -> None:
    """Run the AI Mood Mirror application."""

    LOGGER.info("Starting AI Mood Mirror on %s", config.device)

    try:
        classifier = EmotionClassifier(config.emotion_model, config.device)
        segmenter = FaceSegmenter(
            config.face_segmentation_model, config.device, config.segmentation_min_area
        )
        diffusion_device = config.diffusion_device or config.device
        generator = ImageGenerator(
            config.diffusion_model, config.controlnet_model, diffusion_device
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to initialize models: %s", exc)
        return

    generated_img: Optional[np.ndarray] = None
    last_mask: Optional[np.ndarray] = None
    reference_control: Optional[np.ndarray] = None
    generation_count = 0
    reference_reset_interval = 3
    style_controller = MoodStyleController(transition_seconds=10.0)

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
                    previous_reference_control = reference_control
                    refresh_reference = generation_count % reference_reset_interval == 0
                    control_image = None if refresh_reference else reference_control
                    if control_image is None and not refresh_reference:
                        control_image = generated_img

                    resized_frame = generator.resize_to_output(frame)
                    segmentation = segmenter.segment(resized_frame)
                    if segmentation is not None:
                        last_mask = segmentation.mask
                        masked_frame = apply_subject_mask(resized_frame, last_mask)
                    else:
                        masked_frame = resized_frame

                    emotion: Optional[str] = classifier.classify(masked_frame)
                    prompt = style_controller.build_prompt(emotion)
                    gen_start = time.time()
                    generated, returned_reference = generator.generate(
                        prompt,
                        masked_frame,
                        previous_output=generated_img,
                        reference_control=control_image,
                    )
                    scheduler.record_latency(time.time() - gen_start)
                    reference_control = (
                        returned_reference
                        if returned_reference is not None
                        else control_image or previous_reference_control
                    )
                    if generated is not None:
                        generation_count += 1
                        generated_img = generated

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
