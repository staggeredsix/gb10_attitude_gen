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
from .face_segmentation import FaceSegmenter, SegmentationResult
from .image_generator import ImageGenerator
from .prompt_builder import build_prompt
from .ui import draw_face_overlays, overlay_no_face, show_window

LOGGER = logging.getLogger(__name__)


def _extract_face(frame: cv2.typing.MatLike, segmentation: SegmentationResult) -> Optional[np.ndarray]:
    face_crop = segmentation.crop_face(frame)
    if face_crop is None or face_crop.size == 0:
        return None
    return face_crop


def run(config: AppConfig) -> None:
    """Run the AI Mood Mirror application."""

    LOGGER.info("Starting AI Mood Mirror on %s", config.device)

    try:
        classifier = EmotionClassifier(config.emotion_model, config.device)
        segmenter = FaceSegmenter(
            config.face_segmentation_model, config.device, min_face_ratio=config.segmentation_min_area
        )
        diffusion_device = config.diffusion_device or config.device
        generator = ImageGenerator(
            config.diffusion_model, config.controlnet_model, diffusion_device
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to initialize models: %s", exc)
        return

    last_emotion: Optional[str] = None
    last_gen_time = 0.0
    generated_img: Optional[np.ndarray] = None

    try:
        with Camera(config.camera_index) as camera:
            while True:
                ret, frame = camera.read()
                if not ret:
                    LOGGER.warning("Failed to read frame from camera")
                    time.sleep(0.05)
                    continue

                segmentation = segmenter.segment(frame)
                emotion: Optional[str] = None
                face: Optional[np.ndarray] = None

                if segmentation:
                    face = _extract_face(frame, segmentation)
                    if face is not None:
                        emotion = classifier.classify(face)
                    frame = draw_face_overlays(frame, segmentation, emotion)
                else:
                    frame = overlay_no_face(frame)

                now = time.time()
                should_generate = (
                    emotion
                    and (emotion != last_emotion or now - last_gen_time > config.generation_interval)
                )

                if should_generate:
                    prompt = build_prompt(emotion)
                    generated = generator.generate(prompt, face, previous_output=generated_img)
                    if generated is not None:
                        generated_img = generated
                        last_emotion = emotion
                        last_gen_time = now

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
