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
from .face_detection import BoundingBox, FaceDetector
from .image_generator import ImageGenerator
from .prompt_builder import build_prompt
from .ui import draw_face_overlays, overlay_no_face, show_window

LOGGER = logging.getLogger(__name__)


def _extract_face(frame: cv2.typing.MatLike, box: BoundingBox) -> Optional[np.ndarray]:
    face_crop = frame[box.y1 : box.y2, box.x1 : box.x2]
    if face_crop.size == 0:
        return None
    return face_crop


def run(config: AppConfig) -> None:
    """Run the AI Mood Mirror application."""

    LOGGER.info("Starting AI Mood Mirror on %s", config.device)

    try:
        classifier = EmotionClassifier(config.emotion_model, config.device)
        detector = FaceDetector(min_confidence=config.detection_confidence)
        generator = ImageGenerator(config.diffusion_model, config.device)
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

                boxes = detector.detect(frame)
                emotion: Optional[str] = None
                face: Optional[np.ndarray] = None

                if boxes:
                    face = _extract_face(frame, boxes[0])
                    if face is not None:
                        emotion = classifier.classify(face)
                    frame = draw_face_overlays(frame, boxes, emotion)
                else:
                    frame = overlay_no_face(frame)

                now = time.time()
                should_generate = (
                    emotion
                    and (emotion != last_emotion or now - last_gen_time > config.generation_interval)
                )

                if should_generate:
                    prompt = build_prompt(emotion)
                    generated = generator.generate(prompt, face)
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
        try:
            detector.close()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Detector close failed", exc_info=True)
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
