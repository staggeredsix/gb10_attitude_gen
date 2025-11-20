"""Emotion classification utilities."""
from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

LOGGER = logging.getLogger(__name__)

DEFAULT_LABELS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


class EmotionClassifier:
    """Classify emotions from face crops."""

    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        LOGGER.info("Loading emotion classifier: %s on %s", model_name, device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)

    def classify(self, face_img: cv2.typing.MatLike) -> Optional[str]:
        """Return the predicted emotion for the provided face image."""
        try:
            resized = cv2.resize(face_img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inputs = self.image_processor(images=rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            idx = logits.softmax(dim=-1).argmax().item()
            return DEFAULT_LABELS[idx] if idx < len(DEFAULT_LABELS) else None
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Emotion classification failed: %s", exc)
            return None


__all__ = ["EmotionClassifier", "DEFAULT_LABELS"]
