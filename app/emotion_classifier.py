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
        if device not in {"cuda", "mps"}:
            LOGGER.error("Emotion classification requires a GPU; device '%s' is unsupported", device)
            raise RuntimeError("Emotion classifier requires GPU")

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.error("CUDA requested for emotion classifier but not available")
            raise RuntimeError("CUDA device not available for emotion classifier")

        if device == "mps" and not torch.backends.mps.is_available():
            LOGGER.error("MPS requested for emotion classifier but not available")
            raise RuntimeError("MPS device not available for emotion classifier")

        self.device = device
        LOGGER.info("Loading emotion classifier: %s on %s", model_name, device)
        dtype = torch.float16
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

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
