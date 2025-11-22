"""Emotion analysis powered by a lightweight vision-language model."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

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

_SYNONYM_MAP: Dict[str, str] = {
    "anger": "angry",
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "joy": "happy",
    "joyful": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "surprised": "surprise",
    "surprise": "surprise",
    "calm": "neutral",
    "neutral": "neutral",
}


class EmotionClassifier:
    """Use a VLM to map face crops to a discrete emotion label."""

    def __init__(self, model_name: str, device: str) -> None:
        if device not in {"cuda", "mps", "cpu"}:
            LOGGER.error("Emotion VLM requires a valid device; got %s", device)
            raise RuntimeError("Unsupported device for emotion analysis")

        if device == "cuda" and not torch.cuda.is_bf16_supported():
            LOGGER.info("BF16 not supported on this CUDA device; using float16 for emotion VLM")
            dtype = torch.float16
        elif device == "cpu":
            LOGGER.warning("Running emotion classifier on CPU; performance may be degraded")
            dtype = torch.float32
        else:
            dtype = torch.float16 if device == "mps" else torch.bfloat16
        self.device = device

        LOGGER.info("Loading emotion VLM: %s on %s", model_name, device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

    def _parse_label(self, response: str) -> Optional[str]:
        cleaned = response.strip().lower()
        for token in cleaned.replace("\n", " ").split():
            normalized = token.strip(" ,.;!?")
            if normalized in _SYNONYM_MAP:
                return _SYNONYM_MAP[normalized]
        for label in DEFAULT_LABELS:
            if label in cleaned:
                return label
        return None

    def classify(self, face_img: cv2.typing.MatLike) -> Optional[str]:
        """Return the predicted emotion for the provided face image."""
        try:
            resized = cv2.resize(face_img, (448, 448))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            prompt = (
                "You are a concise emotion rater. Look at the face and choose the single "
                "emotion from this list: angry, disgust, fear, happy, sad, surprise, neutral. "
                "Respond with only the label."
            )
            inputs = self.processor(images=rgb, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=8)
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return self._parse_label(text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Emotion VLM classification failed: %s", exc)
            return None


__all__ = ["EmotionClassifier", "DEFAULT_LABELS"]
