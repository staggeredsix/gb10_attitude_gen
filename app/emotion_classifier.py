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

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested for emotion VLM but unavailable; using CPU instead")
            device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            LOGGER.warning("MPS requested for emotion VLM but unavailable; using CPU instead")
            device = "cpu"

        if device == "cuda" and not torch.cuda.is_bf16_supported():
            LOGGER.info("BF16 not supported on this CUDA device; using float16 for emotion VLM")
            dtype = torch.float16
        elif device == "cpu":
            LOGGER.warning("Running emotion classifier on CPU; performance may be degraded")
            dtype = torch.float32
        else:
            dtype = torch.float16 if device == "mps" else torch.bfloat16
        self.device = device

        def _load(target_device: str, target_dtype: torch.dtype) -> AutoModelForVision2Seq:
            return AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=target_dtype,
            ).to(target_device)

        LOGGER.info("Loading emotion VLM: %s on %s", model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.model = _load(self.device, dtype)
        except RuntimeError as err:
            if self.device != "cpu" and "no kernel image is available for execution" in str(err).lower():
                LOGGER.warning("Falling back to CPU for emotion VLM due to CUDA kernel issue: %s", err)
                self.device = "cpu"
                self.model = _load(self.device, torch.float32)
            else:
                raise

        self._dtype = torch.float32 if self.device == "cpu" else dtype

    def _should_fallback_to_cpu(self, exc: Exception) -> bool:
        if self.device == "cpu":
            return False
        message = str(exc).lower()
        return "no kernel image is available" in message or "cuda error" in message

    def _classify_once(self, face_img: cv2.typing.MatLike) -> Optional[str]:
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

    def _move_to_cpu(self) -> None:
        LOGGER.warning("Retrying emotion VLM on CPU after GPU failure")
        self.device = "cpu"
        self._dtype = torch.float32
        self.model = self.model.to(device=self.device, dtype=self._dtype)

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
            return self._classify_once(face_img)
        except RuntimeError as exc:
            if self._should_fallback_to_cpu(exc):
                self._move_to_cpu()
                try:
                    return self._classify_once(face_img)
                except Exception as retry_exc:  # noqa: BLE001
                    LOGGER.exception("Emotion VLM classification failed after CPU fallback: %s", retry_exc)
                    return None
            LOGGER.exception("Emotion VLM classification failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Emotion VLM classification failed: %s", exc)
            return None


__all__ = ["EmotionClassifier", "DEFAULT_LABELS"]
