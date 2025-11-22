"""Face segmentation utilities using a lightweight transformer model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForImageSegmentation,
    AutoProcessor,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Structured output describing a segmented face."""

    mask: np.ndarray  # boolean mask (H, W) where face pixels are True
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    def crop_face(self, frame_bgr: cv2.typing.MatLike) -> Optional[np.ndarray]:
        """Return a face crop using the bounding box with the mask applied."""
        x1, y1, x2, y2 = self.bbox
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        mask_roi = self.mask[y1:y2, x1:x2]
        masked = cv2.bitwise_and(roi, roi, mask=(mask_roi.astype(np.uint8) * 255))
        return masked


class FaceSegmenter:
    """Segment faces using a GPU-backed semantic segmentation model."""

    def __init__(self, model_name: str, device: str, min_face_ratio: float = 0.01) -> None:
        if device not in {"cuda", "mps", "cpu"}:
            LOGGER.error("Face segmentation requires a valid device; got %s", device)
            raise RuntimeError("Unsupported device for segmentation")

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; falling back to CPU for segmentation")
            device = "cpu"

        if device == "mps" and not torch.backends.mps.is_available():
            LOGGER.warning("MPS requested but not available; falling back to CPU for segmentation")
            device = "cpu"

        self.device = device
        self.min_face_ratio = min_face_ratio
        self.processor = None
        self.model = None
        self._cascade: Optional[cv2.CascadeClassifier] = None

        if device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.float16 if device == "mps" or not torch.cuda.is_bf16_supported() else torch.bfloat16

        LOGGER.info("Loading face segmentation model: %s on %s", model_name, device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)
            LOGGER.info("Face segmentation model ready: %s", model_name)
        except Exception as err:  # noqa: BLE001
            LOGGER.error(
                "Failed to load %s via Transformers; falling back to OpenCV face detection: %s",
                model_name,
                err,
            )
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(cascade_path)
                if self._cascade.empty():
                    raise RuntimeError("Failed to load Haar cascade for fallback segmentation")
                LOGGER.info("Loaded OpenCV Haar cascade fallback for face detection")
            except Exception as cascade_err:  # noqa: BLE001
                LOGGER.error("Fallback face detection unavailable: %s", cascade_err)

    def segment(self, frame_bgr: cv2.typing.MatLike) -> Optional[SegmentationResult]:
        """Return a segmentation mask for the most prominent face."""
        if self.model and self.processor:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            mask = upsampled.argmax(dim=1)[0].detach().cpu().numpy()
            face_mask = mask != 0
        elif self._cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detections = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(detections) == 0:
                return None
            x, y, w, h = max(detections, key=lambda box: box[2] * box[3])
            mask = np.zeros(frame_bgr.shape[:2], dtype=bool)
            mask[y : y + h, x : x + w] = True
            face_mask = mask
        else:
            return None

        if not np.any(face_mask):
            LOGGER.debug("Segmentation did not find a face region")
            return None

        y_indices, x_indices = np.where(face_mask)
        x1, x2 = int(x_indices.min()), int(x_indices.max())
        y1, y2 = int(y_indices.min()), int(y_indices.max())

        height, width = frame_bgr.shape[:2]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if area / float(width * height) < self.min_face_ratio:
            LOGGER.debug("Discarding segmentation with insufficient area (%.5f)", area / float(width * height))
            return None

        return SegmentationResult(mask=face_mask, bbox=(x1, y1, x2, y2))


__all__ = ["FaceSegmenter", "SegmentationResult"]
