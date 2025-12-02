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
        """Return a face crop using an expanded bounding box with the mask applied."""
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.bbox
        box_w = x2 - x1
        box_h = y2 - y1

        # Expand the crop by 300% in both dimensions to capture more of the user's body
        expand_x = box_w
        expand_y = box_h
        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(width, x2 + expand_x)
        y2 = min(height, y2 + expand_y)

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        mask_roi = self.mask[y1:y2, x1:x2]
        masked = cv2.bitwise_and(roi, roi, mask=(mask_roi.astype(np.uint8) * 255))
        return masked


def apply_subject_mask(frame_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Apply a soft subject mask to remove the background."""

    if mask is None:
        return frame_bgr

    if mask.shape[:2] != frame_bgr.shape[:2]:
        mask = cv2.resize(mask.astype(np.float32), (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_float = mask.astype(np.float32)
    softened = cv2.GaussianBlur(mask_float, (9, 9), 0)
    normalized = np.clip(softened, 0.0, 1.0)
    mask_uint8 = (normalized * 255).astype(np.uint8)

    foreground = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_uint8)
    background = np.zeros_like(frame_bgr)
    return cv2.add(foreground, background)


def _expand_mask_to_upper_body(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Expand a face mask downward to include head, shoulders, and upper body."""

    x1, y1, x2, y2 = bbox
    height, width = mask.shape[:2]

    face_w = x2 - x1
    face_h = y2 - y1
    # Extend the mask downward to capture shoulders/upper torso while widening the sides.
    shoulder_extra_h = int(face_h * 1.8)
    shoulder_half_w = int(face_w * 0.9)
    center_x = (x1 + x2) // 2

    body_x1 = max(0, center_x - shoulder_half_w)
    body_x2 = min(width, center_x + shoulder_half_w)
    body_y1 = y1
    body_y2 = min(height, y2 + shoulder_extra_h)

    expanded = np.zeros_like(mask, dtype=bool)
    expanded[body_y1:body_y2, body_x1:body_x2] = True

    # Union the original mask and the expanded torso region, then smooth.
    combined = np.logical_or(mask, expanded)
    dilated = cv2.dilate(combined.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2)
    softened = cv2.GaussianBlur(dilated.astype(np.float32), (7, 7), 0)
    return softened > 0.25


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
            self.processor = self._load_processor(model_name)
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

    @staticmethod
    def _load_processor(model_name: str):
        """Load a compatible image processor for the segmentation model."""

        attempts = (
            ("AutoProcessor", lambda: AutoProcessor.from_pretrained(model_name, trust_remote_code=True)),
            ("AutoImageProcessor", lambda: AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)),
            ("AutoFeatureExtractor", lambda: AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)),
        )

        errors: list[str] = []
        for loader_name, loader in attempts:
            try:
                processor = loader()
                LOGGER.info("Loaded %s for face segmentation model", loader_name)
                return processor
            except Exception as err:  # noqa: BLE001
                LOGGER.debug("%s unavailable for %s: %s", loader_name, model_name, err)
                errors.append(f"{loader_name}: {err}")

        raise RuntimeError("No compatible processor found; tried " + "; ".join(errors))

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

        # Expand to head-and-shoulders coverage to avoid central-box crops.
        expanded_mask = _expand_mask_to_upper_body(face_mask, (x1, y1, x2, y2))
        y_exp, x_exp = np.where(expanded_mask)
        x1, x2 = int(x_exp.min()), int(x_exp.max())
        y1, y2 = int(y_exp.min()), int(y_exp.max())

        height, width = frame_bgr.shape[:2]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if area / float(width * height) < self.min_face_ratio:
            LOGGER.debug("Discarding segmentation with insufficient area (%.5f)", area / float(width * height))
            return None

        return SegmentationResult(mask=expanded_mask, bbox=(x1, y1, x2, y2))


__all__ = ["FaceSegmenter", "SegmentationResult", "apply_subject_mask"]
