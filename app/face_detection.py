"""Face detection utilities using MediaPipe."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp

LOGGER = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Simple bounding box representation."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> "BoundingBox":
        """Clip the bounding box to the given image dimensions."""
        return BoundingBox(
            x1=max(0, min(self.x1, width)),
            y1=max(0, min(self.y1, height)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
        )


class FaceDetector:
    """Detect faces in frames using MediaPipe."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )
        LOGGER.info("MediaPipe face detector initialized (min_confidence=%.2f)", min_confidence)

    def detect(self, frame_bgr: cv2.typing.MatLike) -> List[BoundingBox]:
        """Detect faces and return bounding boxes."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        if not result.detections:
            return []

        height, width = frame_bgr.shape[:2]
        boxes: List[BoundingBox] = []
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * width)
            y1 = int(bbox.ymin * height)
            x2 = int((bbox.xmin + bbox.width) * width)
            y2 = int((bbox.ymin + bbox.height) * height)
            boxes.append(BoundingBox(x1, y1, x2, y2).clip(width, height))
        return boxes

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.detector.close()


__all__ = ["FaceDetector", "BoundingBox"]
