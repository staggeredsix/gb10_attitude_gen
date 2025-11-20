"""Camera utilities for capturing frames from a webcam."""
from __future__ import annotations

import logging
from typing import Tuple

import cv2

LOGGER = logging.getLogger(__name__)


class Camera:
    """Wrapper around OpenCV VideoCapture with graceful error handling."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            LOGGER.error("Cannot open webcam at index %s", self.index)
            raise RuntimeError(f"Cannot open webcam at index {self.index}")
        LOGGER.info("Opened webcam at index %s", self.index)

    def read(self) -> Tuple[bool, cv2.typing.MatLike]:
        """Read a frame from the camera."""
        return self.cap.read()

    def release(self) -> None:
        """Release the camera resource."""
        if self.cap:
            self.cap.release()
            LOGGER.info("Camera released")

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


__all__ = ["Camera"]
