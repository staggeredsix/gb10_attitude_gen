"""UI utilities for displaying frames."""
from __future__ import annotations

import cv2

from .face_detection import BoundingBox


def draw_face_overlays(frame: cv2.typing.MatLike, boxes: list[BoundingBox], emotion: str | None) -> cv2.typing.MatLike:
    """Draw bounding boxes and emotion label on the frame."""
    for box in boxes:
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
    if emotion:
        cv2.putText(frame, f"Emotion: {emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def overlay_no_face(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Overlay a message when no face is detected."""
    message = "No face detected"
    cv2.putText(frame, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame


def show_window(name: str, image: cv2.typing.MatLike) -> None:
    """Show an image in an OpenCV window."""
    cv2.imshow(name, image)


__all__ = ["draw_face_overlays", "overlay_no_face", "show_window"]
