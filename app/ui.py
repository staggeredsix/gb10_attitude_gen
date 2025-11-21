"""UI utilities for displaying frames."""
from __future__ import annotations

import cv2
import numpy as np

from .face_segmentation import SegmentationResult

# NVIDIA-inspired palette
ACCENT = (118, 185, 0)  # BGR green
BG = (24, 26, 27)
TEXT = (230, 235, 227)


def draw_face_overlays(
    frame: cv2.typing.MatLike, segmentation: SegmentationResult | None, emotion: str | None
) -> cv2.typing.MatLike:
    """Blend the face mask and annotate the frame."""
    overlay = frame.copy()
    if segmentation:
        mask_uint8 = (segmentation.mask.astype(np.uint8)) * 180
        tinted = np.zeros_like(frame)
        tinted[:, :] = ACCENT
        overlay = cv2.addWeighted(
            overlay, 1.0, cv2.merge([mask_uint8, mask_uint8, mask_uint8]), 0.25, 0
        )
        overlay = cv2.addWeighted(overlay, 1.0, tinted, 0.35, 0)
        x1, y1, x2, y2 = segmentation.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), ACCENT, 2)
    if emotion:
        cv2.putText(
            overlay,
            f"Emotion: {emotion}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            ACCENT,
            2,
            cv2.LINE_AA,
        )
    return overlay


def overlay_no_face(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Overlay a message when no face is detected."""
    message = "No face segmented"
    cv2.putText(frame, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame


def show_window(name: str, image: cv2.typing.MatLike) -> None:
    """Show an image in an OpenCV window."""
    cv2.imshow(name, image)


__all__ = ["draw_face_overlays", "overlay_no_face", "show_window"]
