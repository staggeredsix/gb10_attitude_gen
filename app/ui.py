"""UI utilities for displaying frames."""
from __future__ import annotations

import cv2

# NVIDIA-inspired palette
ACCENT = (118, 185, 0)  # BGR green
BG = (24, 26, 27)
TEXT = (230, 235, 227)


def annotate_emotion(frame: cv2.typing.MatLike, emotion: str | None) -> cv2.typing.MatLike:
    """Annotate the frame with the current emotion label."""

    overlay = frame.copy()
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


def show_window(name: str, image: cv2.typing.MatLike) -> None:
    """Show an image in an OpenCV window."""
    cv2.imshow(name, image)


__all__ = ["annotate_emotion", "show_window"]
