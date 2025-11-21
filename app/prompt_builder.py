"""Prompt construction from detected emotions."""
from __future__ import annotations

from typing import Dict, Optional

STYLE_MAP: Dict[str, str] = {
    "happy": "vibrant neon vaporwave glowing edges soft colorful",
    "angry": "red black harsh contrast sharp geometry dystopian",
    "sad": "muted watercolor soft rain cold blue tones",
    "surprise": "radial distortion bloom scatter light explosive",
    "fear": "dark surreal twisted shadows fog",
    "disgust": "green distorted warped textures unsettling",
    "neutral": "balanced cinematic soft shadows realistic portrait",
    "cinematic": "cinematic dramatic portrait volumetric lighting film grain",
}


def build_prompt(emotion: str, style_key: Optional[str] = None) -> str:
    """Return a flux diffusion prompt given an emotion and style template."""

    style = STYLE_MAP.get(style_key or emotion, STYLE_MAP.get("cinematic", "cinematic dramatic portrait"))
    return (
        "a surreal artistic portrait of the person with "
        f"{emotion} mood, {style}, professional lighting, detailed face, futuristic aesthetic"
    )


__all__ = ["build_prompt", "STYLE_MAP"]
