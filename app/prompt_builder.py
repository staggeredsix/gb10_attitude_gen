"""Prompt construction from detected emotions."""
from __future__ import annotations

import random
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

WHIMSICAL_SPINS = [
    "dreamlike doodles and playful symbols floating around",
    "whimsical storybook brush strokes and pastel sparks",
    "ethereal glow with floating origami creatures",
    "surreal collage fragments with gentle motion blur",
    "soft bokeh fireflies and iridescent mist",
]

ARTISANAL_DETAILS = [
    "handcrafted texture, artisan pigment, delicate etching",
    "illustrative line art with painterly gradients",
    "fine art studio lighting with cinematic rim light",
    "mixed media look with ink wash and neon pencil marks",
    "high-detail portrait lens with shimmering highlights",
]


def _random_tail() -> str:
    whimsical = random.choice(WHIMSICAL_SPINS)
    detail = random.choice(ARTISANAL_DETAILS)
    return f"{whimsical}, {detail}"


def build_prompt(emotion: str, style_key: Optional[str] = None) -> str:
    """Return a flux diffusion prompt given an emotion and style template."""

    style = STYLE_MAP.get(style_key or emotion, STYLE_MAP.get("cinematic", "cinematic dramatic portrait"))
    return (
        "a surreal artistic portrait of the person with "
        f"{emotion} mood, {style}, {_random_tail()}, professional lighting, detailed face, futuristic aesthetic"
    )


__all__ = ["build_prompt", "STYLE_MAP"]
