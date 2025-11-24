"""Prompt construction for fast, whimsical portrait generation."""
from __future__ import annotations

import random
from typing import Dict, Optional

STYLE_MAP: Dict[str, str] = {
    "whimsical": "whimsical storybook brush strokes and pastel sparks",
    "cinematic": "cinematic dramatic portrait with volumetric lighting and film grain",
    "neon": "vibrant neon vaporwave edges with soft colorful glow",
    "surreal": "surreal collage fragments with gentle motion blur",
    "sketch": "illustrative line art with painterly gradients",
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


def build_whimsical_prompt(style_key: Optional[str] = None) -> str:
    """Return a playful, fast-to-generate prompt for the diffusion pipeline."""

    style = STYLE_MAP.get(style_key or "whimsical", STYLE_MAP["whimsical"])
    return (
        "a lively artistic portrait of the person, "
        f"{style}, {_random_tail()}, soft continuity between frames, minimal color swings"
    )


__all__ = ["build_whimsical_prompt", "STYLE_MAP"]
