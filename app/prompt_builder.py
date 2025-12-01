"""Prompt construction for fast, whimsical portrait generation."""
from __future__ import annotations

import random
from typing import Dict, Optional

STYLE_MAP: Dict[str, str] = {
    "happy": "radiant stained-glass glow, prismatic haloed skyline, floating lanterns",
    "angry": "crimson cyberpunk city lights, jagged chrome sculptures, kinetic shadows",
    "sad": "misty midnight alley with watercolor reflections, moonlit glass shards",
    "surprise": "bursting confetti nebulae, crystalline shards in mid-air, dynamic rim light",
    "fear": "haunted neon forest, bioluminescent fog, fractured silhouettes",
    "disgust": "acidic graffiti tunnels, warped mirrors, glitch textures",
    "neutral": "dreamy art gallery interior, soft volumetric light, floating gauze curtains",
    "cinematic": "cinematic dramatic portrait with volumetric lighting and film grain",
    "whimsical": "storybook fresco walls, pastel sparks, floating ink wisps",
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
    """Return a playful, background-forward prompt for the diffusion pipeline."""

    style = STYLE_MAP.get(style_key or "whimsical", STYLE_MAP["whimsical"])
    return (
        "a lively artistic portrait of the person, "
        f"{style}, {_random_tail()}, immersive painterly background, luminous atmosphere"
    )


class MoodStyleController:
    """Track mood-to-style transitions and produce stabilized prompts."""

    def __init__(self, transition_seconds: float = 10.0) -> None:
        self.transition_seconds = transition_seconds
        self._source_emotion = "neutral"
        self._target_emotion = "neutral"
        self._progress = 1.0
        self._last_update = None
        self._tail = _random_tail()

    def _blend_styles(self, style_key: str, target_key: str, weight: float) -> str:
        source_style = STYLE_MAP.get(style_key, STYLE_MAP["neutral"])
        target_style = STYLE_MAP.get(target_key, STYLE_MAP["neutral"])
        weight_pct = int(weight * 100)
        return (
            f"background is {100 - weight_pct}% {self._source_emotion} ({source_style}) and "
            f"{weight_pct}% {self._target_emotion} ({target_style}), ornate gallery ambience, luminous gradients"
        )

    def _advance(self) -> float:
        import time

        now = time.time()
        if self._last_update is None:
            self._last_update = now
            return self._progress

        elapsed = max(0.0, now - self._last_update)
        self._last_update = now
        increment = elapsed / max(0.5, self.transition_seconds)
        self._progress = min(1.0, self._progress + increment)
        if self._progress >= 1.0:
            self._source_emotion = self._target_emotion
        return self._progress

    def build_prompt(self, emotion: Optional[str], style_key: Optional[str] = None) -> str:
        """Return a flux diffusion prompt that eases between moods."""

        new_target = emotion or self._target_emotion or "neutral"
        if new_target != self._target_emotion:
            self._source_emotion = self._target_emotion
            self._target_emotion = new_target
            self._progress = 0.0
            self._tail = _random_tail()

        progress = self._advance()
        blend_style_key = style_key or self._source_emotion
        target_style_key = style_key or self._target_emotion
        blended_style = self._blend_styles(blend_style_key, target_style_key, progress)

        return (
            "a surreal artistic portrait of the person, "
            f"mood drifting toward {self._target_emotion}, {blended_style}, "
            f"{self._tail}, grand immersive background installation, cinematic glow"
        )


def build_prompt(emotion: str, style_key: Optional[str] = None) -> str:
    """Return a flux diffusion prompt given an emotion and style template."""

    style = STYLE_MAP.get(style_key or emotion, STYLE_MAP.get("cinematic", "cinematic dramatic portrait"))
    return (
        "a surreal artistic portrait of the person with "
        f"{emotion} mood, {style}, {_random_tail()}, dramatic art installation background, professional lighting"
    )


__all__ = [
    "MoodStyleController",
    "build_prompt",
    "build_whimsical_prompt",
    "STYLE_MAP",
]
