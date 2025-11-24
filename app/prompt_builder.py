"""Prompt construction from detected emotions with gradual style transitions."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
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


@dataclass
class MoodStyleController:
    """Track mood-to-style transitions and produce stabilized prompts."""

    transition_seconds: float = 10.0
    _source_emotion: str = "neutral"
    _target_emotion: str = "neutral"
    _progress: float = 1.0
    _last_update: float = field(default_factory=time.time)
    _tail: str = field(default_factory=_random_tail)

    def _blend_styles(self, style_key: str, target_key: str, weight: float) -> str:
        """Return a descriptive blend of the current and target styles."""

        source_style = STYLE_MAP.get(style_key, STYLE_MAP["neutral"])
        target_style = STYLE_MAP.get(target_key, STYLE_MAP["neutral"])
        weight_pct = int(weight * 100)
        return (
            f"palette is {100 - weight_pct}% {self._source_emotion} ({source_style}) and "
            f"{weight_pct}% {self._target_emotion} ({target_style}), gentle hue roll-off, stable lighting"
        )

    def _advance(self) -> float:
        now = time.time()
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
            f"{self._tail}, soft continuity between frames, minimal color swings"
        )


def build_prompt(emotion: str, style_key: Optional[str] = None) -> str:
    """Return a flux diffusion prompt given an emotion and style template."""

    style = STYLE_MAP.get(style_key or emotion, STYLE_MAP.get("cinematic", "cinematic dramatic portrait"))
    return (
        "a surreal artistic portrait of the person with "
        f"{emotion} mood, {style}, {_random_tail()}, professional lighting, detailed face, futuristic aesthetic"
    )


__all__ = ["build_prompt", "STYLE_MAP", "MoodStyleController"]
