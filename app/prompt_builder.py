"""Prompt construction for fast, whimsical portrait generation."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

STYLE_MAP: Dict[str, List[str]] = {
    "happy": [
        "sunlit oil pastel glow, honey-gold rim light, swirling coral and rose backdrop",
        "radiant spring studio light, citrus and blush palette, soft bloom on the cheeks",
        "warm morning skylight, peach haze, impressionist dashes of pink and apricot",
    ],
    "angry": [
        "deep carmine and indigo storm light, molten copper edges, sharp chiaroscuro shadows",
        "embers and iron lighting, scarlet reflections on glossy paint strokes",
        "rusted industrial glow, crimson neon flares, gritty chiaroscuro",
    ],
    "sad": [
        "ink-washed twilight corridor, muted teal and violet haze, gentle rain reflections",
        "overcast dusk window light, desaturated blues, quiet watercolor drips",
        "foggy alleyway sheen, pewter and lilac patina, soft backlit silhouette",
    ],
    "surprise": [
        "electric amethyst flare, prismatic sparks, kinetic brush strokes frozen in air",
        "staccato neon streaks, cyan and violet bursts, suspended droplets of paint",
        "crystal refractions, scattered color prisms, sudden flares of magenta light",
    ],
    "fear": [
        "nocturnal forest silhouettes, cyan bioluminescent mist, fractured moonbeams",
        "ashen moonlit fog, cobalt backlight, tangled branches and distant embers",
        "cold underlit portrait, teal glints, fractured glass reflections on the skin",
    ],
    "disgust": [
        "acid-green neon graffiti, warped chrome reflections, grimy cinematic smoke",
        "murky subway sodium light, toxic slime reflections, smeared oil paint",
        "chartreuse and rust glow, rough metal textures, glitchy urban grime",
    ],
    "neutral": [
        "soft museum gallery ambience, dove-gray silk drapery, balanced studio glow",
        "even skylight diffusion, porcelain gradients, uncluttered background",
        "calm archival studio, linen backdrop, subtle tonal falloff",
    ],
    "cinematic": [
        "dramatic portrait stage with volumetric key light, fine grain, razor focus on the eyes",
        "arthouse portrait lighting, sweeping shadows, subtle film grain and bloom",
        "silver screen glow, balanced contrast, polished editorial sheen",
    ],
    "whimsical": [
        "storybook fresco walls, pastel embers, floating ink wisps and chalk constellations",
        "candy-lacquered fresco, dreamy firefly lights, watercolor nebula haze",
        "hand-painted mural ambience, soft pastel fog, dancing chalk sparkles",
    ],
    "neon": [
        "vibrant vaporwave bloom, magenta and cyan ribbons, luminous rim edges",
        "electric dusk arcade, neon ribbons, chromatic aberration shimmer",
        "holographic studio glow, ultraviolet haze, glowing edge outlines",
    ],
    "surreal": [
        "collage of painted nebulae and cracked marble, slow-motion pigment trails",
        "impossible cathedral of color, melting fresco arches, shimmering dust",
        "floating staircases of paint, cosmic plaster textures, dreamlike gradients",
    ],
    "sketch": [
        "charcoal and colored pencil crosshatching, subtle watercolor fill, expressive strokes",
        "graphite sketchbook shading, paper texture, loose hatching and pastel washes",
        "ink linework with soft gouache fill, gestural strokes, tactile paper grain",
    ],
}

WHIMSICAL_SPINS = [
    "aurora-swept night sky curling behind the subject",
    "cosmic oil paint swirls orbiting the silhouette",
    "hand-tinted etching with glowing dust motes and tiny stars",
    "velvet darkness split by rivers of molten color",
    "quiet storm of soft light leaks and drifting pigments",
]

ARTISANAL_DETAILS = [
    "impasto texture, expressive brushwork, visible pigment ridges",
    "gallery-grade studio lighting with cinematic rim glow",
    "medium format portrait lens feel, sharp eyes and creamy falloff",
    "mixed media layering of oil, pastel, and charcoal, nuanced gradients",
    "painterly grain with subtle bloom and analog richness",
]

SUBJECT_FOCUS = [
    "intimate head-and-shoulders framing, direct gaze, vivid facial texture, strong cheekbone light",
    "half-length portrait with open posture, confident gaze, crisp focus on the eyes",
    "close portrait crop, soft jawline light, delicate catchlights, expressive eyebrows",
]

TEXTURE_GARNISH = [
    "velvet bokeh, translucent veils of color, painterly microcontrast",
    "floating paper scraps, glittering dust motes, luminous mist",
    "subtle film halation, glassy reflections, layered mixed media glaze",
]

BACKDROP_MOTIFS = [
    "gilded gallery mural made of layered brushwork and light leaks",
    "floating canvas shards in a softly lit studio atrium",
    "luminous abstract fresco with drifting pigment clouds",
    "dreamy museum alcove with hand-painted gradients and ink wisps",
    "large-scale mixed media tapestry glowing with cinematic rim lights",
]


def _trim_prompt(prompt: str, max_tokens: int = 70) -> str:
    """Trim prompts to a CLIP-friendly token budget to avoid truncation warnings."""

    tokens = prompt.split()
    if len(tokens) <= max_tokens:
        return prompt

    trimmed = " ".join(tokens[:max_tokens])
    return trimmed.rstrip(",;")


def _random_tail() -> str:
    whimsical = random.choice(WHIMSICAL_SPINS)
    detail = random.choice(ARTISANAL_DETAILS)
    texture = random.choice(TEXTURE_GARNISH)
    return f"{whimsical}, {detail}, {texture}"


def _random_focus() -> str:
    return random.choice(SUBJECT_FOCUS)


def _random_backdrop() -> str:
    return random.choice(BACKDROP_MOTIFS)



def _resolve_style(style_key: str) -> str:
    styles = STYLE_MAP.get(style_key)
    if not styles:
        styles = STYLE_MAP["neutral"]
    return random.choice(styles)


def build_whimsical_prompt(style_key: Optional[str] = None) -> str:
    """Return a playful, background-forward prompt for the diffusion pipeline."""

    style = _resolve_style(style_key or "whimsical")
    prompt = (
        "an emotive painterly portrait of the person with the subject crisply segmented from the background, "
        f"{_random_focus()}, {style}, {_random_tail()}, {_random_backdrop()}, background painted separately, no paint on the face, luminous atmosphere"
    )
    return _trim_prompt(prompt)


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
        source_style = _resolve_style(style_key)
        target_style = _resolve_style(target_key)
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

        prompt = (
            "a surreal, highly detailed portrait of the person with a clean subject mask, "
            f"{_random_focus()}, mood drifting toward {self._target_emotion}, {blended_style}, "
            f"{self._tail}, {_random_backdrop()}, artistic backdrop only, no low-poly rendering, cinematic glow"
        )
        return _trim_prompt(prompt)


def build_prompt(emotion: str, style_key: Optional[str] = None) -> str:
    """Return a flux diffusion prompt given an emotion and style template."""

    style = _resolve_style(style_key or emotion)
    prompt = (
        "a vivid, textured portrait of the person with a separate, imaginative background and "
        f"{emotion} mood, {_random_focus()}, {style}, {_random_tail()}, {_random_backdrop()}, subject remains natural while the background is painterly, professional lighting"
    )
    return _trim_prompt(prompt)


__all__ = [
    "MoodStyleController",
    "build_prompt",
    "build_whimsical_prompt",
    "STYLE_MAP",
]
