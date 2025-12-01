"""Adaptive scheduler to throttle image generation based on latency."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class AdaptiveGenerationScheduler:
    """Self-tuning interval tracker for image generation.

    The scheduler starts with a conservative interval and blends in the
    observed latency for each generation call. This avoids launching
    expensive diffusion steps back-to-back when they take longer than
    expected while still allowing the app to speed up when generation
    is consistently fast.
    """

    initial_interval: float = 7.0
    min_interval: float = 1.5
    max_interval: float = 12.0
    smoothing: float = 0.2
    overhead_ratio: float = 0.25

    def __post_init__(self) -> None:
        self.interval = min(self.max_interval, max(self.min_interval, self.initial_interval))
        # Seed the clock so that the first generation waits for the initial interval.
        self._last_timestamp = time.time()

    def should_generate(self, now: float | None = None) -> bool:
        """Return True when enough time has elapsed to trigger generation."""

        now = time.time() if now is None else now
        return (now - self._last_timestamp) >= self.interval

    def record_latency(self, latency_s: float) -> None:
        """Update the interval based on the most recent generation latency."""

        target = max(latency_s * (1 + self.overhead_ratio), self.min_interval)
        blended = (1 - self.smoothing) * self.interval + self.smoothing * target
        self.interval = min(self.max_interval, max(self.min_interval, blended))
        # Anchor the interval to the generation *start* by rewinding the clock
        # by the time spent rendering. This keeps the start-to-start cadence
        # aligned with the blended interval instead of double-counting latency.
        self._last_timestamp = time.time() - latency_s
