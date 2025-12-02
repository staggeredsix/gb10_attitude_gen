"""Cluster-related helpers for AI Mood Mirror."""
from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from .config import ClusterConfig, ClusterMode

LOGGER = logging.getLogger(__name__)


def generate_remote_image(cluster_cfg: ClusterConfig, payload: dict[str, Any]) -> bytes:
    """Request image generation from the secondary Spark diffusion worker."""

    if cluster_cfg.mode != ClusterMode.DUAL:
        raise ValueError("Remote generation only available when CLUSTER_MODE=dual")

    cluster_cfg.validate()
    url = f"http://{cluster_cfg.second_spark_ip}:9000/generate"
    LOGGER.info("Requesting remote diffusion from %s", url)
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    image_b64 = data.get("image_b64")
    if not image_b64:
        raise ValueError("Remote worker response missing 'image_b64'")

    return base64.b64decode(image_b64)


__all__ = ["generate_remote_image"]
