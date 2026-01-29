import logging
import os
from pathlib import Path

LOGGER = logging.getLogger("settings_loader")
_LOADED_KEYS: set[str] = set()


def _parse_settings(path: Path) -> tuple[dict[str, str], int | None]:
    settings: dict[str, str] = {}
    inference_steps: int | None = None
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip().lower() or None
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if current_section == "inference" and key.lower() == "steps":
            try:
                inference_steps = int(value)
            except ValueError:
                LOGGER.warning("Invalid [inference].steps=%s; ignoring.", value)
            continue
        if current_section is None:
            settings[key] = value
    return settings, inference_steps


def load_settings_conf(*, override: bool = False) -> None:
    settings_path = Path(__file__).resolve().parent / "settings.conf"
    if not settings_path.is_file():
        return
    try:
        settings, inference_steps = _parse_settings(settings_path)
        # Remove stale keys that were previously set by this loader.
        for key in list(_LOADED_KEYS):
            if key not in settings:
                os.environ.pop(key, None)
                _LOADED_KEYS.discard(key)
        for key, value in settings.items():
            if key in os.environ and key not in _LOADED_KEYS and not override:
                continue
            os.environ[key] = value
            _LOADED_KEYS.add(key)
        if inference_steps is not None:
            os.environ["LTX2_REALTIME_STEPS"] = str(inference_steps)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to load settings.conf")
