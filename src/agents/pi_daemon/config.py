from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PiDaemonConfig:
    sample_interval_seconds: float = 2.0
    camera_index: int = 0
    audio_device: str | int | None = None
    replay_dir: str = "/var/lib/thebrain/replay"
    episode_batch_size: int = 256
    max_replay_files: int = 10


_ENV_KEYS = {
    "sample_interval_seconds": "PI_SAMPLE_INTERVAL_SECONDS",
    "camera_index": "PI_CAMERA_INDEX",
    "audio_device": "PI_AUDIO_DEVICE",
    "replay_dir": "PI_REPLAY_DIR",
    "episode_batch_size": "PI_EPISODE_BATCH_SIZE",
    "max_replay_files": "PI_MAX_REPLAY_FILES",
}


def _parse_env_value(key: str, default: object) -> object:
    value = os.getenv(_ENV_KEYS[key])
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    return value


def load_config() -> PiDaemonConfig:
    defaults = PiDaemonConfig()
    values: dict[str, object] = {}
    for field_name, env_key in _ENV_KEYS.items():
        default = getattr(defaults, field_name)
        values[field_name] = _parse_env_value(field_name, default)
    return PiDaemonConfig(**values)
