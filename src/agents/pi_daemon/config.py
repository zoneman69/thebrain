from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PiDaemonConfig:
    sample_interval_seconds: float = 2.0
    frame_path: str | None = None
    spec_path: str | None = None          # NEW: spectrogram file path
    camera_index: int | str = 0
    camera_retry_attempts: int = 3
    camera_retry_delay_seconds: float = 2.0
    audio_device: str | int | None = None
    replay_dir: str = "/var/lib/thebrain/replay"
    episode_batch_size: int = 256
    max_replay_files: int = 10


_ENV_KEYS = {
    "sample_interval_seconds": "PI_SAMPLE_INTERVAL_SECONDS",
    "frame_path": "PI_FRAME_PATH",
    "spec_path": "PI_SPEC_PATH",          # NEW
    "camera_index": "PI_CAMERA_INDEX",
    "camera_retry_attempts": "PI_CAMERA_RETRY_ATTEMPTS",
    "camera_retry_delay_seconds": "PI_CAMERA_RETRY_DELAY_SECONDS",
    "audio_device": "PI_AUDIO_DEVICE",
    "replay_dir": "PI_REPLAY_DIR",
    "episode_batch_size": "PI_EPISODE_BATCH_SIZE",
    "max_replay_files": "PI_MAX_REPLAY_FILES",
}


def _parse_env_value(key: str, default: object) -> object:
    value = os.getenv(_ENV_KEYS[key])
    if value is None:
        # Allow the daemon to piggy-back on the feed/UI frame & spec files
        # without extra config. HIPPO_FRAME/HIPPO_SPEC are set in systemd
        # units and default to /tmp/hippo_latest.jpg / /tmp/hippo_spec.png.
        if key == "frame_path":
            value = os.getenv("HIPPO_FRAME")
        elif key == "spec_path":
            value = os.getenv("HIPPO_SPEC")
        else:
            value = None
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if key == "camera_index":
        # Allow either a numeric index (e.g. 0, 1) or a device path such as
        # "/dev/video2" for USB webcams that do not map to the default index.
        try:
            return int(value)
        except ValueError:
            return value
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
