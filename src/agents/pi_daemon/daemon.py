from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_config
from .encoders import build_default_audio_encoder, build_default_vision_encoder
from .episodes import Episode, ReplayWriter, episode_to_replay
from .sensors import CameraSensor, FrameFileSensor, MicrophoneSensor

logger = logging.getLogger(__name__)


def run_daemon(iterations: int | None = None) -> None:
    cfg = load_config()
    logger.info("Starting Pi daemon with replay_dir=%s", cfg.replay_dir)

    # Vision source: either shared frame file (from thebrain-feed) or direct camera
    if cfg.frame_path:
        cam = FrameFileSensor(
            frame_path=Path(cfg.frame_path),
            retry_attempts=cfg.camera_retry_attempts,
            retry_delay_seconds=cfg.camera_retry_delay_seconds,
        )
        logger.info("Using frame file source at %s", cfg.frame_path)
    else:
        cam = CameraSensor(
            camera_index=cfg.camera_index,
            retry_attempts=cfg.camera_retry_attempts,
            retry_delay_seconds=cfg.camera_retry_delay_seconds,
        )
        logger.info("Using direct camera source at index %s", cfg.camera_index)

    mic = MicrophoneSensor(cfg.audio_device)
    vision_enc = build_default_vision_encoder()
    audio_enc = build_default_audio_encoder()
    writer = ReplayWriter(cfg.replay_dir, cfg.episode_batch_size, cfg.max_replay_files)

    # If audio fails once (no device, misconfig, etc.), we switch to silence
    audio_enabled = True

    try:
        count = 0
        while True:
            t0 = time.time()

            # --- vision ---
            frame = cam.capture_frame()

            # --- audio (with graceful fallback) ---
            if audio_enabled:
                try:
                    waveform = mic.capture_window(duration_seconds=1.0)
                except Exception as exc:
                    logger.warning(
                        "Microphone capture failed (%s); disabling audio and using silence instead.",
                        exc,
                    )
                    audio_enabled = False
                    # 1 second of silence at the mic's sample rate (or 16 kHz default)
                    sr = getattr(mic, "sample_rate", 16000)
                    waveform = np.zeros(int(sr * 1.0), dtype=np.float32)
            else:
                # Already disabled: just use silence
                sr = getattr(mic, "sample_rate", 16000)
                waveform = np.zeros(int(sr * 1.0), dtype=np.float32)

            # --- encode multimodal inputs ---
            vision_vec = vision_enc.encode_frame(frame)
            audio_vec = audio_enc.encode_waveform(waveform, sample_rate=mic.sample_rate)

            episode = Episode(
                inputs={"vision": vision_vec, "auditory": audio_vec},
                metadata={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "pi",
                },
            )
            replay_record = episode_to_replay(episode)
            writer.append(replay_record)

            count += 1
            logger.info("Recorded episode %d", count)

            dt = time.time() - t0
            sleep_time = max(0.0, cfg.sample_interval_seconds - dt)
            time.sleep(sleep_time)

            if iterations is not None and count >= iterations:
                logger.info("Reached iteration limit (%d), stopping.", iterations)
                break
    finally:
        writer.flush()
        cam.close()
