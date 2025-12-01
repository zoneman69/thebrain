from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .config import load_config
from .encoders import build_default_audio_encoder, build_default_vision_encoder
from .episodes import Episode, ReplayWriter, episode_to_replay
from .sensors import CameraSensor, MicrophoneSensor

logger = logging.getLogger(__name__)


def run_daemon(iterations: int | None = None) -> None:
    cfg = load_config()
    logger.info("Starting Pi daemon with replay_dir=%s", cfg.replay_dir)

    cam = CameraSensor(
        camera_index=cfg.camera_index,
        retry_attempts=cfg.camera_retry_attempts,
        retry_delay_seconds=cfg.camera_retry_delay_seconds,
    )
    mic = MicrophoneSensor(cfg.audio_device)
    vision_enc = build_default_vision_encoder()
    audio_enc = build_default_audio_encoder()
    writer = ReplayWriter(cfg.replay_dir, cfg.episode_batch_size, cfg.max_replay_files)

    try:
        count = 0
        while True:
            t0 = time.time()
            frame = cam.capture_frame()
            waveform = mic.capture_window(duration_seconds=1.0)

            vision_vec = vision_enc.encode_frame(frame)
            audio_vec = audio_enc.encode_waveform(waveform, sample_rate=mic.sample_rate)

            episode = Episode(
                inputs={"vision": vision_vec, "auditory": audio_vec},
                metadata={"timestamp": datetime.now(timezone.utc).isoformat(), "source": "pi"},
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
