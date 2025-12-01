from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_config
from .encoders import build_default_audio_encoder, build_default_vision_encoder
from .episodes import Episode, ReplayWriter, episode_to_replay
from .sensors import (
    CameraSensor,
    FrameFileSensor,
    MicrophoneSensor,
    SpectrogramFileSensor,
)

logger = logging.getLogger(__name__)


def run_daemon(iterations: int | None = None) -> None:
    cfg = load_config()
    logger.info("Starting Pi daemon with replay_dir=%s", cfg.replay_dir)

    # --- vision source: shared frame file or direct camera ---
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

    # --- audio source: shared spectrogram file OR microphone ---
    spec_sensor: SpectrogramFileSensor | None = None
    mic: MicrophoneSensor | None = None
    audio_enabled = True

    if cfg.spec_path:
        spec_sensor = SpectrogramFileSensor(
            spec_path=Path(cfg.spec_path),
            retry_attempts=cfg.camera_retry_attempts,
            retry_delay_seconds=cfg.camera_retry_delay_seconds,
        )
        logger.info("Using spectrogram file source at %s", cfg.spec_path)
    else:
        mic = MicrophoneSensor(cfg.audio_device)

    vision_enc = build_default_vision_encoder()
    audio_enc = build_default_audio_encoder()
    writer = ReplayWriter(cfg.replay_dir, cfg.episode_batch_size, cfg.max_replay_files)

    try:
        count = 0
        while True:
            t0 = time.time()

            # --- vision ---
            frame = cam.capture_frame()

            # --- audio ---
            if spec_sensor is not None:
                # Read spectrogram image (e.g., HIPPO_SPEC) and treat it as mel-like
                mel_like = spec_sensor.capture_spectrogram()  # 2D float32
                audio_vec = audio_enc.encode_mel(mel_like)
            else:
                # Microphone path with graceful fallback to silence
                if audio_enabled and mic is not None:
                    try:
                        waveform = mic.capture_window(duration_seconds=1.0)
                    except Exception as exc:
                        logger.warning(
                            "Microphone capture failed (%s); disabling audio and using silence instead.",
                            exc,
                        )
                        audio_enabled = False
                        sr = getattr(mic, "sample_rate", 16000)
                        waveform = np.zeros(int(sr * 1.0), dtype=np.float32)
                    else:
                        sr = mic.sample_rate
                else:
                    # Already disabled or no mic configured: use silence
                    sr = getattr(mic, "sample_rate", 16000) if mic is not None else 16000
                    waveform = np.zeros(int(sr * 1.0), dtype=np.float32)

                audio_vec = audio_enc.encode_waveform(waveform, sample_rate=sr)

            # --- build episode & write replay ---
            episode = Episode(
                inputs={"vision": vision_vec, "auditory": audio_vec}
                if (vision_vec := vision_enc.encode_frame(frame)) is not None
                else {"vision": vision_enc.encode_frame(frame), "auditory": audio_vec},
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
        if spec_sensor is not None:
            spec_sensor.close()
