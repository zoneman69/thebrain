from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.pi_daemon import (
    Episode,
    ReplayWriter,
    build_default_audio_encoder,
    build_default_vision_encoder,
    episode_to_replay,
    load_config,
)
from agents.pi_daemon.daemon import run_daemon
from agents.pi_daemon.sensors import MockSensorPair


def test_episode_to_replay_shapes():
    vision = np.ones((64, 64, 3), dtype=np.uint8)
    audio = np.ones(16000, dtype=np.float32)

    vision_enc = build_default_vision_encoder()
    audio_enc = build_default_audio_encoder()

    vision_vec = vision_enc.encode_frame(vision)
    audio_vec = audio_enc.encode_waveform(audio, sample_rate=16000)

    episode = Episode(inputs={"vision": vision_vec, "auditory": audio_vec}, metadata={})
    record = episode_to_replay(episode)
    assert record.fused.shape == (256,)
    assert record.fused.dtype == np.float32


def test_config_allows_camera_device_path(monkeypatch):
    monkeypatch.setenv("PI_CAMERA_INDEX", "/dev/video2")
    cfg = load_config()
    assert cfg.camera_index == "/dev/video2"


def test_replay_writer_creates_files(tmp_path: Path):
    writer = ReplayWriter(replay_dir=tmp_path, episode_batch_size=2, max_replay_files=3)
    dummy = np.zeros(256, dtype=np.float32)
    for _ in range(4):
        writer.append(episode_to_replay(Episode({"vision": dummy}, metadata={})))
    writer.flush()
    files = sorted(tmp_path.glob("replay-*.npz"))
    assert len(files) <= 3
    assert files, "Replay files should be created"


def test_daemon_loop_with_mocks(tmp_path: Path, monkeypatch):
    mock_sensors = MockSensorPair(
        camera_frame=np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
        audio_waveform=np.random.randn(16000).astype(np.float32),
    )

    monkeypatch.setenv("PI_REPLAY_DIR", str(tmp_path))
    monkeypatch.setenv("PI_EPISODE_BATCH_SIZE", "3")
    monkeypatch.setenv("PI_MAX_REPLAY_FILES", "2")

    monkeypatch.setattr(
        "agents.pi_daemon.daemon.CameraSensor",
        lambda *args, **kwargs: mock_sensors.build_camera(),
    )
    monkeypatch.setattr(
        "agents.pi_daemon.daemon.MicrophoneSensor",
        lambda *args, **kwargs: mock_sensors.build_microphone(),
    )

    run_daemon(iterations=5)

    files = list(tmp_path.glob("replay-*.npz"))
    assert files, "Daemon should write replay files"
    latest = max(files, key=lambda p: p.stat().st_mtime)
    data = np.load(latest, allow_pickle=True)
    fused = data["fused"]
    assert fused.shape[1] == 256
    assert fused.dtype == np.float32

    # Ensure cleanup policy applied
    assert len(files) <= 2
