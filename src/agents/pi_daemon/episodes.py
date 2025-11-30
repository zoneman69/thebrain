from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    inputs: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class ReplayRecord:
    fused: np.ndarray
    targets: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class _ReplayProjector:
    def __init__(self, output_dim: int = 256, seed: int = 2024):
        self.output_dim = output_dim
        self.seed = seed
        self._projection: np.ndarray | None = None

    def _ensure_matrix(self, input_dim: int) -> np.ndarray:
        if self._projection is None or self._projection.shape[0] != input_dim:
            rng = np.random.default_rng(self.seed)
            self._projection = rng.standard_normal((input_dim, self.output_dim)).astype(np.float32)
        return self._projection

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        matrix = self._ensure_matrix(vector.shape[0])
        return (vector.astype(np.float32) @ matrix).astype(np.float32)


def episode_to_replay(episode: Episode) -> ReplayRecord:
    embeddings: List[np.ndarray] = []
    for key in sorted(episode.inputs.keys()):
        embeddings.append(episode.inputs[key].astype(np.float32))
    if not embeddings:
        raise ValueError("Episode has no inputs to fuse")
    fused_raw = np.concatenate(embeddings).astype(np.float32)
    projector = _ReplayProjector()
    fused = projector(fused_raw)
    return ReplayRecord(fused=fused, targets={}, metadata=dict(episode.metadata))


class ReplayWriter:
    def __init__(self, replay_dir: str, episode_batch_size: int, max_replay_files: int):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.episode_batch_size = episode_batch_size
        self.max_replay_files = max_replay_files
        self._buffer: List[ReplayRecord] = []
        self._file_counter = 0

    def append(self, record: ReplayRecord) -> None:
        self._buffer.append(record)
        logger.info("Buffered replay record %d/%d", len(self._buffer), self.episode_batch_size)
        if len(self._buffer) >= self.episode_batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        fused = np.stack([r.fused for r in self._buffer], axis=0)
        metadata = np.array([r.metadata for r in self._buffer], dtype=object)
        targets = np.array([r.targets for r in self._buffer], dtype=object)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = self.replay_dir / f"replay-{timestamp}-{self._file_counter:04d}.npz"
        np.savez_compressed(filename, fused=fused, targets=targets, metadata=metadata)
        logger.info("Wrote replay file %s", filename)
        self._file_counter += 1
        self._buffer.clear()
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        files = sorted(self.replay_dir.glob("replay-*.npz"))
        if len(files) <= self.max_replay_files:
            return
        to_delete = files[:-self.max_replay_files]
        for path in to_delete:
            try:
                path.unlink()
                logger.info("Deleted old replay file %s", path)
            except OSError:
                logger.warning("Failed to delete old replay file %s", path)

    def flush(self) -> None:
        self._flush()
