from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

LOGGER = logging.getLogger(__name__)

# Fused embedding dimensionality; must match hippocampus config (decoders).
FUSED_DIM = int(os.getenv("HIPPO_FUSED_DIM", "192"))


@dataclass
class Episode:
    """Single multimodal snapshot from the Pi."""

    inputs: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _ensure_vec(x: np.ndarray) -> np.ndarray:
    """Convert input to a 1D float32 vector of length FUSED_DIM."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] == FUSED_DIM:
        return arr
    # Truncate or pad to FUSED_DIM as a safety net
    if arr.shape[0] > FUSED_DIM:
        LOGGER.debug("Truncating vector from %d -> %d dims", arr.shape[0], FUSED_DIM)
        return arr[:FUSED_DIM]
    LOGGER.debug("Padding vector from %d -> %d dims", arr.shape[0], FUSED_DIM)
    return np.pad(arr, (0, FUSED_DIM - arr.shape[0]), mode="constant")


def episode_to_replay(ep: Episode) -> Dict[str, Any]:
    """
    Convert an Episode to a replay record with:
      - 'fused': FUSED_DIM vector
      - 'targets': dict of modality -> FUSED_DIM vector
      - 'metadata': shallow copy of ep.metadata
    """
    targets: Dict[str, np.ndarray] = {}

    # Store each modality as a target
    for name, vec in ep.inputs.items():
        try:
            targets[name] = _ensure_vec(vec)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to normalize modality %s: %s", name, exc)

    # Fuse by averaging available modalities (simple, but works)
    if targets:
        stacked = np.stack(list(targets.values()), axis=0)  # (num_modalities, FUSED_DIM)
        fused_vec = stacked.mean(axis=0).astype(np.float32)
    else:
        fused_vec = np.zeros(FUSED_DIM, dtype=np.float32)

    return {
        "fused": fused_vec,
        "targets": targets,
        "metadata": dict(ep.metadata),
    }


class ReplayWriter:
    """
    Buffered writer that collects Episodes (as replay records) and writes
    NPZ batches with keys: 'fused', 'targets', 'metadata'.

    Each NPZ file contains up to batch_size episodes. Oldest files are
    deleted when max_files is exceeded.
    """

    def __init__(self, replay_dir: str | os.PathLike[str], batch_size: int = 256, max_files: int = 10):
        self.replay_dir = Path(replay_dir)
        self.batch_size = int(batch_size)
        self.max_files = int(max_files)

        self.replay_dir.mkdir(parents=True, exist_ok=True)

        self._fused_buf: List[np.ndarray] = []
        self._targets_buf: List[Dict[str, np.ndarray]] = []
        self._meta_buf: List[Dict[str, Any]] = []
        self._file_counter: int = 0

        LOGGER.info(
            "Initialized ReplayWriter at %s (batch_size=%d, max_files=%d)",
            self.replay_dir,
            self.batch_size,
            self.max_files,
        )

    def append(self, record: Dict[str, Any]) -> None:
        """
        Append a single replay record as returned by episode_to_replay().
        """
        fused = _ensure_vec(record["fused"])
        targets = record.get("targets", {})
        metadata = record.get("metadata", {})

        self._fused_buf.append(fused)
        self._targets_buf.append(targets)
        self._meta_buf.append(metadata)

        n = len(self._fused_buf)
        LOGGER.info("Buffered replay record %d/%d", n, self.batch_size)

        if n >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        if not self._fused_buf:
            return

        fused_arr = np.stack(self._fused_buf, axis=0).astype(np.float32)  # (N, FUSED_DIM)
        targets_arr = np.array(self._targets_buf, dtype=object)  # (N,)
        meta_arr = np.array(self._meta_buf, dtype=object)  # (N,)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        idx = self._file_counter % 10000
        filename = self.replay_dir / f"replay-{ts}-{idx:04d}.npz"
        np.savez_compressed(filename, fused=fused_arr, targets=targets_arr, metadata=meta_arr)

        LOGGER.info("Wrote replay file %s", filename)

        self._file_counter += 1
        self._fused_buf.clear()
        self._targets_buf.clear()
        self._meta_buf.clear()

        self._enforce_limit()

    def _enforce_limit(self) -> None:
        files = sorted(self.replay_dir.glob("replay-*.npz"))
        excess = len(files) - self.max_files
        for f in files[:max(0, excess)]:
            try:
                f.unlink()
                LOGGER.info("Removed old replay file %s", f)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to remove old replay file %s: %s", f, exc)

    def flush(self) -> None:
        """Flush any buffered episodes to disk."""
        self._flush_batch()
