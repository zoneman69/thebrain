#!/usr/bin/env python
"""Train hippocampus decoders from Pi replay NPZ files."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch

from hippocampus.artifacts import export_decoder_artifacts
from hippocampus.cli import instantiate_hippocampus, load_config

LOGGER = logging.getLogger("train_from_replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hippocampus decoders from replay logs")
    parser.add_argument("--config", default="hippo.yaml", help="Path to hippocampus YAML config")
    parser.add_argument(
        "--replay-dir",
        default=os.getenv("PI_REPLAY_DIR", "/home/image/thebrain/replay"),
        help="Directory containing replay NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("HIPPO_ARTIFACTS_DIR", "artifacts/trained"),
        help="Directory to write updated decoder artifacts",
    )
    parser.add_argument("--min-samples", type=int, default=1, help="Require at least this many samples before writing artifacts")
    return parser.parse_args()


def _iter_samples(path: Path) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    try:
        payload = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load %s: %s", path, exc)
        return

    fused = payload.get("fused")
    targets = payload.get("targets")
    if fused is None:
        LOGGER.debug("Replay file %s missing 'fused'", path)
        return

    targets = targets if targets is not None else np.array([])

    for idx in range(fused.shape[0]):
        fused_vec = torch.tensor(fused[idx], dtype=torch.float32)
        target_map: Dict[str, torch.Tensor] = {}

        if idx < targets.shape[0]:
            raw_target = targets[idx]
            if isinstance(raw_target, dict) and raw_target:
                target_map = {
                    k: torch.tensor(v, dtype=torch.float32)
                    for k, v in raw_target.items()
                }

        # If there are no targets, let the caller decide whether to skip
        if not target_map:
            yield fused_vec, target_map
            continue

        # --- Align fused dim with target dim ---
        # Use the first modality's dim as canonical (e.g. 192)
        first_tgt = next(iter(target_map.values()))
        tgt_dim = first_tgt.shape[0]

        if fused_vec.shape[0] != tgt_dim:
            LOGGER.debug(
                "Adjusting fused dim from %d -> %d for %s",
                fused_vec.shape[0],
                tgt_dim,
                path.name,
            )
            if fused_vec.shape[0] > tgt_dim:
                # Truncate extra dims
                fused_vec = fused_vec[:tgt_dim]
            else:
                # Pad with zeros if fused is shorter
                pad = tgt_dim - fused_vec.shape[0]
                fused_vec = torch.nn.functional.pad(fused_vec, (0, pad))

        yield fused_vec, target_map

        yield fused_vec, target_map


def iter_replay_records(replay_dir: Path) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    for path in sorted(replay_dir.glob("replay-*.npz")):
        yield from _iter_samples(path)


def train_from_replay(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    hip = instantiate_hippocampus(cfg)
    hip.eval()

    replay_dir = Path(args.replay_dir)
    if not replay_dir.exists():
        raise FileNotFoundError(f"Replay dir {replay_dir} does not exist")

    sample_count = 0
    trained_count = 0
    for fused_vec, targets in iter_replay_records(replay_dir):
        sample_count += 1
        if not targets:
            LOGGER.debug("Skipping sample without targets")
            continue
        fused_vec = fused_vec.to(hip.mem.device)
        targets = {k: v.to(hip.mem.device) for k, v in targets.items()}
        hip.decoders.observe(fused_vec, targets)
        trained_count += 1

    if trained_count < args.min_samples:
        raise RuntimeError(
            f"Only collected {trained_count} trainable sample(s) from {replay_dir}; need at least {args.min_samples}"
        )

    output_dir = Path(args.output_dir)
    manifest = export_decoder_artifacts(hip, output_dir, samples=trained_count, source=str(replay_dir.resolve()))
    LOGGER.info("Exported decoder artifacts to %s", output_dir)
    LOGGER.info("Modalities: %s", manifest.modalities)
    LOGGER.info("Samples seen: %d; trained on: %d", sample_count, trained_count)
    return trained_count


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_from_replay(args)


if __name__ == "__main__":
    main()
