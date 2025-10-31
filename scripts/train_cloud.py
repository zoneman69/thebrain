#!/usr/bin/env python
"""Runpod-friendly training loop for hippocampus decoders."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import torch

from hippocampus.cli import instantiate_hippocampus, load_config

logger = logging.getLogger("train_cloud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hippocampus decoders from episodic logs")
    parser.add_argument("--config", required=True, type=str, help="Path to hippocampus YAML config")
    parser.add_argument(
        "--log-dir", required=True, type=str, help="Directory containing episodic replay logs"
    )
    parser.add_argument(
        "--output-dir", default="artifacts", type=str, help="Directory to store weight artifacts"
    )
    parser.add_argument(
        "--threads", default=4, type=int, help="torch.set_num_threads value for CPU-only training"
    )
    parser.add_argument(
        "--min-samples",
        default=1,
        type=int,
        help="Require at least this many fused samples before exporting weights",
    )
    parser.add_argument(
        "--lam", type=float, default=None, help="Optional ridge regression lambda override for decoders"
    )
    return parser.parse_args()


def _coerce_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().float()
    return torch.tensor(value, dtype=torch.float32)


def _iter_records_from_payload(payload) -> Iterator[Dict]:
    if payload is None:
        return
    if isinstance(payload, dict):
        if "fused" in payload and "targets" in payload:
            yield payload
        else:
            for key in ("records", "samples", "events"):
                if key in payload:
                    for item in payload[key]:
                        yield from _iter_records_from_payload(item)
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        for item in payload:
            yield from _iter_records_from_payload(item)


def iter_training_records(root: Path) -> Iterator[Dict]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".pt", ".pth", ".bin"}:
                payload = torch.load(path, map_location="cpu")
                yield from _iter_records_from_payload(payload)
            elif suffix in {".json", ".jsonl"}:
                with path.open("r", encoding="utf-8") as handle:
                    if suffix == ".jsonl":
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            yield from _iter_records_from_payload(json.loads(line))
                    else:
                        yield from _iter_records_from_payload(json.load(handle))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s due to error: %s", path, exc)
            continue


def _batched_pairs(record: Dict[str, torch.Tensor]) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    fused = _coerce_tensor(record["fused"])
    targets = {mod: _coerce_tensor(tensor) for mod, tensor in record["targets"].items()}
    if fused.ndim == 1:
        yield fused, targets
        return
    if fused.ndim != 2:
        raise ValueError(f"Unsupported fused tensor shape: {tuple(fused.shape)}")
    length = fused.shape[0]
    for modality, tensor in targets.items():
        if tensor.ndim == 1:
            targets[modality] = tensor.unsqueeze(0).expand(length, -1)
    for idx in range(length):
        yield fused[idx], {mod: tensor[idx] for mod, tensor in targets.items()}


def train_decoders(args: argparse.Namespace) -> Dict[str, int]:
    torch.set_num_threads(max(1, args.threads))
    config = load_config(args.config)
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory {log_dir} does not exist")
    hip = instantiate_hippocampus(config)
    if args.lam is not None:
        for decoder in hip.decoders.decoders.values():
            decoder.lam = args.lam
    hip.eval()

    sample_count = 0
    for record in iter_training_records(log_dir):
        if not {"fused", "targets"}.issubset(record):
            continue
        for fused_vec, target_map in _batched_pairs(record):
            hip.decoders.observe(fused_vec, target_map)
            sample_count += 1

    if sample_count < args.min_samples:
        raise RuntimeError(
            f"Not enough samples ({sample_count}) collected from {args.log_dir!s}; "
            f"need at least {args.min_samples}"
        )

    output_dir = Path(args.output_dir)
    decoder_dir = output_dir / "decoders"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hip.fusion.state_dict(), output_dir / "fused_head.pt")
    torch.save(hip.mix.state_dict(), output_dir / "mix.pt")
    for name, decoder in hip.decoders.decoders.items():
        torch.save(decoder.state_dict(), decoder_dir / f"{name}.pt")
    manifest = {
        "samples": sample_count,
        "shared_dim": hip.decoders.shared_dim,
        "modalities": sorted(hip.decoders.decoders.keys()),
        "source_logs": str(Path(args.log_dir).resolve()),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Exported %d samples to %s", sample_count, output_dir)
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = train_decoders(args)
    logger.info("Training complete: modalities=%s", manifest["modalities"])


if __name__ == "__main__":
    main()
