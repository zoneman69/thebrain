#!/usr/bin/env python
"""Runpod-friendly training loop for hippocampus decoders."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import torch

from hippocampus import Event
from hippocampus.artifacts import export_decoder_artifacts
from hippocampus.cli import instantiate_hippocampus, load_config

logger = logging.getLogger("train_cloud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hippocampus decoders from episodic logs")
    parser.add_argument("--config", type=str, default="hippo.yaml", help="Path to hippocampus YAML config")
    parser.add_argument(
        "--log-dir", type=str, help="Directory containing episodic replay logs"
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate synthetic data instead of consuming a log directory",
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


def _generate_quick_data(hip: "Hippocampus", exposures: int = 6) -> int:
    """Populate the hippocampus with synthetic episodes for quick artifacts."""

    dims = {name: hip.enc.encoders[name][0].in_features for name in hip.modalities}
    t = 0.0
    for idx in range(exposures):
        base_feats = {name: torch.randn(dim) for name, dim in dims.items()}
        for offset, modality in enumerate(sorted(dims.keys())):
            dt = offset * 0.05
            affect = base_feats.get("affect") if modality == "affect" else None
            hip(Event(modality, base_feats[modality], t=t + dt, affect=affect), mode="encode")
        t += hip.window_size * 1.5
    hip.flush_pending()
    return hip.mem.V.shape[0]


def train_decoders(args: argparse.Namespace) -> Dict[str, int]:
    if not args.quick and (args.log_dir is None or args.config is None):
        raise SystemExit("--config and --log-dir are required unless --quick is provided")
    torch.set_num_threads(max(1, args.threads))
    config = load_config(args.config)
    hip = instantiate_hippocampus(config)
    if args.lam is not None:
        for decoder in hip.decoders.decoders.values():
            decoder.lam = args.lam
    hip.eval()

    sample_count = 0
    if args.quick:
        sample_count = _generate_quick_data(hip)
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory {log_dir} does not exist")
        for record in iter_training_records(log_dir):
            if not {"fused", "targets"}.issubset(record):
                continue
            for fused_vec, target_map in _batched_pairs(record):
                hip.decoders.observe(fused_vec, target_map)
                sample_count += 1

    if sample_count < args.min_samples:
        source = args.log_dir or "synthetic quick run"
        raise RuntimeError(
            f"Not enough samples ({sample_count}) collected from {source}; "
            f"need at least {args.min_samples}"
        )

    output_dir = Path(args.output_dir)
    source = str(Path(args.log_dir).resolve()) if args.log_dir else "synthetic"
    manifest = export_decoder_artifacts(hip, output_dir, samples=sample_count, source=source)
    logger.info("Exported %d samples to %s", sample_count, output_dir)
    return manifest.to_dict()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = train_decoders(args)
    logger.info("Training complete: modalities=%s", manifest["modalities"])


if __name__ == "__main__":
    main()
