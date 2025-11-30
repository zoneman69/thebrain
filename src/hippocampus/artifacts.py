"""Artifact contract helpers for cloud training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch


@dataclass
class ArtifactsManifest:
    """Canonical metadata describing exported decoder artifacts."""

    format_version: str
    samples: int
    shared_dim: int
    modalities: List[str]
    artifacts: Dict[str, str]
    source_logs: str
    notes: str | None = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ArtifactsManifest":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)


def export_decoder_artifacts(hip, output_dir: Path, *, samples: int, source: str) -> ArtifactsManifest:
    """Persist decoder state dicts plus the gating stack to disk.

    Returns a validated :class:`ArtifactsManifest` that callers can publish or
    hand to downstream jobs. All paths in the manifest are relative to
    ``output_dir`` for portability.
    """

    decoder_dir = output_dir / "decoders"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hip.window_fusion.state_dict(), output_dir / "window_fusion.pt")
    torch.save(hip.fusion_gate.state_dict(), output_dir / "fusion_gate.pt")
    torch.save(hip.mix.state_dict(), output_dir / "mix.pt")
    torch.save(hip.time_mixer.state_dict(), output_dir / "time_mixer.pt")
    decoder_paths: Dict[str, str] = {
        "window_fusion": "window_fusion.pt",
        "fusion_gate": "fusion_gate.pt",
        "mix": "mix.pt",
        "time_mixer": "time_mixer.pt",
    }
    for name, decoder in hip.decoders.decoders.items():
        rel_path = f"decoders/{name}.pt"
        torch.save(decoder.state_dict(), output_dir / rel_path)
        decoder_paths[f"decoder_{name}"] = rel_path

    manifest = ArtifactsManifest(
        format_version="1.0",
        samples=samples,
        shared_dim=hip.decoders.shared_dim,
        modalities=sorted(hip.decoders.decoders.keys()),
        artifacts=decoder_paths,
        source_logs=source,
    )
    manifest.save(output_dir / "manifest.json")
    return manifest


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or validate hippocampus artifacts")
    parser.add_argument("artifact_dir", type=str, help="Directory containing manifest.json and tensors")
    parser.add_argument("--show", action="store_true", help="Print manifest contents")
    parser.add_argument("--check", action="store_true", help="Verify all referenced files exist")
    return parser


def _validate(manifest: ArtifactsManifest, root: Path) -> List[str]:
    missing: List[str] = []
    for rel_path in manifest.artifacts.values():
        if not (root / rel_path).exists():
            missing.append(rel_path)
    return missing


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    root = Path(args.artifact_dir)
    manifest_path = root / "manifest.json"
    manifest = ArtifactsManifest.load(manifest_path)
    if args.show:
        print(json.dumps(manifest.to_dict(), indent=2))
    if args.check:
        missing = _validate(manifest, root)
        if missing:
            raise SystemExit(f"Missing artifact files: {', '.join(missing)}")
        print("All artifact files present")


if __name__ == "__main__":
    main()
