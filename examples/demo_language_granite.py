"""Language-only Granite Nano demo."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hippocampus import Hippocampus
from hippocampus.integrations import GraniteModelUnavailable, GraniteNanoEncoder


def _fallback_encoder(_: str) -> torch.Tensor:
    # Deterministic fallback so examples stay stable without transformers
    torch.manual_seed(0)
    return torch.randn(1024)


def build_encoder() -> Callable[[str], torch.Tensor]:
    try:
        granite = GraniteNanoEncoder()
        return granite.encode
    except GraniteModelUnavailable:
        print("Granite Nano unavailable; using Gaussian fallback for language embeddings")
        return _fallback_encoder


def main(config: dict | None = None, hip: Hippocampus | None = None) -> None:
    embed = build_encoder()
    hippo = hip or Hippocampus({"language": 1024}, shared_dim=256, time_dim=32, capacity=64)

    utterances = [
        "welcome to the brain demo",
        "granite nano produces compact embeddings",
        "retrieval should bring back prior phrases",
    ]
    t = 0.0
    for phrase in utterances:
        vec = embed(phrase).float()
        hippo.encode("language", vec, t=t)
        t += hippo.window_size * 1.2
    hippo.flush_pending()

    cue = embed("bring back prior phrases").float()
    res = hippo.recall("language", cue, t=t + 0.5, decode_modalities=["language"])
    print("Mode:", res["mode"], "novelty=", round(res["novelty"], 4))
    print("Top attention indices:", res.get("indices", [])[:5])
    print("Fused norm:", res["output"].norm().item())


if __name__ == "__main__":
    main()
