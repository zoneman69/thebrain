import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hippocampus import Hippocampus, Event
from examples.demo_multimodal import build_multimodal_hippocampus


def main():
    torch.manual_seed(321)
    hip = build_multimodal_hippocampus()
    dims = hip.enc.encoders

    def encode_episode(base_time: float, offset: float):
        episode = {name: torch.randn(dims[name][0].in_features) for name in dims.keys()}
        for modality, dt in {
            "vision": 0.0,
            "cerebellum": 0.04,
            "frontal": 0.08,
            "auditory": 0.12,
            "language": 0.16,
            "affect": 0.20,
        }.items():
            affect_vec = episode["affect"] if modality == "affect" else None
            hip(Event(modality, episode[modality], t=base_time + offset + dt, affect=affect_vec), mode="encode")
        return episode

    episodes = [encode_episode(i * 5.0, i * 0.01) for i in range(3)]
    hip.flush_pending()

    cue = episodes[1]["language"] + 0.02 * torch.randn_like(episodes[1]["language"])
    res = hip(Event("language", cue, t=6.5, prediction=episodes[1]["language"]), mode="retrieve")

    print("Mode:", res["mode"])
    print("Top-5 indices and scores:")
    for idx, score, meta in zip(res["indices"][:5], res["attn"][res["indices"][:5]], res["metadata"][:5]):
        window = meta.get("t_window", (None, None))
        mods = list(meta.get("modalities", {}).keys())
        print(json.dumps({
            "idx": idx,
            "score": round(float(score), 4),
            "window": window,
            "modalities": mods,
        }))

    print("Memory occupancy:", hip.mem.K.shape[0])


if __name__ == "__main__":
    main()
