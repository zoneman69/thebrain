import torch
import torch.nn.functional as F
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hippocampus import Hippocampus, Event
from examples.demo_multimodal import build_multimodal_hippocampus


def main(config: dict | None = None, hip: Hippocampus | None = None):
    torch.manual_seed((config or {}).get("seed", 123))
    hip = hip or build_multimodal_hippocampus()
    dims = hip.enc.encoders

    episode_gap = 12.0
    window_dt = {
        "vision": 0.0,
        "cerebellum": 0.05,
        "frontal": 0.10,
        "auditory": 0.15,
        "language": 0.20,
        "affect": 0.25,
    }

    def build_episode(seed_offset: int = 0):
        torch.manual_seed(100 + seed_offset)
        return {name: torch.randn(dims[name][0].in_features) for name in dims.keys()}

    episode_a = build_episode(0)
    episode_b = {name: (episode_a[name] + 0.03 * torch.randn_like(episode_a[name])) for name in dims.keys()}

    for base_time, episode, affect_scale in [
        (0.0, episode_a, 1.0),
        (episode_gap, episode_b, -1.0),
    ]:
        for modality, dt in window_dt.items():
            affect_vec = episode["affect"] * affect_scale if modality == "affect" else None
            hip(Event(modality, episode[modality], t=base_time + dt, affect=affect_vec), mode="encode")
    hip.flush_pending()

    auditory_cue = episode_a["auditory"] + 0.02 * torch.randn_like(episode_a["auditory"])
    res_a = hip(Event("auditory", auditory_cue, t=episode_gap + 0.5, prediction=episode_a["auditory"]), mode="retrieve")
    fused = res_a["output"].detach()

    vision_a = hip.enc(episode_a["vision"].to(hip.mem.device), "vision")
    vision_b = hip.enc(episode_b["vision"].to(hip.mem.device), "vision")
    cos_a = F.cosine_similarity(fused, vision_a, dim=0).item()
    cos_b = F.cosine_similarity(fused, vision_b, dim=0).item()

    print("Mode:", res_a["mode"])
    print("Cosine(out, vision-enc A):", round(cos_a, 3))
    print("Cosine(out, vision-enc B):", round(cos_b, 3))
    print("Acceptance: out vs A >= 0.85, out vs B much lower")


if __name__ == "__main__":
    main()
