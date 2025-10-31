import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hippocampus import Hippocampus, Event
from examples.demo_multimodal import build_multimodal_hippocampus

WINDOW_DTS = {
    "vision": 0.0,
    "cerebellum": 0.05,
    "frontal": 0.10,
    "auditory": 0.15,
    "language": 0.20,
    "affect": 0.25,
}


def encode_episode(hip: Hippocampus, base_time: float, context: str, features: dict[str, torch.Tensor]):
    for modality, dt in WINDOW_DTS.items():
        affect_vec = features["affect"] if modality == "affect" else None
        hip(Event(modality, features[modality], t=base_time + dt, context=context, affect=affect_vec), mode="encode")


def build_benchmark(hip: Hippocampus, pairs: int, time_shift: float):
    dims = hip.enc.encoders
    episodes: list[tuple[str, dict[str, torch.Tensor], float]] = []
    for idx in range(pairs):
        base_feats = {name: torch.randn(dims[name][0].in_features) for name in dims.keys()}
        dup_feats = {name: base_feats[name] + 0.04 * torch.randn_like(base_feats[name]) for name in dims.keys()}
        context_a = f"ep{idx}_A"
        context_b = f"ep{idx}_B"
        encode_episode(hip, base_time=idx * (time_shift + 1.0), context=context_a, features=base_feats)
        encode_episode(hip, base_time=idx * (time_shift + 1.0) + time_shift, context=context_b, features=dup_feats)
        episodes.append((context_a, base_feats, idx * (time_shift + 1.0)))
        episodes.append((context_b, dup_feats, idx * (time_shift + 1.0) + time_shift))
    hip.flush_pending()
    return episodes


def pattern_completion_metric(hip: Hippocampus, episodes, retrievals, vision_encs):
    cosines = []
    for context, _, _ in episodes:
        res = retrievals[context]
        cos = F.cosine_similarity(res["output"], vision_encs[context], dim=0).item()
        cosines.append(cos)
    return statistics.mean(cosines)


def time_separation_metric(hip: Hippocampus, episodes, retrievals, vision_encs):
    deltas = []
    for context, _, _ in episodes:
        if context.endswith("_B"):
            pair_a = context.replace("_B", "_A")
            fused = retrievals[context]["output"]
            cos_a = F.cosine_similarity(fused, vision_encs[pair_a], dim=0).item()
            cos_b = F.cosine_similarity(fused, vision_encs[context], dim=0).item()
            deltas.append(cos_a - cos_b)
    return statistics.mean(deltas)


def train_probe(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor, steps: int = 200) -> float:
    model = torch.nn.Linear(train_x.size(1), int(train_y.max().item()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = model(eval_x).argmax(dim=-1)
        acc = (preds == eval_y).float().mean().item()
    return acc


def replay_probe_metrics(hip: Hippocampus, retrievals, vision_encs):
    contexts = hip.mem.contexts
    label_map = {ctx: idx for idx, ctx in enumerate(sorted(set(contexts)))}
    base_train_x = hip.mem.V.detach().cpu()
    base_train_y = torch.tensor([label_map[c] for c in contexts], dtype=torch.long)

    replay = hip.replay(batch=len(contexts))
    replay_x = replay["values"].detach().cpu()
    replay_y = torch.tensor([label_map[m.get("context", c)] for m, c in zip(replay["metadata"], [contexts[i] for i in replay["indices"]])], dtype=torch.long)

    eval_x = torch.stack([retrievals[c]["output"].detach().cpu() for c in contexts], dim=0)
    eval_y = torch.tensor([label_map[c] for c in contexts], dtype=torch.long)

    acc_base = train_probe(base_train_x, base_train_y, eval_x, eval_y)
    combined_x = torch.cat([base_train_x, replay_x], dim=0)
    combined_y = torch.cat([base_train_y, replay_y], dim=0)
    acc_replay = train_probe(combined_x, combined_y, eval_x, eval_y)
    return acc_base, acc_replay


def run_benchmark(pairs: int, time_shift: float, seed: int = 99, hip: Hippocampus | None = None):
    torch.manual_seed(seed)
    hip_instance = hip or build_multimodal_hippocampus()
    episodes = build_benchmark(hip_instance, pairs=pairs, time_shift=time_shift)

    vision_encs = {}
    retrievals = {}
    for context, features, t in episodes:
        vision_encs[context] = hip_instance.enc(features["vision"].to(hip_instance.mem.device), "vision")
        cue = features["language"] + 0.02 * torch.randn_like(features["language"])
        retrievals[context] = hip_instance(Event("language", cue, t=t + 0.4, prediction=features["language"], context=context), mode="retrieve")

    pattern = pattern_completion_metric(hip_instance, episodes, retrievals, vision_encs)
    separation = time_separation_metric(hip_instance, episodes, retrievals, vision_encs)
    acc_base, acc_replay = replay_probe_metrics(hip_instance, retrievals, vision_encs)

    print("metric\tvalue")
    print(f"pattern_completion\t{pattern:.3f}")
    print(f"time_separation_delta\t{separation:.3f}")
    print(f"probe_accuracy_no_replay\t{acc_base:.3f}")
    print(f"probe_accuracy_with_replay\t{acc_replay:.3f}")


def run(config: dict | None = None, hip: Hippocampus | None = None):
    cfg = config or {}
    pairs = cfg.get("benchmark_pairs", 4)
    time_shift = cfg.get("benchmark_time_shift", 8.0)
    seed = cfg.get("seed", 99)
    run_benchmark(pairs=pairs, time_shift=time_shift, seed=seed, hip=hip)


def main():
    parser = argparse.ArgumentParser(description="Benchmark hippocampus recall metrics")
    parser.add_argument("--pairs", type=int, default=4, help="Number of episode pairs")
    parser.add_argument("--time-shift", type=float, default=8.0, help="Temporal gap between paired episodes")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--quick", action="store_true", help="Run a lightweight configuration")
    args = parser.parse_args()

    pairs = 2 if args.quick else args.pairs
    time_shift = 6.0 if args.quick else args.time_shift
    run_benchmark(pairs=pairs, time_shift=time_shift, seed=args.seed)


if __name__ == "__main__":
    main()
