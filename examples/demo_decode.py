"""Demonstration of cross-modal reconstruction using hippocampal decoders."""

import torch
import torch.nn.functional as F

from hippocampus import Hippocampus, Event


def build_demo_hippocampus() -> Hippocampus:
    input_dims = {
        "vision": 512,
        "auditory": 256,
        "language": 256,
    }
    return Hippocampus(
        input_dims=input_dims,
        shared_dim=192,
        time_dim=64,
        capacity=512,
        sparsity=0.05,
        novelty_threshold=0.2,
        window_size=0.3,
        tau=0.15,
    )


def format_loss(t: torch.Tensor) -> float:
    return float(t.detach().cpu().item())


def main():
    torch.manual_seed(7)
    hip = build_demo_hippocampus()

    dims = {name: hip.enc.encoders[name][0].in_features for name in hip.modalities}

    t0 = 0.0
    t1 = 6.0

    episode_A = {name: torch.randn(dim) for name, dim in dims.items()}
    episode_B = {name: torch.randn(dim) for name, dim in dims.items()}

    for t, episode in ((t0, episode_A), (t1, episode_B)):
        hip(Event("vision", episode["vision"], t=t), mode="encode")
        hip(Event("auditory", episode["auditory"], t=t + 0.05), mode="encode")
        hip(Event("language", episode["language"], t=t + 0.10), mode="encode")

    hip.flush_pending()

    # Retrieve with an auditory cue from episode B
    cue = episode_B["auditory"] + 0.05 * torch.randn_like(episode_B["auditory"])
    res = hip(Event("auditory", cue, t=t1 + 0.4, prediction=episode_B["auditory"]), mode="retrieve")
    fused = res["output"]

    recon_modalities = ["vision", "language"]
    recon = hip.decode(fused, recon_modalities)
    targets = {mod: episode_B[mod] for mod in recon_modalities}
    losses = hip.reconstruction_losses(fused, targets, recon_modalities)

    print("Mode:", res["mode"])
    print("Novelty:", round(res["novelty"], 3))

    for modality in recon_modalities:
        l1 = format_loss(losses[modality]["l1"])
        l2 = format_loss(losses[modality]["l2"])
        cosine = F.cosine_similarity(recon[modality], targets[modality], dim=0).item()
        print(f"{modality.title()} reconstruction:")
        print(f"  L1: {l1:.4f} | L2: {l2:.4f} | cosine: {cosine:.3f}")


if __name__ == "__main__":
    main()
