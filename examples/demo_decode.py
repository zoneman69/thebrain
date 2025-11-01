"""Demonstration of cross-modal reconstruction using hippocampal decoders."""

import torch
import torch.nn.functional as F

from hippocampus import Event, Hippocampus
from hippocampus.telemetry import log_event


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
        sparsity=0.04,
        novelty_threshold=0.2,
        window_size=0.35,
        tau=0.1,
        read_topk=1,
        temporal_beta=2.5,
        temporal_sigma=0.35,
    )


def format_loss(t: torch.Tensor) -> float:
    return float(t.detach().cpu().item())


def main():
    torch.manual_seed(7)
    hip = build_demo_hippocampus()

    dims = {name: hip.enc.encoders[name][0].in_features for name in hip.modalities}

    t0 = 0.0
    t1 = 5.0

    episode_A = {name: torch.randn(dim) for name, dim in dims.items()}
    episode_B = {name: torch.randn(dim) for name, dim in dims.items()}

    exposures = 8
    for i in range(exposures):
        base = i * 2.5
        for offset, episode in ((0.0, episode_A), (1.2, episode_B)):
            t_base = base + offset
            hip(Event("vision", episode["vision"], t=t_base), mode="encode")
            hip(Event("auditory", episode["auditory"], t=t_base + 0.05), mode="encode")
            hip(Event("language", episode["language"], t=t_base + 0.10), mode="encode")

    hip.flush_pending()

    cue_noise = 0.05 * torch.randn_like(episode_B["auditory"])
    cue = episode_B["auditory"] + cue_noise

    res_B = hip(
        Event("auditory", cue, t=t1 + 0.4, prediction=episode_B["auditory"]),
        mode="retrieve",
    )
    fused_B = res_B["output"]
    recon_modalities = ["vision", "language"]
    recon_B = hip.decode(fused_B, recon_modalities)
    targets_enc = {}
    with torch.no_grad():
        for mod in recon_modalities:
            targets_enc[mod] = hip.enc(episode_B[mod].to(hip.mem.device), mod)
    losses_B = hip.reconstruction_losses(fused_B, targets_enc, recon_modalities)

    print("Episode B cue -> retrieve")
    print("Mode:", res_B["mode"], "Novelty:", round(res_B["novelty"], 3))
    meta_B = res_B["metadata"][0] if res_B["metadata"] else {}
    print("Selected window:", meta_B.get("window_id"), "time range:", meta_B.get("t_window"))

    for modality in recon_modalities:
        l1 = format_loss(losses_B[modality]["l1"])
        l2 = format_loss(losses_B[modality]["l2"])
        cosine = F.cosine_similarity(recon_B[modality], targets_enc[modality], dim=0).item()
        print(f"  {modality.title()} cosine: {cosine:.3f} | L1 {l1:.4f} | L2 {l2:.4f}")

    # Re-run the same cue but shift the retrieval time near episode A
    res_A = hip(
        Event("auditory", cue, t=t0 + 0.4, prediction=episode_B["auditory"]),
        mode="retrieve",
    )
    fused_A = res_A["output"]
    with torch.no_grad():
        targets_enc_A = {
            mod: hip.enc(episode_A[mod].to(hip.mem.device), mod) for mod in recon_modalities
        }
    recon_A = hip.decode(fused_A, recon_modalities)

    print("\nSame cue, early time prior -> retrieve")
    meta_A = res_A["metadata"][0] if res_A["metadata"] else {}
    print("Mode:", res_A["mode"], "Novelty:", round(res_A["novelty"], 3))
    print("Selected window:", meta_A.get("window_id"), "time range:", meta_A.get("t_window"))
    for modality in recon_modalities:
        cosine = F.cosine_similarity(recon_A[modality], targets_enc_A[modality], dim=0).item()
        print(f"  {modality.title()} cosine vs episode A: {cosine:.3f}")
    # after computing each event dict:
    print(json.dumps(ev))   # keep your print
    log_event(ev)           # also write to /tmp/hippo.jsonl


if __name__ == "__main__":
    main()
