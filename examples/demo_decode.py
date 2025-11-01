import json
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


def _event_from_result(
    *,
    res: dict,
    event_modality: str,
    t: float,
    recon_cosines: dict | None = None,
) -> dict:
    """
    Build a compact JSON-serializable event dict for telemetry/logging.
    Expects the result dict returned by Hippocampus(..., mode='retrieve').
    """
    meta = res.get("metadata") or []
    m0 = meta[0] if len(meta) else {}
    # Optional attention top-k: [(idx, score), ...]
    attn_pairs = m0.get("attn_topk") or []
    attn_indices = [int(i) for (i, _) in attn_pairs]
    attn_scores = [float(s) for (_, s) in attn_pairs]

    ev = {
        "event_modality": event_modality,
        "t": float(t),
        "mode": res.get("mode"),
        "novelty": float(res.get("novelty", 0.0)),
        "memory_size": int(res.get("memory_size", 0)),
        "pending_windows": int(res.get("pending_windows", 0)),
        "selected_window_id": m0.get("window_id"),
        "selected_t_window": m0.get("t_window"),
        "attn_indices": attn_indices if attn_indices else None,
        "attn_scores": attn_scores if attn_scores else None,
    }
    if recon_cosines:
        # e.g., {"vision": 0.97, "language": 0.96}
        for k, v in recon_cosines.items():
            ev[f"recon/{k}_cos"] = float(v)
    return ev


def main():
    torch.manual_seed(7)
    hip = build_demo_hippocampus()

    # Grab input dims from encoders that already exist
    dims = {name: hip.enc.encoders[name][0].in_features for name in hip.modalities}

    # Two synthetic episodes, A and B
    t0 = 0.0
    t1 = 5.0
    episode_A = {name: torch.randn(dim) for name, dim in dims.items()}
    episode_B = {name: torch.randn(dim) for name, dim in dims.items()}

    # Write multiple windows for A and B (staggered)
    exposures = 8
    for i in range(exposures):
        base = i * 2.5
        for offset, episode in ((0.0, episode_A), (1.2, episode_B)):
            t_base = base + offset
            hip(Event("vision", episode["vision"], t=t_base), mode="encode")
            hip(Event("auditory", episode["auditory"], t=t_base + 0.05), mode="encode")
            hip(Event("language", episode["language"], t=t_base + 0.10), mode="encode")

    # Commit any pending windows
    hip.flush_pending()

    # Build an auditory cue from episode B (with small noise)
    cue_noise = 0.05 * torch.randn_like(episode_B["auditory"])
    cue = episode_B["auditory"] + cue_noise

    # ---- Retrieval near Episode B time ----
    res_B = hip(
        Event("auditory", cue, t=t1 + 0.4, prediction=episode_B["auditory"]),
        mode="retrieve",
    )
    fused_B = res_B["output"]

    recon_modalities = ["vision", "language"]
    recon_B = hip.decode(fused_B, recon_modalities)

    # Targets in encoder space for Episode B
    targets_enc_B = {}
    with torch.no_grad():
        for mod in recon_modalities:
            targets_enc_B[mod] = hip.enc(episode_B[mod].to(hip.mem.device), mod)

    # Optional loss diagnostics (L1/L2)
    losses_B = hip.reconstruction_losses(fused_B, targets_enc_B, recon_modalities)

    print("Episode B cue -> retrieve")
    print("Mode:", res_B["mode"], "Novelty:", round(res_B["novelty"], 3))
    meta_B = res_B["metadata"][0] if res_B["metadata"] else {}
    print("Selected window:", meta_B.get("window_id"), "time range:", meta_B.get("t_window"))

    recon_cos_B = {}
    for modality in recon_modalities:
        l1 = format_loss(losses_B[modality]["l1"])
        l2 = format_loss(losses_B[modality]["l2"])
        cosine = F.cosine_similarity(recon_B[modality], targets_enc_B[modality], dim=0).item()
        recon_cos_B[modality] = cosine
        print(f"  {modality.title()} cosine: {cosine:.3f} | L1 {l1:.4f} | L2 {l2:.4f}")

    # Telemetry log for B retrieval
    ev_B = _event_from_result(
        res=res_B,
        event_modality="auditory",
        t=t1 + 0.4,
        recon_cosines=recon_cos_B,
    )
    print(json.dumps(ev_B))
    log_event(ev_B)

    # ---- Same cue, but retrieve with early time prior (Episode A) ----
    res_A = hip(
        Event("auditory", cue, t=t0 + 0.4, prediction=episode_B["auditory"]),
        mode="retrieve",
    )
    fused_A = res_A["output"]

    with torch.no_grad():
        targets_enc_A = {mod: hip.enc(episode_A[mod].to(hip.mem.device), mod) for mod in recon_modalities}
    recon_A = hip.decode(fused_A, recon_modalities)

    print("\nSame cue, early time prior -> retrieve")
    meta_A = res_A["metadata"][0] if res_A["metadata"] else {}
    print("Mode:", res_A["mode"], "Novelty:", round(res_A["novelty"], 3))
    print("Selected window:", meta_A.get("window_id"), "time range:", meta_A.get("t_window"))

    recon_cos_A = {}
    for modality in recon_modalities:
        cosine = F.cosine_similarity(recon_A[modality], targets_enc_A[modality], dim=0).item()
        recon_cos_A[modality] = cosine
        print(f"  {modality.title()} cosine vs episode A: {cosine:.3f}")

    # Telemetry log for A retrieval
    ev_A = _event_from_result(
        res=res_A,
        event_modality="auditory",
        t=t0 + 0.4,
        recon_cosines=recon_cos_A,
    )
    print(json.dumps(ev_A))
    log_event(ev_A)


if __name__ == "__main__":
    main()
