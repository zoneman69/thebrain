
import torch
import torch.nn.functional as F

from hippocampus import Hippocampus, Event


def build_default_hippocampus():
    input_dims = {
        "vision": 512,
        "cerebellum": 64,
        "frontal": 128,
        "auditory": 256,
        "language": 256,
        "affect": 16,
    }
    return Hippocampus(
        input_dims=input_dims,
        shared_dim=192,
        time_dim=64,
        capacity=1024,
        sparsity=0.05,
        novelty_threshold=0.25,
        window_size=0.3,
        tau=0.15,
    )


def main(config: dict | None = None, hip: Hippocampus | None = None):
    seed = (config or {}).get("seed", 7)
    torch.manual_seed(seed)
    hip = hip or build_default_hippocampus()
    dims = hip.enc.encoders

    t0 = 0.0
    v = torch.randn(dims["vision"][0].in_features)
    cb = torch.randn(dims["cerebellum"][0].in_features)
    fr = torch.randn(dims["frontal"][0].in_features)
    au = torch.randn(dims["auditory"][0].in_features)
    ln = torch.randn(dims["language"][0].in_features)
    af = torch.randn(dims["affect"][0].in_features)

    hip(Event("vision", v, t=t0), mode="encode")
    hip(Event("cerebellum", cb, t=t0 + 0.05), mode="encode")
    hip(Event("frontal", fr, t=t0 + 0.10), mode="encode")
    hip(Event("auditory", au, t=t0 + 0.15), mode="encode")
    hip(Event("language", ln, t=t0 + 0.20), mode="encode")
    hip(Event("affect", af, t=t0 + 0.25, affect=af), mode="encode")
    hip.flush_pending()

    au_cue = au + 0.05 * torch.randn_like(au)
    res = hip(Event("auditory", au_cue, t=t0 + 0.45, prediction=au), mode="retrieve")

    out = res["output"].detach().cpu()
    true_enc = hip.enc(v.to(hip.mem.device), "vision").detach().cpu()
    sim_to_vision = F.cosine_similarity(out, true_enc, dim=0).item()

    print("Mode:", res["mode"])
    print("Novelty:", round(res["novelty"], 3))
    print("Cosine(out, vision-enc):", round(sim_to_vision, 3))
    print("Memory size:", hip.mem.K.shape[0])
    replay = hip.replay(batch=6)
    print("Replay batch shape:", tuple(replay["values"].shape))


if __name__ == "__main__":
    main()
