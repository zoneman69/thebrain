
import torch
from hippocampus import Hippocampus, Event
import torch.nn.functional as F

def main():
    torch.manual_seed(7)
    input_dims = {
        "vision": 512,
        "cerebellum": 64,
        "frontal": 128,
        "auditory": 256,
        "language": 256,
        "affect": 16,
    }
    hip = Hippocampus(input_dims=input_dims, shared_dim=128, time_dim=16, capacity=512, sparsity=0.07)

    # Simulate one episode across several modalities
    t0 = 0.0
    v = torch.randn(input_dims["vision"])
    cb = torch.randn(input_dims["cerebellum"])
    fr = torch.randn(input_dims["frontal"])
    au = torch.randn(input_dims["auditory"])
    ln = torch.randn(input_dims["language"])
    af = torch.randn(input_dims["affect"])

    # Encode: send a burst of events around the same time window
    hip(Event("vision", v, t=t0), mode="encode")
    hip(Event("cerebellum", cb, t=t0 + 0.05), mode="encode")
    hip(Event("frontal", fr, t=t0 + 0.10), mode="encode")
    hip(Event("auditory", au, t=t0 + 0.15), mode="encode")
    hip(Event("language", ln, t=t0 + 0.20), mode="encode")
    hip(Event("affect", af, t=t0 + 0.25), mode="encode")

    # Retrieve: use a noisy auditory cue near the same time window
    au_cue = au + 0.05 * torch.randn_like(au)
    res = hip(Event("auditory", au_cue, t=t0 + 0.18, prediction=au), mode="retrieve")

    out = res["output"].detach().cpu()
    true_enc = hip.enc(v, "vision").detach().cpu()
    sim_to_vision = F.cosine_similarity(out, true_enc, dim=0).item()

    print("Mode:", res["mode"])
    print("Novelty:", round(res["novelty"], 3))
    print("Cosine(out, vision-enc):", round(sim_to_vision, 3))
    print("Memory size:", hip.mem.K.shape[0])
    print("Replay batch shape:", tuple(hip.replay(batch=6).shape))

if __name__ == "__main__":
    main()
