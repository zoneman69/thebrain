
import torch
from hippocampus import Hippocampus, Event
import torch.nn.functional as F

def main():
    torch.manual_seed(42)
    input_dims = {
        "vision": 512,
        "cerebellum": 64,
        "frontal": 128,
        "auditory": 256,
        "language": 256,
        "affect": 16,
    }
    hip = Hippocampus(input_dims=input_dims, shared_dim=192, time_dim=24, capacity=768, sparsity=0.05)

    # Episode A (scene 1)
    t = 0.0
    A = {
        "vision": torch.randn(input_dims["vision"]),
        "cerebellum": torch.randn(input_dims["cerebellum"]),
        "frontal": torch.randn(input_dims["frontal"]),
        "auditory": torch.randn(input_dims["auditory"]),
        "language": torch.randn(input_dims["language"]),
        "affect": torch.randn(input_dims["affect"]),
    }
    # Episode B (scene 2): near-duplicate vision but different context (time separation)
    tB = 5.0
    B = {k: torch.randn(d) for k, d in input_dims.items()}
    B["vision"] = A["vision"] + 0.1 * torch.randn_like(A["vision"])  # similar scene

    # Encode both episodes
    for i, (tm, ep) in enumerate([(t, A), (tB, B)]):
        hip(Event("vision", ep["vision"], t=tm), mode="encode")
        hip(Event("cerebellum", ep["cerebellum"], t=tm + 0.05), mode="encode")
        hip(Event("frontal", ep["frontal"], t=tm + 0.10), mode="encode")
        hip(Event("auditory", ep["auditory"], t=tm + 0.15), mode="encode")
        hip(Event("language", ep["language"], t=tm + 0.20), mode="encode")
        hip(Event("affect", ep["affect"], t=tm + 0.25), mode="encode")

    # Cue with language from episode A; expect retrieval toward A's fused encoding
    cue = A["language"] + 0.03 * torch.randn_like(A["language"])
    res = hip(Event("language", cue, t=t + 0.22, prediction=A["language"]), mode="retrieve")
    out = res["output"]

    # Compare to vision encodings of A and B
    visA = hip.enc(A["vision"], "vision")
    visB = hip.enc(B["vision"], "vision")
    simA = F.cosine_similarity(out, visA, dim=0).item()
    simB = F.cosine_similarity(out, visB, dim=0).item()

    print("Mode:", res["mode"])
    print("Novelty:", round(res["novelty"], 3))
    print("Similarity to Episode A (vision):", round(simA, 3))
    print("Similarity to Episode B (vision):", round(simB, 3))
    print("Expected: simA > simB (time-based separation)")
    print("Memory size:", hip.mem.K.shape[0])

if __name__ == "__main__":
    main()
