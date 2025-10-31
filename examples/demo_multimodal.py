
import logging

import torch
import torch.nn.functional as F

from hippocampus import Event, Hippocampus
from hippocampus.integrations import GraniteModelUnavailable, GraniteNanoEncoder

logger = logging.getLogger(__name__)


def build_multimodal_hippocampus():
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
        window_size=0.35,
        tau=0.15,
    )


def main(config: dict | None = None, hip: Hippocampus | None = None):
    torch.manual_seed((config or {}).get("seed", 42))
    hip = hip or build_multimodal_hippocampus()
    dims = hip.enc.encoders

    t = 0.0
    tB = 10.0
    jitter = {name: 0.02 * torch.randn(dims[name][0].in_features) for name in dims.keys()}
    A = {name: torch.randn(dims[name][0].in_features) for name in dims.keys()}
    B = {name: (A[name] + jitter[name]) for name in dims.keys()}

    lang_dim = dims["language"][0].in_features
    language_texts = [
        "Episode A: a calm living room conversation about robotics on the Raspberry Pi.",
        "Episode B: the same space later in the day discussing remote Runpod training runs.",
    ]
    try:
        granite = GraniteNanoEncoder(target_dim=lang_dim)
        language_embeddings = granite.encode_many(language_texts)
        A["language"], B["language"] = language_embeddings
    except GraniteModelUnavailable as exc:
        logger.warning("Granite Nano unavailable (%s); falling back to gaussian language features", exc)
        # Fallback already initialised in A/B dictionaries

    for tm, episode in [(t, A), (tB, B)]:
        hip(Event("vision", episode["vision"], t=tm), mode="encode")
        hip(Event("cerebellum", episode["cerebellum"], t=tm + 0.05), mode="encode")
        hip(Event("frontal", episode["frontal"], t=tm + 0.10), mode="encode")
        hip(Event("auditory", episode["auditory"], t=tm + 0.15), mode="encode")
        hip(Event("language", episode["language"], t=tm + 0.20), mode="encode")
        hip(Event("affect", episode["affect"], t=tm + 0.25, affect=episode["affect"]), mode="encode")
    hip.flush_pending()

    cue = A["language"] + 0.03 * torch.randn_like(A["language"])
    res = hip(Event("language", cue, t=tB + 0.5, prediction=A["language"]), mode="retrieve")
    out = res["output"].detach()

    visA = hip.enc(A["vision"].to(hip.mem.device), "vision")
    visB = hip.enc(B["vision"].to(hip.mem.device), "vision")
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
