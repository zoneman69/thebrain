
import torch
import torch.nn.functional as F
from hippocampus import Hippocampus, Event

def test_encode_retrieve_roundtrip():
    torch.manual_seed(0)
    hip = Hippocampus({"vision": 32}, shared_dim=32, time_dim=8, capacity=16, sparsity=0.1)
    x = torch.randn(32)
    hip(Event("vision", x, t=0.0), mode="encode")
    hip.flush_pending()
    res = hip(Event("vision", x * 0.95, t=0.6, prediction=x), mode="retrieve")
    assert res["output"].shape[-1] == 32
    assert res["mode"] == "retrieve"


def test_fused_window_single_write_and_cross_modal_similarity():
    torch.manual_seed(1)
    input_dims = {"vision": 64, "auditory": 48}
    hip = Hippocampus(input_dims, shared_dim=96, time_dim=32, capacity=32, sparsity=0.05, window_size=0.25)
    episode_time = 0.0
    vision = torch.randn(input_dims["vision"])
    auditory = torch.randn(input_dims["auditory"])
    hip(Event("vision", vision, t=episode_time), mode="encode")
    hip(Event("auditory", auditory, t=episode_time + 0.05), mode="encode")
    hip.flush_pending()
    assert hip.mem.K.shape[0] == 1
    res = hip(Event("auditory", auditory * 0.97, t=episode_time + 0.8, prediction=auditory), mode="retrieve")
    vis_enc = hip.enc(vision.to(hip.mem.device), "vision")
    cos = F.cosine_similarity(res["output"], vis_enc, dim=0).item()
    assert cos >= 0.85


def test_time_separation_delta():
    torch.manual_seed(2)
    input_dims = {
        "vision": 128,
        "language": 64,
        "auditory": 64,
        "affect": 16,
    }
    hip = Hippocampus(input_dims, shared_dim=160, time_dim=64, capacity=128, sparsity=0.04,
                      window_size=0.3, tau=0.05, time_scale=8.0, time_gain=5.0)
    base = {k: torch.randn(d) for k, d in input_dims.items()}
    episode_a_time = 0.0
    episode_b_time = 10.0
    episode_a = {}
    episode_b = {}
    for modality, t in [("vision", episode_a_time),
                        ("language", episode_a_time + 0.05),
                        ("auditory", episode_a_time + 0.1),
                        ("affect", episode_a_time + 0.15)]:
        feat = base[modality]
        affect = feat if modality == "affect" else None
        episode_a[modality] = feat
        hip(Event(modality, feat, t=t, affect=affect), mode="encode")
    for modality, t in [("vision", episode_b_time),
                        ("language", episode_b_time + 0.05),
                        ("auditory", episode_b_time + 0.1),
                        ("affect", episode_b_time + 0.15)]:
        jitter = 0.6 * torch.randn_like(base[modality])
        base_feat = base[modality]
        if modality == "vision":
            feat = -10.0 * base_feat + jitter
        else:
            feat = base_feat + jitter
        affect = (feat * -1.0) if modality == "affect" else None
        episode_b[modality] = feat
        hip(Event(modality, feat, t=t, affect=affect), mode="encode")
    hip.flush_pending()
    cue = base["auditory"] + 0.02 * torch.randn_like(base["auditory"])
    res = hip(Event("auditory", cue, t=episode_a_time + 0.4, prediction=base["auditory"]), mode="retrieve")
    vis_a = hip.enc(episode_a["vision"].to(hip.mem.device), "vision")
    vis_b = hip.enc(episode_b["vision"].to(hip.mem.device), "vision")
    cos_a = F.cosine_similarity(res["output"], vis_a, dim=0).item()
    cos_b = F.cosine_similarity(res["output"], vis_b, dim=0).item()
    assert cos_a - cos_b >= 0.25


def test_lru_and_capacity_controls():
    torch.manual_seed(3)
    hip = Hippocampus({"vision": 16}, shared_dim=24, capacity=8, sparsity=0.1, max_total=3, max_per_context=2)
    for i in range(3):
        vec = torch.randn(16)
        hip(Event("vision", vec, t=float(i), context="A"), mode="encode")
        hip.flush_pending()
    assert hip.mem.K.shape[0] == 2  # capped per context
    # Access first slot to make it recent
    hip(Event("vision", torch.randn(16), t=3.0, prediction=torch.randn(16)), mode="retrieve")
    hip(Event("vision", torch.randn(16), t=4.0, context="B"), mode="encode")
    hip.flush_pending()
    assert hip.mem.K.shape[0] <= 3
    contexts = set(hip.mem.contexts)
    assert contexts == {"A", "B"}


def test_affect_strength_biases_replay():
    torch.manual_seed(4)
    input_dims = {"vision": 32, "affect": 8}
    hip = Hippocampus(input_dims, shared_dim=48, capacity=8, sparsity=0.05, window_size=0.2)
    high_affect = torch.ones(8)
    low_affect = -torch.ones(8)
    hip(Event("affect", high_affect, t=0.0, affect=high_affect), mode="encode")
    hip(Event("vision", torch.randn(32), t=0.05), mode="encode")
    hip.flush_pending()
    hip(Event("affect", low_affect, t=1.0, affect=low_affect), mode="encode")
    hip(Event("vision", torch.randn(32), t=1.05), mode="encode")
    hip.flush_pending()
    strengths = hip.mem.strength.cpu()
    assert strengths.max() > strengths.min()
    hits = torch.zeros_like(strengths)
    for _ in range(32):
        replay = hip.replay(batch=1)
        samples = replay["values"]
        if samples.numel() == 0:
            continue
        idx = replay["indices"][0]
        hits[idx] += 1
    assert int(torch.argmax(hits)) == int(torch.argmax(strengths))
