
import torch
from hippocampus import Hippocampus, Event

def test_encode_retrieve_roundtrip():
    torch.manual_seed(0)
    hip = Hippocampus({"vision": 32}, shared_dim=32, time_dim=8, capacity=16, sparsity=0.1)
    x = torch.randn(32)
    hip(Event("vision", x, t=0.0), mode="encode")
    res = hip(Event("vision", x*0.95, t=0.1, prediction=x), mode="retrieve")
    assert res["output"].shape[-1] == 32
    assert res["mode"] == "retrieve"
