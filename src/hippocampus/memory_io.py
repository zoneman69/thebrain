import torch
from pathlib import Path


def save_memory(hip, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "K": hip.mem.K.cpu(),
        "V": hip.mem.V.cpu(),
        "t": hip.mem.t.cpu(),
        "meta": hip.mem.meta,
        "v": 1,
    }
    torch.save(state, p)


def load_memory(hip, path: str):
    s = torch.load(path, map_location="cpu")
    hip.mem.K = s["K"].to(hip.mem.device)
    hip.mem.V = s["V"].to(hip.mem.device)
    hip.mem.t = s["t"].to(hip.mem.device)
    hip.mem.meta = s.get("meta", [])
