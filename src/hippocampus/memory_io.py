import torch
from pathlib import Path
from typing import Any, Dict


def save_memory(hip, path: str) -> None:
    """Persist raw memory tensors plus metadata to disk."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "K": hip.mem.K.cpu(),
        "V": hip.mem.V.cpu(),
        "strength": getattr(hip.mem, "strength", torch.tensor([])).cpu(),
        "times": getattr(hip.mem, "times", torch.tensor([])).cpu(),
        "last_used": getattr(hip.mem, "last_used", torch.tensor([])).cpu(),
        "metadata": getattr(hip.mem, "metadata", []),
        "v": 2,
    }
    torch.save(state, p)


def load_memory(hip, path: str) -> None:
    """Load a previously saved memory snapshot."""

    s = torch.load(path, map_location="cpu")
    hip.mem.K = s["K"].to(hip.mem.device)
    hip.mem.V = s["V"].to(hip.mem.device)
    hip.mem.strength = s.get("strength", torch.zeros(0)).to(hip.mem.device)
    hip.mem.times = s.get("times", torch.zeros(0)).to(hip.mem.device)
    hip.mem.last_used = s.get("last_used", torch.zeros(0)).to(hip.mem.device)
    hip.mem.metadata = s.get("metadata", [])
