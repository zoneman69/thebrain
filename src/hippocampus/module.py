
import math
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_time_pos_enc(t: float, d: int) -> torch.Tensor:
    pe = torch.zeros(d)
    position = torch.tensor([t], dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[0::2] = torch.sin(position * div_term)
    pe[1::2] = torch.cos(position * div_term)
    return pe

def kwta(x: torch.Tensor, sparsity: float = 0.05) -> torch.Tensor:
    if sparsity <= 0 or sparsity >= 1:
        return x
    k = max(1, int(x.shape[-1] * sparsity))
    topk, _ = torch.topk(x, k=k, dim=-1)
    thresh = topk[..., -1].unsqueeze(-1)
    mask = (x >= thresh).float()
    return x * mask

class ModalityEncoder(nn.Module):
    def __init__(self, input_dims: Dict[str, int], out_dim: int):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d, 2*out_dim),
                nn.ReLU(),
                nn.Linear(2*out_dim, out_dim)
            ) for name, d in input_dims.items()
        })
        self.mod_tags = nn.ParameterDict({
            name: nn.Parameter(torch.randn(out_dim)) for name in input_dims.keys()
        })
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}")
        h = self.encoders[modality](x)
        return h + self.mod_tags[modality]

class FastHebbMemory(nn.Module):
    def __init__(self, dim: int, capacity: int = 512, tau: float = 0.2, device: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.tau = tau
        self.register_buffer("K", torch.zeros((0, dim)))
        self.register_buffer("V", torch.zeros((0, dim)))
        self.register_buffer("strength", torch.zeros(0))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def write(self, key: torch.Tensor, value: torch.Tensor, strength: float = 1.0):
        key = F.normalize(key.detach(), dim=-1).unsqueeze(0)
        value = value.detach().unsqueeze(0)
        s = torch.tensor([strength], device=self.device)
        if self.K.shape[0] >= self.capacity:
            self.K = self.K[1:]; self.V = self.V[1:]; self.strength = self.strength[1:]
        self.K = torch.cat([self.K, key.to(self.device)], dim=0)
        self.V = torch.cat([self.V, value.to(self.device)], dim=0)
        self.strength = torch.cat([self.strength, s], dim=0)

    def read(self, query: torch.Tensor, topk: int = 32):
        if self.K.shape[0] == 0:
            return torch.zeros(self.dim, device=self.device), {"attn": None}
        q = F.normalize(query, dim=-1).unsqueeze(0)
        logits = (q @ self.K.T) / self.tau
        if topk is not None and self.K.shape[0] > topk:
            vals, idxs = torch.topk(logits, k=topk, dim=-1)
            attn = F.softmax(vals, dim=-1)
            out = (attn @ self.V[idxs.squeeze(0)]).squeeze(0)
            full_attn = torch.zeros_like(logits)
            full_attn[0, idxs.squeeze(0)] = attn
            return out, {"attn": full_attn.squeeze(0).detach().cpu()}
        attn = F.softmax(logits, dim=-1)
        out = (attn @ self.V).squeeze(0)
        return out, {"attn": attn.squeeze(0).detach().cpu()}

    def decay(self, lam: float = 0.001):
        if self.strength.numel() == 0:
            return
        self.strength = (self.strength * (1 - lam)).clamp(min=0.0)
        keep = self.strength > 1e-4
        self.K = self.K[keep]; self.V = self.V[keep]; self.strength = self.strength[keep]

    def sample(self, batch: int = 8) -> torch.Tensor:
        if self.K.shape[0] == 0:
            return torch.empty((0, self.dim), device=self.device)
        probs = (self.strength + 1e-6) / torch.sum(self.strength + 1e-6)
        idx = torch.multinomial(probs, num_samples=min(batch, self.K.shape[0]), replacement=False)
        return self.V[idx]

@dataclass
class Event:
    modality: str
    features: torch.Tensor
    t: float
    prediction: Optional[torch.Tensor] = None

class Hippocampus(nn.Module):
    def __init__(self, input_dims: Dict[str, int], shared_dim: int = 256, time_dim: int = 32,
                 capacity: int = 1024, sparsity: float = 0.05, novelty_threshold: float = 0.3):
        super().__init__()
        self.enc = ModalityEncoder(input_dims, shared_dim)
        self.sep_sparsity = sparsity
        self.time_dim = time_dim
        self.novelty_threshold = novelty_threshold
        self.mix = nn.Sequential(
            nn.Linear(shared_dim + time_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        self.mem = FastHebbMemory(dim=shared_dim, capacity=capacity)
        self.to(self.mem.device)

    def encode_dg(self, x: torch.Tensor, t: float) -> torch.Tensor:
        time_code = sinusoidal_time_pos_enc(t, self.time_dim).to(x.device)
        h = torch.cat([x, time_code], dim=-1)
        h = self.mix(h)
        h = kwta(h, sparsity=self.sep_sparsity)
        return h

    @staticmethod
    def novelty(a: torch.Tensor, b: Optional[torch.Tensor]) -> float:
        if b is None: return 1.0
        a_n = F.normalize(a, dim=-1); b_n = F.normalize(b, dim=-1)
        sim = torch.sum(a_n * b_n).item()
        return float(max(0.0, 1.0 - sim))

    def forward(self, event: Event, mode: Optional[str] = None):
        x = self.enc(event.features.to(self.mem.device), event.modality)
        z = self.encode_dg(x, t=event.t)
        pred_enc = None
        if event.prediction is not None:
            pred_enc = self.enc(event.prediction.to(self.mem.device), event.modality).detach()
        nov = self.novelty(x, pred_enc)
        chosen = mode or ("encode" if nov > self.novelty_threshold else "retrieve")

        if chosen == "encode":
            self.mem.write(key=z, value=x, strength=1.0)
            out, attn = x.detach(), None
        elif chosen == "retrieve":
            out, info = self.mem.read(query=z, topk=32); attn = info["attn"]
        else:
            raise ValueError("mode must be None, 'encode', or 'retrieve'")
        return {"output": out, "novelty": nov, "mode": chosen, "attn": attn}

    def replay(self, batch: int = 8) -> torch.Tensor:
        return self.mem.sample(batch=batch)

    def decay(self, lam: float = 0.001):
        self.mem.decay(lam=lam)
