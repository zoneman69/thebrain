
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("hippocampus")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    def __init__(self, dim: int, capacity: int = 512, tau: float = 0.2,
                 device: Optional[str] = None, max_total: Optional[int] = None,
                 max_per_context: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.tau = tau
        self.max_total = max_total or capacity
        self.max_per_context = max_per_context
        self.register_buffer("K", torch.zeros((0, dim)))
        self.register_buffer("V", torch.zeros((0, dim)))
        self.register_buffer("strength", torch.zeros(0))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.contexts: List[str] = []
        self.metadata: List[Dict[str, torch.Tensor]] = []
        self.register_buffer("last_used", torch.zeros(0))
        self.access_counter = 0.0
        self.to(self.device)

    def _evict_indices(self, indices: List[int]):
        if not indices:
            return
        keep = torch.ones(self.K.shape[0], dtype=torch.bool, device=self.device)
        keep[torch.tensor(indices, device=self.device)] = False
        self.K = self.K[keep]
        self.V = self.V[keep]
        self.strength = self.strength[keep]
        self.last_used = self.last_used[keep]
        new_contexts = []
        new_metadata = []
        for i, keep_flag in enumerate(keep.tolist()):
            if keep_flag:
                new_contexts.append(self.contexts[i])
                new_metadata.append(self.metadata[i])
        self.contexts = new_contexts
        self.metadata = new_metadata

    def _ensure_capacity(self, context: str):
        current_total = self.K.shape[0]
        if current_total >= self.max_total:
            # Evict global least recently used
            lru_idx = int(torch.argmin(self.last_used).item()) if current_total > 0 else None
            if lru_idx is not None:
                self._evict_indices([lru_idx])
        if self.max_per_context is not None:
            ctx_indices = [i for i, c in enumerate(self.contexts) if c == context]
            if len(ctx_indices) >= self.max_per_context:
                # Evict least recently used within context
                ctx_last = self.last_used[ctx_indices]
                rel_idx = int(torch.argmin(ctx_last).item())
                self._evict_indices([ctx_indices[rel_idx]])

    def write(self, key: torch.Tensor, value: torch.Tensor, strength: float = 1.0,
              context: Optional[str] = None, metadata: Optional[Dict[str, torch.Tensor]] = None):
        context = context or "global"
        key = F.normalize(key.detach(), dim=-1).unsqueeze(0)
        value = value.detach().unsqueeze(0)
        s = torch.tensor([strength], device=self.device)
        self._ensure_capacity(context)
        self.K = torch.cat([self.K, key.to(self.device)], dim=0)
        self.V = torch.cat([self.V, value.to(self.device)], dim=0)
        self.strength = torch.cat([self.strength, s], dim=0)
        self.last_used = torch.cat([self.last_used, torch.tensor([self.access_counter], device=self.device)], dim=0)
        self.contexts.append(context)
        self.metadata.append(metadata or {})

    def read(self, query: torch.Tensor, topk: int = 32):
        if self.K.shape[0] == 0:
            return torch.zeros(self.dim, device=self.device), {"attn": None, "indices": []}
        q = F.normalize(query, dim=-1).unsqueeze(0)
        logits = (q @ self.K.T) / self.tau
        meta = []
        if topk is not None and self.K.shape[0] > topk:
            vals, idxs = torch.topk(logits, k=topk, dim=-1)
            attn = F.softmax(vals, dim=-1)
            value_sel = self.V[idxs.squeeze(0)]
            out = (attn @ value_sel).squeeze(0)
            full_attn = torch.zeros_like(logits)
            full_attn[0, idxs.squeeze(0)] = attn
            indices = idxs.squeeze(0).tolist()
            meta = [self.metadata[i] for i in indices]
        else:
            attn = F.softmax(logits, dim=-1)
            out = (attn @ self.V).squeeze(0)
            full_attn = attn
            indices = list(range(self.K.shape[0]))
            meta = [self.metadata[i] for i in indices]
        # Update LRU timestamps
        if indices:
            idx_tensor = torch.tensor(indices, device=self.device)
            self.access_counter += 1.0
            self.last_used[idx_tensor] = self.access_counter
        return out, {"attn": full_attn.squeeze(0).detach().cpu(), "indices": indices, "metadata": meta}

    def decay(self, lam: float = 0.001):
        if self.strength.numel() == 0:
            return
        self.strength = (self.strength * (1 - lam)).clamp(min=0.0)
        keep = self.strength > 1e-4
        self._evict_indices([i for i, flag in enumerate(keep.tolist()) if not flag])

    def sample(self, batch: int = 8):
        if self.K.shape[0] == 0:
            return {"values": torch.empty((0, self.dim), device=self.device),
                    "metadata": [], "indices": []}
        probs = (self.strength + 1e-6)
        probs = probs / torch.sum(probs)
        idx = torch.multinomial(probs, num_samples=min(batch, self.K.shape[0]), replacement=False)
        meta = [self.metadata[i] for i in idx.tolist()]
        return {"values": self.V[idx], "metadata": meta, "indices": idx.tolist()}

@dataclass
class Event:
    modality: str
    features: torch.Tensor
    t: float
    prediction: Optional[torch.Tensor] = None
    context: Optional[str] = None
    affect: Optional[torch.Tensor] = None


@dataclass
class PendingWindow:
    window_id: int
    t_start: float
    modalities: Dict[str, torch.Tensor] = field(default_factory=dict)
    raw: Dict[str, torch.Tensor] = field(default_factory=dict)
    affects: List[torch.Tensor] = field(default_factory=list)
    context: Optional[str] = None
    last_t: float = 0.0

class Hippocampus(nn.Module):
    def __init__(self, input_dims: Dict[str, int], shared_dim: int = 256, time_dim: int = 64,
                 capacity: int = 1024, sparsity: float = 0.04, novelty_threshold: float = 0.3,
                 window_size: float = 0.5, tau: float = 0.2, time_scale: float = 1.0, time_gain: float = 1.0,
                 max_total: Optional[int] = None, max_per_context: Optional[int] = None,
                 orth_penalty: float = 0.0):
        super().__init__()
        self.enc = ModalityEncoder(input_dims, shared_dim)
        self.modalities = list(input_dims.keys())
        self.sep_sparsity = sparsity
        self.time_dim = time_dim
        self.time_scale = time_scale
        self.time_gain = time_gain
        self.novelty_threshold = novelty_threshold
        self.window_size = window_size
        self.orth_penalty = orth_penalty
        self.mix = nn.Sequential(
            nn.Linear(shared_dim + time_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        self.time_mixer = nn.Linear(time_dim, shared_dim)
        self.mem = FastHebbMemory(dim=shared_dim, capacity=capacity, tau=tau,
                                  max_total=max_total or capacity, max_per_context=max_per_context)
        self.fusion = nn.Linear(shared_dim * 2, shared_dim)
        with torch.no_grad():
            eye = torch.eye(shared_dim)
            self.fusion.weight[:] = torch.cat([0.5 * eye, 0.5 * eye], dim=1)
            self.fusion.bias.zero_()
        self.modality_gates = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in self.modalities
        })
        if "vision" in self.modality_gates:
            self.modality_gates["vision"].data += 4.0
        affect_dim = input_dims.get("affect")
        if affect_dim is not None:
            self.affect_linear = nn.Linear(affect_dim, 1)
            with torch.no_grad():
                self.affect_linear.weight.fill_(1.0 / max(1, affect_dim))
                self.affect_linear.bias.zero_()
        else:
            self.affect_linear = None
        self.pending: Dict[int, PendingWindow] = {}
        self.to(self.mem.device)

    def encode_dg(self, x: torch.Tensor, t: float) -> torch.Tensor:
        time_code = sinusoidal_time_pos_enc(t * self.time_scale, self.time_dim).to(x.device)
        time_emb = self.time_mixer(time_code) * self.time_gain
        combined = x + time_emb
        h = torch.cat([combined, time_code], dim=-1)
        h = self.mix(h)
        h = kwta(h, sparsity=self.sep_sparsity)
        return h

    def _modalities_per_window(self) -> int:
        return len(self.modalities)

    def _get_window(self, window_id: int, t: float, context: Optional[str]) -> PendingWindow:
        if window_id not in self.pending:
            self.pending[window_id] = PendingWindow(window_id=window_id, t_start=window_id * self.window_size,
                                                    context=context, last_t=t)
        win = self.pending[window_id]
        if context is not None and win.context is None:
            win.context = context
        win.last_t = max(win.last_t, t)
        return win

    def _flush_window(self, window_id: int):
        win = self.pending.pop(window_id, None)
        if win is None or not win.modalities:
            return
        names = sorted(win.modalities.keys())
        device = self.mem.device
        encs = torch.stack([win.modalities[n].to(device) for n in names], dim=0)
        avg_enc = encs.mean(dim=0)
        gate_logits = torch.stack([self.modality_gates[n] for n in names]).squeeze(-1)
        weights = torch.softmax(gate_logits, dim=0).unsqueeze(-1)
        fused = torch.sum(weights * encs, dim=0)
        time_code = sinusoidal_time_pos_enc((win.t_start + self.window_size * 0.5) * self.time_scale, self.time_dim).to(device)
        time_emb = self.time_mixer(time_code) * self.time_gain
        agg_key = self.fusion(torch.cat([fused, avg_enc], dim=-1)) + time_emb
        fused_value = fused + time_emb
        window_center = win.t_start + self.window_size * 0.5
        key = self.encode_dg(agg_key, t=window_center)
        metadata = {"modalities": {n: win.modalities[n].detach().cpu() for n in names},
                    "raw": {n: win.raw[n].detach().cpu() for n in names},
                    "t_window": (win.t_start, win.last_t),
                    "window_id": window_id,
                    "context": win.context or "global"}
        strength = 1.0
        if self.affect_linear is not None and win.affects:
            affect_stack = torch.stack([a.to(self.affect_linear.weight.device) for a in win.affects], dim=0)
            mean_affect = affect_stack.mean(dim=0)
            strength = F.softplus(self.affect_linear(mean_affect)).item()
        self.mem.write(key=key, value=fused_value, strength=strength, context=win.context, metadata=metadata)

    def _flush_due_windows(self, current_t: Optional[float] = None):
        to_flush = []
        for wid, win in self.pending.items():
            if len(win.modalities) >= self._modalities_per_window():
                to_flush.append(wid)
            elif current_t is not None and current_t - win.last_t >= self.window_size:
                to_flush.append(wid)
        for wid in to_flush:
            self._flush_window(wid)

    def flush_pending(self):
        for wid in list(self.pending.keys()):
            self._flush_window(wid)

    @staticmethod
    def novelty(a: torch.Tensor, b: Optional[torch.Tensor]) -> float:
        if b is None: return 1.0
        a_n = F.normalize(a, dim=-1); b_n = F.normalize(b, dim=-1)
        sim = torch.sum(a_n * b_n).item()
        return float(max(0.0, 1.0 - sim))

    def forward(self, event: Event, mode: Optional[str] = None):
        device = self.mem.device
        features = event.features.to(device)
        x = self.enc(features, event.modality)
        pred_enc = None
        if event.prediction is not None:
            pred_enc = self.enc(event.prediction.to(device), event.modality).detach()
        nov = self.novelty(x, pred_enc)
        chosen = mode or ("encode" if nov > self.novelty_threshold else "retrieve")
        z = self.encode_dg(x, t=event.t)
        orth = 0.0
        if self.orth_penalty > 0 and self.mem.K.shape[0] > 0:
            key_norm = F.normalize(z, dim=-1)
            sims = torch.abs(torch.matmul(key_norm.unsqueeze(0), self.mem.K.T))
            orth = float(self.orth_penalty * sims.mean().item())

        indices: List[int] = []
        attn = None
        metadata = []
        if chosen == "encode":
            bucket = int(event.t / self.window_size)
            window = self._get_window(bucket, event.t, event.context)
            window.modalities[event.modality] = x.detach()
            window.raw[event.modality] = event.features.detach().cpu()
            if event.affect is not None:
                window.affects.append(event.affect.detach())
            self._flush_due_windows(event.t)
            out = x.detach()
        elif chosen == "retrieve":
            self._flush_due_windows(event.t + self.window_size)
            out, info = self.mem.read(query=z, topk=32)
            attn = info["attn"]
            indices = info.get("indices", [])
            metadata = info.get("metadata", [])
        else:
            raise ValueError("mode must be None, 'encode', or 'retrieve'")

        log_payload = {
            "event_modality": event.modality,
            "t": event.t,
            "mode": chosen,
            "novelty": nov,
            "memory_size": int(self.mem.K.shape[0]),
            "pending_windows": len(self.pending),
        }
        if attn is not None and indices:
            log_payload["attn_indices"] = indices[:5]
            log_payload["attn_scores"] = [float(attn[i].item()) for i in indices[:5]]
        logger.info(json.dumps(log_payload))

        return {"output": out, "novelty": nov, "mode": chosen, "attn": attn, "indices": indices,
                "metadata": metadata, "orth_penalty": orth}

    def replay(self, batch: int = 8):
        return self.mem.sample(batch=batch)

    def decay(self, lam: float = 0.001):
        self.mem.decay(lam=lam)
