import torch
import torch.nn as nn
from typing import Dict, Iterable


class DecoderHead(nn.Module):
    """Simple MLP decoder mapping shared space back to modality space."""

    def __init__(self, shared_dim: int, out_dim: int, hidden_multiplier: int = 2):
        super().__init__()
        hidden_dim = max(shared_dim, hidden_multiplier * shared_dim)
        self.net = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoders(nn.Module):
    """Registry of modality-specific decoder heads."""

    def __init__(self, shared_dim: int, output_dims: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {modality: DecoderHead(shared_dim, dim) for modality, dim in output_dims.items()}
        )

    def available_modalities(self) -> Iterable[str]:
        return self.heads.keys()

    def forward(self, x: torch.Tensor, modalities: Iterable[str]) -> Dict[str, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        outputs: Dict[str, torch.Tensor] = {}
        for modality in modalities:
            if modality not in self.heads:
                raise KeyError(f"Unknown modality requested for decoding: {modality}")
            decoded = self.heads[modality](x)
            outputs[modality] = decoded.squeeze(0) if squeeze else decoded
        return outputs
