"""Online ridge regression decoders for hippocampal fusion vectors."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn

class OnlineRidgeDecoder(nn.Module):
    """Maintain closed-form ridge regression weights for one modality."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lam: float = 1e-2,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lam = lam
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.register_buffer("weights", torch.zeros(input_dim, output_dim, device=dev))
        self.register_buffer("xtx", torch.zeros(input_dim, input_dim, device=dev))
        self.register_buffer("xty", torch.zeros(input_dim, output_dim, device=dev))
        self.register_buffer("count", torch.tensor(0.0, device=dev))

    @property
    def device(self) -> torch.device:
        return self.weights.device

    def reset_parameters(self) -> None:
        self.weights.zero_()
        self.xtx.zero_()
        self.xty.zero_()
        self.count.zero_()

    def _solve(self) -> None:
        if self.count.item() == 0:
            return
        reg_matrix = self.xtx + self.lam * torch.eye(self.input_dim, device=self.device)
        try:
            solution = torch.linalg.solve(reg_matrix, self.xty)
        except RuntimeError:
            solution = torch.linalg.lstsq(reg_matrix, self.xty).solution
        self.weights.copy_(solution)

    def observe(self, fused: torch.Tensor, target: torch.Tensor) -> None:
        if fused.ndim == 1:
            fused = fused.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        fused = fused.to(self.device)
        target = target.to(self.device)
        self.xtx += fused.T @ fused
        self.xty += fused.T @ target
        self.count += fused.shape[0]
        self._solve()

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim == 1:
            return fused.to(self.device) @ self.weights
        return fused.to(self.device) @ self.weights


class Decoders(nn.Module):
    """Registry of modality-specific online ridge decoders."""

    def __init__(
        self,
        shared_dim: int,
        output_dims: Mapping[str, int],
        lam: float = 1e-2,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.decoders = nn.ModuleDict(
            {
                name: OnlineRidgeDecoder(shared_dim, dim, lam=lam, device=dev)
                for name, dim in output_dims.items()
            }
        )

    def available_modalities(self) -> Iterable[str]:
        return self.decoders.keys()

    def observe(
        self, fused: torch.Tensor, targets: Mapping[str, torch.Tensor]
    ) -> None:
        if not targets:
            return
        for modality, target in targets.items():
            if modality not in self.decoders:
                continue
            self.decoders[modality].observe(fused, target)

    def decode(self, modality: str, fused: torch.Tensor) -> torch.Tensor:
        if modality not in self.decoders:
            raise KeyError(f"Unknown modality '{modality}' for decoding")
        return self.decoders[modality](fused)

    def decode_many(
        self, fused: torch.Tensor, modalities: Optional[Iterable[str]] = None
    ) -> Dict[str, torch.Tensor]:
        mods = list(modalities) if modalities is not None else list(self.available_modalities())
        return {mod: self.decode(mod, fused) for mod in mods}

