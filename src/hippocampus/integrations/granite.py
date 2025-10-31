"""Optional Granite Nano sentence encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

try:  # pragma: no cover - optional dependency guard
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - import fallback
    AutoModel = AutoTokenizer = None
    _HAS_TRANSFORMERS = False

LOGGER = logging.getLogger(__name__)


class GraniteModelUnavailable(RuntimeError):
    """Raised when Granite models cannot be instantiated."""


@dataclass
class GraniteNanoEncoder:
    """Sentence encoder backed by IBM Granite 4.0 Nano."""

    name: str = "ibm-granite/granite-4.0-h-350m-base"
    device: Optional[str] = None
    use_half: bool = True

    def __post_init__(self) -> None:
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not _HAS_TRANSFORMERS:
            raise GraniteModelUnavailable(
                "transformers is not installed; install hippocampus[granite] to enable Granite Nano"
            )
        dtype = torch.float16 if (self.use_half and torch.cuda.is_available()) else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModel.from_pretrained(self.name, torch_dtype=dtype)
        self._model.to(self._device).eval()
        LOGGER.info("Loaded Granite Nano encoder %s on %s", self.name, self._device)

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True).to(self._device)
        hidden = self._model(**inputs).last_hidden_state.mean(dim=1)
        return hidden.squeeze(0).float().cpu()
