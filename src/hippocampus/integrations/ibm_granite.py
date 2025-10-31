"""Thin wrapper around IBM Granite Nano embeddings with a deterministic fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency branch
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - runtime import guard
    AutoModel = AutoTokenizer = None
    _HAS_TRANSFORMERS = False


class GraniteModelUnavailable(RuntimeError):
    """Raised when Granite models cannot be loaded and no fallback is allowed."""


@dataclass
class GraniteNanoEncoder:
    """Sentence encoder backed by IBM Granite Nano embeddings.

    The encoder tries to load an official Granite Nano model when the
    :mod:`transformers` package is available. When that is not possible, a
    deterministic noise fallback is used so downstream pipelines can still run
    (useful for unit tests or CPU-only development).
    """

    model_name: str = "ibm-granite/granite-3.0-2b-instruct"
    target_dim: Optional[int] = None
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    max_length: int = 256
    seed: int = 13
    normalize: bool = True

    def __post_init__(self) -> None:
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._generator = torch.Generator().manual_seed(self.seed)
        self._projection: Optional[torch.Tensor] = None
        if _HAS_TRANSFORMERS:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self._model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self._model.to(self._device)
            self._model.eval()
            if self.target_dim is None:
                hidden = getattr(self._model.config, "hidden_size", None)
                if hidden is not None:
                    self.target_dim = int(hidden)
            logger.info("Loaded Granite model %s on %s", self.model_name, self._device)
        else:
            self._tokenizer = None
            self._model = None
            if self.target_dim is None:
                raise GraniteModelUnavailable(
                    "transformers is not installed and no fallback dimension provided; "
                    "install hippocampus[granite] or set target_dim"
                )
            logger.warning(
                "transformers not available; GraniteNanoEncoder will emit deterministic noise of dim=%d",
                self.target_dim,
            )

    def _match_dimension(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.target_dim is None or embedding.shape[-1] == self.target_dim:
            return embedding
        if embedding.shape[-1] > self.target_dim:
            return embedding[..., : self.target_dim]
        pad = self.target_dim - embedding.shape[-1]
        return F.pad(embedding, (0, pad))

    def _project(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.target_dim is None:
            return embedding
        if embedding.shape[-1] == self.target_dim:
            return embedding
        if self._projection is None:
            proj = torch.randn(
                embedding.shape[-1],
                self.target_dim,
                generator=self._generator,
                dtype=embedding.dtype,
            )
            self._projection = proj
        return embedding @ self._projection

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single string into the requested representation."""

        if self._model is None or self._tokenizer is None:
            if self.target_dim is None:
                raise GraniteModelUnavailable("target_dim required when using fallback encoder")
            noise = torch.randn(self.target_dim, generator=self._generator)
            return F.normalize(noise, dim=-1) if self.normalize else noise
        with torch.no_grad():
            tokens = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            outputs = self._model(**tokens)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output[0]
            else:
                hidden_state = outputs.last_hidden_state
                embedding = hidden_state[:, 0, :].squeeze(0)
            embedding = embedding.detach().cpu()
        embedding = self._match_dimension(embedding)
        embedding = self._project(embedding)
        if self.normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

    def encode_many(self, texts: Sequence[str]) -> List[torch.Tensor]:
        return [self.encode(text) for text in texts]

    def encode_batch(self, texts: Iterable[str]) -> torch.Tensor:
        vectors = [self.encode(text) for text in texts]
        if not vectors:
            return torch.empty((0, self.target_dim or 0))
        return torch.stack(vectors, dim=0)
