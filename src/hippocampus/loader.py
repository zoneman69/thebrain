"""Helper that keeps hippocampus decoder artifacts in sync."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from .artifacts import ArtifactsManifest
from .module import Hippocampus

logger = logging.getLogger(__name__)


class HippocampusLoader:
    """Load decoder artifacts onto a live :class:`Hippocampus` instance.

    The loader watches an artifact directory (defaulting to the
    ``HIPPO_ARTIFACTS_DIR`` environment variable) for ``manifest.json``. When a
    new manifest appears or the file's mtime increases, the loader will
    refresh the hippocampus' decoders and fusion layers.
    """

    def __init__(self, hip: Hippocampus, artifact_dir: str | Path | None = None):
        self.hip = hip
        self.artifact_dir = Path(artifact_dir or os.getenv("HIPPO_ARTIFACTS_DIR", "artifacts"))
        self._last_manifest_mtime: Optional[float] = None

    def _load_state_dict(self, module, path: Path) -> None:
        state = torch.load(path, map_location=self.hip.mem.device)
        module.load_state_dict(state)

    def _apply_manifest(self, manifest: ArtifactsManifest, root: Path) -> None:
        for key, rel_path in manifest.artifacts.items():
            full_path = root / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"Artifact {rel_path} missing under {root}")

        for key, rel_path in manifest.artifacts.items():
            full_path = root / rel_path
            if key == "window_fusion":
                self._load_state_dict(self.hip.window_fusion, full_path)
            elif key == "fusion_gate":
                self._load_state_dict(self.hip.fusion_gate, full_path)
            elif key == "mix":
                self._load_state_dict(self.hip.mix, full_path)
            elif key == "time_mixer":
                self._load_state_dict(self.hip.time_mixer, full_path)
            elif key.startswith("decoder_"):
                name = key.replace("decoder_", "", 1)
                decoder = self.hip.decoders.decoders.get(name)
                if decoder is None:
                    logger.warning("Skipping decoder '%s' (not present on this hippocampus)", name)
                    continue
                self._load_state_dict(decoder, full_path)
            else:
                logger.debug("Ignoring unknown artifact key: %s", key)

    def load_latest(self) -> bool:
        """Reload artifacts if ``manifest.json`` is newer.

        Returns ``True`` when a refresh occurred, otherwise ``False``.
        """

        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.exists():
            logger.debug("No manifest found under %s", self.artifact_dir)
            return False

        mtime = manifest_path.stat().st_mtime
        if self._last_manifest_mtime is not None and mtime <= self._last_manifest_mtime:
            return False

        manifest = ArtifactsManifest.load(manifest_path)
        try:
            self._apply_manifest(manifest, manifest_path.parent)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to apply artifacts from %s: %s", manifest_path, exc)
            return False

        self._last_manifest_mtime = mtime
        logger.info("Loaded hippocampus artifacts from %s", manifest_path.parent)
        return True

    # -- Convenience proxies -------------------------------------------------
    def store(self, modality: str, features: torch.Tensor, **kwargs):
        return self.hip.encode(modality, features, **kwargs)

    def recall(self, modality: str, cue: torch.Tensor, **kwargs):
        return self.hip.recall(modality, cue, **kwargs)
