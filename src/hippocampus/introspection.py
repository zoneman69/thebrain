"""Utilities for inspecting hippocampus memory and attention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch


@dataclass
class AttentionReport:
    """Structured view of a recall attention pattern."""

    indices: List[int]
    scores: List[float]
    contexts: List[str]
    windows: List[str]

    def to_table(self) -> str:
        header = "idx | score  | context | window"
        lines = [header, "-" * len(header)]
        for idx, score, ctx, window in zip(self.indices, self.scores, self.contexts, self.windows):
            lines.append(f"{idx:>3} | {score:>6.4f} | {ctx:<7} | {window}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "indices": self.indices,
                "scores": self.scores,
                "contexts": self.contexts,
                "windows": self.windows,
            },
            indent=2,
        )


def memory_summary(hip) -> Dict[str, int]:
    """Return a compact dictionary describing the current memory state."""

    contexts = hip.mem.contexts if hasattr(hip.mem, "contexts") else []
    return {
        "slots": int(getattr(hip.mem, "capacity", 0)),
        "occupied": int(getattr(hip.mem, "K", torch.empty(0)).shape[0]),
        "contexts": len(set(contexts)),
    }


def attention_report(attn: torch.Tensor, metadata: Iterable[dict], top: int = 5) -> AttentionReport:
    """Format attention weights and metadata into a human-friendly report."""

    attn = attn.detach().cpu()
    scores, indices = torch.topk(attn, k=min(top, attn.shape[-1]))
    contexts: List[str] = []
    windows: List[str] = []
    meta_list = list(metadata)
    for idx in indices.tolist():
        meta = meta_list[idx] if idx < len(meta_list) else {}
        ctx = meta.get("context", "global") if isinstance(meta, dict) else "global"
        window = meta.get("t_window", (None, None)) if isinstance(meta, dict) else (None, None)
        windows.append(f"{window[0]}–{window[1]}")
        contexts.append(str(ctx))
    return AttentionReport(indices=indices.tolist(), scores=scores.tolist(), contexts=contexts, windows=windows)


def dump_memory(hip) -> List[Dict]:
    """Convert each memory slot into a JSON-friendly dictionary for debugging."""

    items: List[Dict] = []
    for idx, (key, value, meta) in enumerate(zip(hip.mem.K.detach().cpu(), hip.mem.V.detach().cpu(), hip.mem.metadata)):
        items.append(
            {
                "idx": idx,
                "key_norm": float(torch.linalg.vector_norm(key)),
                "value_norm": float(torch.linalg.vector_norm(value)),
                "context": meta.get("context") if isinstance(meta, dict) else None,
                "modalities": sorted(meta.get("modalities", {}).keys()) if isinstance(meta, dict) else [],
                "t_window": meta.get("t_window") if isinstance(meta, dict) else None,
            }
        )
    return items


def format_memory_table(hip) -> str:
    """Render a small ASCII table of memory contents."""

    lines = ["idx | context | window | modalities", "------------------------------------"]
    for item in dump_memory(hip):
        lines.append(
            f"{item['idx']:>3} | {str(item['context']):<7} | {item['t_window']} | {','.join(item['modalities'])}"
        )
    return "\n".join(lines)
