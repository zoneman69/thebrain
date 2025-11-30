"""Conversational agent-style demo for hippocampus."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hippocampus import Hippocampus
from hippocampus.introspection import attention_report, format_memory_table


def embed_text(text: str, dim: int = 256) -> torch.Tensor:
    torch.manual_seed(abs(hash(text)) % (2**32))
    vec = torch.randn(dim)
    return torch.nn.functional.normalize(vec, dim=-1)


def main(config: dict | None = None, hip: Hippocampus | None = None) -> None:
    hippo = hip or Hippocampus({"language": 256}, shared_dim=192, time_dim=32, capacity=48)
    dialogue: List[str] = [
        "hello brain, store this meeting context",
        "the budget discussion happens at 2pm",
        "remember to follow up with alex about metrics",
        "set a reminder to summarize the roadmap",
    ]

    t = 0.0
    print("Encoding conversation...")
    for line in dialogue:
        emb = embed_text(line)
        hippo.encode("language", emb, t=t, context="chat")
        t += hippo.window_size * 1.3
    hippo.flush_pending()

    print(format_memory_table(hippo))

    cue = embed_text("what should i follow up on?")
    res = hippo.recall("language", cue, t=t + 0.5)
    report = attention_report(res["attn"], res.get("metadata", []), top=3)
    print("\nAttention over memory:")
    print(report.to_table())

    # Fake agent response using the top attended metadata window
    top_meta = res.get("metadata", [{}])[0]
    reminder = top_meta.get("modalities", {}).get("language") if isinstance(top_meta, dict) else None
    print("\nAgent response:")
    if reminder is not None:
        print("I recall:", reminder)
    else:
        print("No strong memory match yet, try prompting me again.")


if __name__ == "__main__":
    main()
