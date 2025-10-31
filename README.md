# hippocampus

A hippocampus-like memory module for AI systems: multi-source episodic binding, pattern separation/completion, novelty-gated encoding, and replay for consolidation.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m examples.demo
```


## New modalities
You can register additional brain-layer inputs by adding them to `input_dims` at construction time.

Included example modalities:
- `auditory`: 256-d auditory/speech embedding
- `language`: 256-d sentence/utterance embedding
- `affect`: 16-d reward/neuromodulator vector (can modulate write strength and replay priority in future)

See `examples/demo_multimodal.py` for a cross-modal retrieval example.
