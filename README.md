# hippocampus

A hippocampus-like memory module for AI systems: multi-source episodic binding, pattern separation/completion, novelty-gated encoding, and replay for consolidation.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m examples.demo
```


## Cloud training workflow
- `scripts/train_cloud.py` consumes episodic replay exports (JSON/JSONL or torch `.pt` files
  containing dictionaries with `{"fused": ..., "targets": {...}}`) and writes refreshed
  decoder weights to `artifacts/` (`fused_head.pt`, `mix.pt`, and one file per modality).
- `scripts/pull_push_artifacts.py` automates GitHub release transfers so the Pi can pull
  nightly weight drops and push new replay buffers upstream.

See `.github/workflows/runpod.yml` for an example of wiring these utilities into a CI job.


## New modalities
You can register additional brain-layer inputs by adding them to `input_dims` at construction time.

Included example modalities:
- `auditory`: 256-d auditory/speech embedding
- `language`: 256-d sentence/utterance embedding
- `affect`: 16-d reward/neuromodulator vector (can modulate write strength and replay priority in future)

See `examples/demo_multimodal.py` for a cross-modal retrieval example.

### IBM Granite Nano language embeddings
`examples/demo_multimodal.py` now tries to load IBM's Granite Nano encoder via
`hippocampus.integrations.GraniteNanoEncoder`. Install the optional dependencies with

```bash
pip install -e .[granite]
```

Without `transformers` present the example falls back to deterministic Gaussian language features
so demos continue to run on lightweight Pi deployments.
