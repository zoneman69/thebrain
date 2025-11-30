# hippocampus

A hippocampus-like memory module for AI systems: multi-source episodic binding, pattern separation/completion, novelty-gated encoding, and replay for consolidation.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m examples.demo
```

The public Python surface is centered on the `Hippocampus` class:

```python
from hippocampus import Hippocampus

hip = Hippocampus({"language": 256})
hip.encode("language", features, t=0.0)
hip.flush_pending()
recall = hip.recall("language", cue, t=0.5, decode_modalities=["language"])
records = hip.export("logs/replay.pt")  # fused+targets for cloud training
```

`encode`, `recall`, and `export` are now stable entry points; demos and scripts use them instead of raw `forward` calls.

## Architecture
The hippocampus binds temporally co-occurring modality encodings into fused vectors that live in a sparse Hebbian store. Each write window collects one event per modality, mixes time codes, and writes a key/value pair. Reads perform novelty gating, attention over stored keys, and optional temporal penalties before decoding back into modalities.

Key components:
- **Modality encoder/decoder heads** (`Decoders` in `src/hippocampus/decoders.py`)
- **Hebbian memory** (`FastHebbMemory` in `src/hippocampus/module.py`)
- **Window fusion + time mixers** (gating layers saved in artifact manifests)

## IBM Granite Nano language embeddings
`examples/demo_language_granite.py` demonstrates a language-only loop that prefers IBM's Granite Nano encoder via `hippocampus.integrations.GraniteNanoEncoder`. Install the optional dependencies with

```bash
pip install -e .[granite]
```

Without `transformers` present the demo falls back to deterministic Gaussian embeddings so it remains lightweight.

## Demos
- `hippo.yaml` + `python -m hippocampus.cli --demo basic --config hippo.yaml` (simple encoding)
- `--demo multimodal` or `multimodal_fused` (cross-modal completion)
- `--demo viz_attention` (prints attention over stored windows)
- `--demo language_granite` (Granite Nano or fallback language-only example)
- `--demo agent` (conversational/agent flavored retrieval)

## Cloud training workflow
- `hippocampus.Hippocampus.export()` writes replay logs (`{"fused": tensor, "targets": {...}}`) for training.
- `scripts/train_cloud.py` consumes those logs (JSON/JSONL or torch `.pt`) and writes refreshed decoder weights to `artifacts/`.
- Artifacts follow a manifest contract defined in `src/hippocampus/artifacts.py` and can be validated with `python -m hippocampus.artifacts <artifact_dir> --check --show`.
- `scripts/pull_push_artifacts.py` automates GitHub release transfers so the Pi can pull
  nightly weight drops and push new replay buffers upstream.

See `.github/workflows/runpod.yml` for an example of wiring these utilities into a CI job.

## New modalities
You can register additional brain-layer inputs by adding them to `input_dims` at construction time.

Included example modalities:
- `auditory`: 256-d auditory/speech embedding
- `language`: 256-d sentence/utterance embedding
- `affect`: 16-d reward/neuromodulator vector (can modulate write strength and replay priority in future)

See `examples/demo_multimodal.py` for a cross-modal retrieval example and `src/hippocampus/introspection.py` for debugging helpers.
