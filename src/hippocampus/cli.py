import argparse
import importlib
import sys
from pathlib import Path

import torch
import yaml

from .module import Hippocampus

DEMO_MAP = {
    "basic": "examples.demo:main",
    "multimodal": "examples.demo_multimodal:main",
    "multimodal_fused": "examples.demo_multimodal_fused:main",
    "viz_attention": "examples.viz_attention:main",
    "bench_recall": "examples.bench_recall:run",
}

ROOT = Path(__file__).resolve().parents[2]
if ROOT.exists() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: str | None):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def instantiate_hippocampus(config: dict) -> Hippocampus:
    hip_cfg = config.get("hippocampus", {})
    if not hip_cfg:
        raise ValueError("hippocampus config missing in file")
    return Hippocampus(**hip_cfg)


def resolve_demo(name: str):
    if name not in DEMO_MAP:
        raise ValueError(f"Unknown demo '{name}'. Options: {', '.join(sorted(DEMO_MAP.keys()))}")
    module_name, func_name = DEMO_MAP[name].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run hippocampus demos via config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--demo", type=str, default="basic", choices=sorted(DEMO_MAP.keys()))
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument("--list", action="store_true", help="List available demos")
    args = parser.parse_args(argv)

    if args.list:
        for key in sorted(DEMO_MAP.keys()):
            print(key)
        return

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    torch.manual_seed(config.get("seed", 0))
    hip = instantiate_hippocampus(config)
    demo_fn = resolve_demo(args.demo)
    demo_fn(config=config, hip=hip)


if __name__ == "__main__":
    main()
