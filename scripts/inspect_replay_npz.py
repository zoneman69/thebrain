#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def inspect_npz(path: Path, max_items: int = 3) -> None:
    print(f"\n=== {path} ===")
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"  ERROR: failed to load NPZ: {exc}")
        return

    keys = list(data.keys())
    print(f"  Keys: {keys}")

    for key in keys:
        arr = data[key]
        print(f"  [{key}]")
        print(f"    dtype: {arr.dtype}")
        print(f"    shape: {arr.shape}")

        # If this looks like a collection of records, show a few
        if arr.size == 0:
            print("    (empty)")
            continue

        # Handle 1D collections specially
        if arr.ndim == 1 and arr.shape[0] > 0:
            print(f"    first {min(max_items, arr.shape[0])} item(s):")
            for i in range(min(max_items, arr.shape[0])):
                item = arr[i]
                print(f"      [{i}] type={type(item)}")

                # Try to introspect dict-like records
                if isinstance(item, dict):
                    sample_keys = list(item.keys())
                    print(f"         dict keys: {sample_keys}")
                    for sk in sample_keys:
                        v = item[sk]
                        if isinstance(v, np.ndarray):
                            print(f"         {sk}: ndarray shape={v.shape} dtype={v.dtype}")
                        else:
                            print(f"         {sk}: type={type(v)} value={repr(v)[:64]}")
                else:
                    # Just print a truncated representation
                    print(f"         value={repr(item)[:120]}")
        else:
            # Non-1D arrays: just print a small slice
            flat_preview = arr.ravel()[: min(max_items, arr.size)]
            print(f"    preview: {flat_preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect hippocampus replay NPZ files.")
    parser.add_argument(
        "replay_dir",
        nargs="?",
        default="/home/image/thebrain/replay",
        help="Directory containing NPZ replay files (default: /home/image/thebrain/replay)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=3,
        help="Max items per key to preview (default: 3)",
    )
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    if not replay_dir.is_dir():
        print(f"Replay dir {replay_dir} does not exist or is not a directory.")
        return

    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {replay_dir}")
        return

    print(f"Found {len(files)} NPZ file(s) in {replay_dir}")
    for path in files:
        inspect_npz(path, max_items=args.max_items)


if __name__ == "__main__":
    main()
