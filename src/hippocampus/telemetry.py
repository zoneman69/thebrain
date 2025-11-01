# src/hippocampus/telemetry.py
import json, time, os, threading
from pathlib import Path
from typing import Dict, Any

_DEFAULT_LOG = os.environ.get("HIPPO_LOG", "/tmp/hippo.jsonl")
_lock = threading.Lock()

def log_event(data: Dict[str, Any], path: str = _DEFAULT_LOG) -> None:
    # Add timestamp if missing
    if "ts" not in data:
        data["ts"] = time.time()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
