from pathlib import Path
import json

from agents.pi_daemon.episodes import load_replay_file  # or similar if you have a loader

replay_dir = Path("/home/image/thebrain/replay")
latest = sorted(replay_dir.glob("*"))[-1]
print("Inspecting", latest)

records = list(load_replay_file(latest))  # adjust to your actual API
print("Num records:", len(records))
print("First record keys:", records[0].keys())
print("Inputs keys:", records[0]["inputs"].keys())
print("Vision dim:", len(records[0]["inputs"]["vision"]))
print("Auditory dim:", len(records[0]["inputs"]["auditory"]))
print("Metadata:", records[0]["metadata"])
