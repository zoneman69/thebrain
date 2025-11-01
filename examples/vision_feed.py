import os, time, json
import cv2
import torch, torchvision

from hippocampus import Hippocampus, Event
from hippocampus.telemetry import log_event

# Where to save the current frame for the frontend to read
FRAME_PATH = os.environ.get("HIPPO_FRAME", "/tmp/hippo_latest.jpg")

CTRL_PATH = os.environ.get("HIPPO_CTRL", "/tmp/hippo_cmd.json")
BOOK_DIR = os.environ.get("HIPPO_BOOK", "/tmp/hippo_bookmarks")
os.makedirs(BOOK_DIR, exist_ok=True)

# --- Hippocampus config (windowing + temporal prior help a lot) ---
hip = Hippocampus(
    input_dims={"vision": 576},   # MobileNetV3-Small penultimate feature size
    shared_dim=192,
    time_dim=64,
    capacity=1024,
    sparsity=0.04,
    novelty_threshold=0.25,       # try 0.2–0.35
    window_size=0.50,             # fuse frames within 500 ms as one episode
    tau=0.1,
    read_topk=1,
    temporal_beta=2.0,
    temporal_sigma=0.6,
)

def read_command():
    if not os.path.exists(CTRL_PATH):
        return None
    try:
        with open(CTRL_PATH, "r") as f:
            cmd = json.load(f)
        # clear after reading
        os.remove(CTRL_PATH)
        return cmd
    except Exception:
        return None


# --- MobileNetV3-Small feature extractor ---
# NOTE: if the weights download fails (no internet), change weights=None
try:
    m = torchvision.models.mobilenet_v3_small(weights="DEFAULT").eval()
except Exception:
    m = torchvision.models.mobilenet_v3_small(weights=None).eval()
backbone = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten())

pre = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

# we’re not training in this loop
torch.set_grad_enabled(False)

# --- Camera ---
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "No camera found"

# --- EMA predictor state ---
ema_feat = None
ema_alpha = 0.90              # higher = slower change (0.85–0.95 is reasonable)
encode_cooldown = 1.0         # seconds between encodes when motion is tiny
last_encode_t = 0.0
last_frame_save = 0.0

try:
    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        

        # (A) feature extraction
        x = pre(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            feat = backbone(x).squeeze(0).float().clone()   # [576]

        t = time.time()

        # (B) build prediction from EMA of prior features
        if ema_feat is None:
            ema_feat = feat.clone()
        else:
            ema_feat = ema_alpha * ema_feat + (1.0 - ema_alpha) * feat
        pred = ema_feat.clone()

        # (C) motion gate (avoid spam encodes on static scenes)
        l2 = torch.norm((feat - pred), p=2).item()
        elapsed = t - last_encode_t
        force_retrieve = (l2 < 0.5 and elapsed < encode_cooldown)

        # (D) seed memory: first write is forced encode (no prediction)
        if hip.mem.K.shape[0] == 0:
            mode = "encode"
            pred_for_novelty = None
        else:
            mode = None if not force_retrieve else "retrieve"
            pred_for_novelty = pred

        res = hip(Event("vision", feat, t=t, prediction=pred_for_novelty), mode=mode)
        # ---- handle control commands from the dashboard ----
        cmd = read_command()
        if cmd:
            if cmd.get("type") == "retrieve_now":
                # do a retrieval anchored to now; reuse last feat as cue
                cue_t = t
                cue = feat
                res_now = hip(Event("vision", cue, t=cue_t, prediction=pred_for_novelty), mode="retrieve")
                # Extract attention info if present
                meta = (res_now.get("metadata") or [None])[0] or {}
                attn_pairs = meta.get("attn_topk") or []
                attn_idx = [int(i) for i, _ in attn_pairs]
                attn_s = [float(s) for _, s in attn_pairs]
                ev_now = {
                    "event_modality": "vision",
                    "t": float(cue_t),
                    "mode": res_now["mode"],
                    "novelty": float(res_now["novelty"]),
                    "memory_size": int(hip.mem.K.shape[0]),
                    "pending_windows": int(res_now.get("pending_windows", 0)),
                    "attn_indices": attn_idx or None,
                    "attn_scores": attn_s or None,
                    "action": "retrieve_now"
                }
                print(json.dumps(ev_now)); log_event(ev_now)
        
            elif cmd.get("type") == "bookmark":
                # save frame + a small json note
                ts = int(t)
                fname = os.path.join(BOOK_DIR, f"book_{ts}.jpg")
                jname = os.path.join(BOOK_DIR, f"book_{ts}.json")
                try:
                    cv2.imwrite(fname, frame)
                except Exception as e:
                    print("Bookmark save failed:", e)
                note = {
                    "t": float(t),
                    "mem_size": int(hip.mem.K.shape[0]),
                    "mode": res["mode"],
                    "novelty": float(res["novelty"]),
                    "path": fname
                }
                with open(jname, "w") as f:
                    json.dump(note, f)
                # also log a telemetry line so it shows up in the table
                note["action"] = "bookmark"
                print(json.dumps(note)); log_event(note)
        
            elif cmd.get("type") == "set_cooldown":
                val = float(cmd.get("seconds", encode_cooldown))
                encode_cooldown = max(0.0, min(3.0, val))  # clamp 0–3s
                print(json.dumps({"action":"set_cooldown","value":encode_cooldown}))


        if res["mode"] == "encode":
            last_encode_t = t

        # (E) save a frame for the frontend (≈2 Hz)
        if t - last_frame_save > 0.5:
            try:
                cv2.imwrite(FRAME_PATH, frame)
            except Exception as e:
                print("Could not save frame:", e)
            last_frame_save = t

        # (F) telemetry for dashboard
        ev = {
            "event_modality": "vision",
            "t": float(t),
            "mode": res["mode"],
            "novelty": float(res["novelty"]),
            "memory_size": int(hip.mem.K.shape[0]),
            "pending_windows": int(res.get("pending_windows", 0)),
            "l2_to_pred": float(l2),
        }
        print(json.dumps(ev))
        log_event(ev)

        # (G) light console status ~1 Hz
        if t - last_print > 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] mode={res['mode']} "
                  f"novelty={res['novelty']:.3f} mem={hip.mem.K.shape[0]} "
                  f"l2_to_pred={l2:.3f}")
            last_print = t

        # ~5 Hz
        time.sleep(0.2)

finally:
    cap.release()
