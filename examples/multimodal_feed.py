"""
Multimodal live feed (vision + audio + language) into a single Hippocampus instance.

- Vision: MobileNetV3-Small 576-D features
- Audio: 256-D embedding from log-spectrogram (NumPy FFT + random projection)
- Language: tiny hash->Embedding bag-of-words encoder to 256-D
- EMA-based novelty + gating per modality
- Telemetry to HIPPO_LOG and live frame/spectrogram to HIPPO_FRAME / HIPPO_SPEC
- Control commands via HIPPO_CTRL:
    {"type":"retrieve_now"}
    {"type":"bookmark","label":"..."}
    {"type":"set_cooldown","seconds":1.5}
    {"type":"language","text":"..."}
    {"type":"save_snapshot","path":"/tmp/snap.pt"}
    {"type":"load_snapshot","path":"/tmp/snap.pt"}
"""

import os, time, json, threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import cv2
import torch, torchvision
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hippocampus import Event, Hippocampus, HippocampusLoader
from hippocampus.telemetry import log_event

# ---------- Paths / env ----------
LOG_PATH   = os.environ.get("HIPPO_LOG",  "/tmp/hippo.jsonl")
FRAME_PATH = os.environ.get("HIPPO_FRAME","/tmp/hippo_latest.jpg")
SPEC_PATH  = os.environ.get("HIPPO_SPEC","/tmp/hippo_spec.png")
CTRL_PATH  = os.environ.get("HIPPO_CTRL", "/tmp/hippo_cmd.json")
BOOK_DIR   = os.environ.get("HIPPO_BOOK", "/tmp/hippo_bookmarks")
ARTIFACTS_DIR = Path(os.environ.get("HIPPO_ARTIFACTS_DIR", "/home/image/thebrain/artifacts"))
ARTIFACT_POLL_SECONDS = float(os.environ.get("HIPPO_ARTIFACTS_POLL", "60"))
Path(BOOK_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Hippocampus ----------
hip = Hippocampus(
    input_dims={"vision": 576, "auditory": 256, "language": 256},
    shared_dim=192,
    time_dim=64,
    capacity=1024,
    sparsity=0.04,
    novelty_threshold=0.25,
    window_size=0.50,
    tau=0.1,
    read_topk=1,
    temporal_beta=2.0,
    temporal_sigma=0.6,
)
torch.set_grad_enabled(False)
artifact_loader = HippocampusLoader(hip, ARTIFACTS_DIR)
artifact_loader.load_latest()

# ---------- Vision (MobileNetV3 Small) ----------
try:
    m = torchvision.models.mobilenet_v3_small(weights="DEFAULT").eval()
except Exception:
    m = torchvision.models.mobilenet_v3_small(weights=None).eval()
vision_backbone = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten())
vision_pre = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
cam = cv2.VideoCapture(0)
assert cam.isOpened(), "No camera found"

# ---------- Audio capture (sounddevice) ----------
SR = 16000
BLOCK = 1024               # ~64ms
RING_SECONDS = 1.0
RING_LEN = int(RING_SECONDS * SR)
ring = np.zeros(RING_LEN, dtype=np.float32)
ring_pos = 0
ring_lock = threading.Lock()
audio_rms = 0.0

FFT = 512
HOP = 256
N_BINS = 128
EMB_DIM = 256
rng = np.random.default_rng(7)
proj_audio = rng.standard_normal((N_BINS * 32, EMB_DIM)).astype(np.float32) / np.sqrt(N_BINS * 32)

def compute_audio_embedding(wave: np.ndarray):
    """Return (256-D embedding, rms, spectrogram[H,W])."""
    seg_len = min(len(wave), SR // 2)
    x = wave[-seg_len:].astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2) + 1e-9))
    frames = []
    for start in range(0, max(1, seg_len - FFT), HOP):
        win = x[start:start+FFT]
        if len(win) < FFT:
            pad = np.zeros(FFT, dtype=np.float32); pad[:len(win)] = win; win = pad
        w = np.hanning(FFT).astype(np.float32)
        spec = np.fft.rfft(win * w).astype(np.complex64)
        mag = np.abs(spec)
        frames.append(mag[:N_BINS])
        if len(frames) >= 32: break
    if not frames:
        frames = [np.zeros(N_BINS, dtype=np.float32)]
    S = np.stack(frames, axis=1)  # [N_BINS, T]
    S = np.log1p(S)
    if S.shape[1] < 32:
        S = np.concatenate([S, np.zeros((N_BINS, 32 - S.shape[1]), dtype=np.float32)], axis=1)
    elif S.shape[1] > 32:
        S = S[:, :32]
    emb = (S.reshape(-1) @ proj_audio).astype(np.float32)  # [256]
    return emb, rms, S

def audio_callback(indata, frames, timeinfo, status):
    global ring, ring_pos, audio_rms
    if status: print("Audio status:", status)
    x = indata.copy().astype(np.float32).mean(axis=1, keepdims=False)
    with ring_lock:
        n = len(x); end = ring_pos + n
        if end <= RING_LEN:
            ring[ring_pos:end] = x
        else:
            k = RING_LEN - ring_pos
            ring[ring_pos:] = x[:k]
            ring[:n-k] = x[k:]
        ring_pos = (ring_pos + n) % RING_LEN
        audio_rms = float(np.sqrt(np.mean(ring**2) + 1e-9))

stream = sd.InputStream(channels=1, samplerate=SR, blocksize=BLOCK, dtype="float32", callback=audio_callback)
stream.start()

# ---------- Language tiny encoder ----------
VOCAB = 8192
lang_embed = torch.nn.Embedding(VOCAB, 256)
with torch.no_grad():
    lang_embed.weight.normal_(mean=0.0, std=1.0 / np.sqrt(256))
lang_embed.weight.requires_grad_(False)

def text_to_indices(txt: str):
    toks = [t for t in txt.strip().split() if t]
    if not toks: return torch.empty(0, dtype=torch.long)
    idxs = [abs(hash(t)) % VOCAB for t in toks]
    return torch.tensor(idxs, dtype=torch.long)

def encode_text(txt: str) -> torch.Tensor:
    idx = text_to_indices(txt)
    if idx.numel() == 0:
        return torch.zeros(256)
    with torch.no_grad():
        emb = lang_embed(idx)          # [n,256]
        return emb.mean(dim=0).float() # [256]

# ---------- EMA state & helpers ----------
ema_vision = None
ema_audio = None
ema_lang  = None
EMA_ALPHA_V = 0.90
EMA_ALPHA_A = 0.90
EMA_ALPHA_L = 0.90

ENC_COOLDOWN = 1.0
last_encode_t = 0.0
last_frame_save = 0.0
last_spec_save  = 0.0

# last seen features (targets for recon cosines during retrieve_now)
last_vfeat = None
last_aemb  = None
last_lfeat = None

def now_sec() -> float: return time.time()

def read_command():
    p = Path(CTRL_PATH)
    if not p.exists(): return None
    try:
        cmd = json.loads(p.read_text())
    except Exception:
        try: p.unlink()
        except Exception: pass
        return None
    try: p.unlink()
    except Exception: pass
    return cmd

# thumbnail projection for bookmarks (576 -> 128)
proj_thumb = rng.standard_normal((576, 128)).astype(np.float32) / np.sqrt(576)

try:
    last_print = 0.0
    last_artifact_poll = 0.0
    while True:
        t = now_sec()

        if t - last_artifact_poll >= ARTIFACT_POLL_SECONDS:
            if artifact_loader.load_latest():
                print(json.dumps({"action": "reload_artifacts", "artifact_dir": str(ARTIFACTS_DIR)}))
            last_artifact_poll = t

        # ===== Vision =====
        ok, frame = cam.read()
        if not ok: break
        x = vision_pre(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            vfeat = vision_backbone(x).squeeze(0).float().clone()  # [576]
        last_vfeat = vfeat

        if ema_vision is None:
            ema_vision = vfeat.clone()
        else:
            ema_vision = EMA_ALPHA_V * ema_vision + (1.0 - EMA_ALPHA_V) * vfeat
        vpred = ema_vision.clone()
        v_l2 = float(torch.norm(vfeat - vpred, p=2))

        # ===== Audio =====
        with ring_lock:
            wave = ring.copy()
        a_emb, a_rms, S_log = compute_audio_embedding(wave)
        a_emb_t = torch.from_numpy(a_emb)  # [256]
        last_aemb = a_emb_t

        if ema_audio is None:
            ema_audio = a_emb_t.clone()
        else:
            ema_audio = EMA_ALPHA_A * ema_audio + (1.0 - EMA_ALPHA_A) * a_emb_t
        apred = ema_audio.clone()
        a_l2 = float(torch.norm(a_emb_t - apred, p=2))

        # ===== Policy: if both low motion and cooldown -> retrieve; else None (auto gate) =====
        elapsed = t - last_encode_t
        force_retrieve = ((v_l2 < 0.5) and (a_l2 < 0.5) and (elapsed < ENC_COOLDOWN))
        mode = None if not force_retrieve else "retrieve"

        # Seed memory on very first write
        if hip.mem.K.shape[0] == 0:
            r_v = hip(Event("vision", vfeat, t=t, prediction=None),   mode="encode")
            r_a = hip(Event("auditory", a_emb_t, t=t+0.05, prediction=None), mode="encode")
        else:
            r_v = hip(Event("vision", vfeat, t=t, prediction=vpred),  mode=mode)
            r_a = hip(Event("auditory", a_emb_t, t=t+0.05, prediction=apred), mode=mode)

        # Save frame atomically (~2 Hz)
        if t - last_frame_save > 0.5:
            try:
                ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_enc:
                    tmp = FRAME_PATH + ".tmp"  # arbitrary suffix; we write raw bytes
                    with open(tmp, "wb") as f:
                        f.write(buf.tobytes())
                    os.replace(tmp, FRAME_PATH)  # atomic rename
                else:
                    print("Could not encode frame to JPEG")
            except Exception as e:
                print("Could not save frame:", e)
            last_frame_save = t
        
        # Save spectrogram atomically (~1 Hz)
        if t - last_spec_save > 1.0:
            try:
                plt.figure(figsize=(4,2), dpi=100)
                plt.imshow(S_log, aspect="auto", origin="lower")
                plt.title("Log-spectrogram")
                plt.tight_layout()
                tmp = SPEC_PATH + ".tmp"     # temp path with arbitrary suffix
                plt.savefig(tmp, format="png")  # tell matplotlib it's a PNG
                plt.close()
                os.replace(tmp, SPEC_PATH)   # atomic rename
            except Exception as e:
                print("Could not save spectrogram:", e)
            last_spec_save = t


        # Track last encode
        if r_v["mode"] == "encode" or r_a["mode"] == "encode":
            last_encode_t = t

        # Telemetry lines (vision + audio)
        ev_v = {
            "event_modality": "vision", "t": float(t),
            "mode": r_v["mode"], "novelty": float(r_v["novelty"]),
            "memory_size": int(hip.mem.K.shape[0]),
            "pending_windows": int(r_v.get("pending_windows", 0)),
            "l2_to_pred": float(v_l2)
        }
        print(json.dumps(ev_v)); log_event(ev_v)

        ev_a = {
            "event_modality": "auditory", "t": float(t),
            "mode": r_a["mode"], "novelty": float(r_a["novelty"]),
            "memory_size": int(hip.mem.K.shape[0]),
            "pending_windows": int(r_a.get("pending_windows", 0)),
            "l2_to_pred_audio": float(a_l2),
            "audio_rms": float(a_rms),
            "audio_db": float(20.0 * np.log10(a_rms + 1e-8)),
        }
        print(json.dumps(ev_a)); log_event(ev_a)

        # ---- Handle control commands ----
        cmd = read_command()
        if cmd:
            ctype = cmd.get("type")
            if ctype == "set_cooldown":
                val = float(cmd.get("seconds", ENC_COOLDOWN))
                ENC_COOLDOWN = max(0.0, min(3.0, val))
                print(json.dumps({"action":"set_cooldown","value":ENC_COOLDOWN}))
            elif ctype == "language":
                text = (cmd.get("text") or "").strip()
                if text:
                    lfeat = encode_text(text)
                    last_lfeat = lfeat
                    # keep a simple EMA prediction for language too
                    if ema_lang is None: ema_lang = lfeat.clone()
                    else: ema_lang = EMA_ALPHA_L * ema_lang + (1.0 - EMA_ALPHA_L) * lfeat
                    r_l = hip(Event("language", lfeat, t=t+0.1, prediction=ema_lang), mode=None)
                    ev_l = {
                        "event_modality": "language", "t": float(t),
                        "mode": r_l["mode"], "novelty": float(r_l["novelty"]),
                        "memory_size": int(hip.mem.K.shape[0]),
                        "pending_windows": int(r_l.get("pending_windows", 0)),
                    }
                    print(json.dumps(ev_l)); log_event(ev_l)
            elif ctype == "retrieve_with_text":
                text = (cmd.get("text") or "").strip()
                if text:
                    lcue = encode_text(text)
                    res = hip(Event("language", lcue, t=t+0.2, prediction=ema_lang), mode="retrieve")
                    fused = res["output"]
                    recon = hip.decode(fused, ["vision", "auditory", "language"])

                    tgt_v = last_vfeat if last_vfeat is not None else torch.zeros_like(recon["vision"])
                    tgt_a = last_aemb if last_aemb is not None else torch.zeros_like(recon["auditory"])
                    tgt_l = lcue

                    cos_v = float(F.cosine_similarity(recon["vision"], tgt_v, dim=0))
                    cos_a = float(F.cosine_similarity(recon["auditory"], tgt_a, dim=0))
                    cos_l = float(F.cosine_similarity(recon["language"], tgt_l, dim=0))

                    meta = (res.get("metadata") or [None])[0] or {}
                    attn_pairs = meta.get("attn_topk") or []
                    attn_idx = [int(i) for i, _ in attn_pairs]
                    attn_s = [float(s) for _, s in attn_pairs]

                    ev_r = {
                        "event_modality": "language",
                        "t": float(t),
                        "mode": res["mode"],
                        "novelty": float(res["novelty"]),
                        "memory_size": int(hip.mem.K.shape[0]),
                        "pending_windows": int(res.get("pending_windows", 0)),
                        "attn_indices": attn_idx or None,
                        "attn_scores": attn_s or None,
                        "selected_window_id": meta.get("window_id"),
                        "selected_t_window": meta.get("t_window"),
                        "action": "retrieve_with_text",
                        "recon/vision_cos": cos_v,
                        "recon/auditory_cos": cos_a,
                        "recon/language_cos": cos_l,
                    }
                    print(json.dumps(ev_r)); log_event(ev_r)
            elif ctype == "retrieve_now":
                # cue = last seen vision; retrieve and decode into all modalities
                cue = last_vfeat if last_vfeat is not None else vfeat
                res = hip(Event("vision", cue, t=t+0.2, prediction=ema_vision), mode="retrieve")
                fused = res["output"]
                recon = hip.decode(fused, ["vision","auditory","language"])

                # build targets: last seen features (fallback to zeros if missing)
                tgt_v = last_vfeat if last_vfeat is not None else torch.zeros_like(recon["vision"])
                tgt_a = last_aemb  if last_aemb  is not None else torch.zeros_like(recon["auditory"])
                tgt_l = last_lfeat if last_lfeat is not None else torch.zeros_like(recon["language"])

                cos_v = float(F.cosine_similarity(recon["vision"],   tgt_v, dim=0))
                cos_a = float(F.cosine_similarity(recon["auditory"], tgt_a, dim=0))
                cos_l = float(F.cosine_similarity(recon["language"], tgt_l, dim=0))

                meta = (res.get("metadata") or [None])[0] or {}
                attn_pairs = meta.get("attn_topk") or []
                attn_idx = [int(i) for i, _ in attn_pairs]
                attn_s = [float(s) for _, s in attn_pairs]

                ev_r = {
                    "event_modality": "vision",
                    "t": float(t),
                    "mode": res["mode"],
                    "novelty": float(res["novelty"]),
                    "memory_size": int(hip.mem.K.shape[0]),
                    "pending_windows": int(res.get("pending_windows", 0)),
                    "attn_indices": attn_idx or None,
                    "attn_scores": attn_s or None,
                    "selected_window_id": meta.get("window_id"),
                    "selected_t_window": meta.get("t_window"),
                    "action": "retrieve_now",
                    "recon/vision_cos": cos_v,
                    "recon/auditory_cos": cos_a,
                    "recon/language_cos": cos_l,
                }
                print(json.dumps(ev_r)); log_event(ev_r)

            elif ctype == "bookmark":
                label = (cmd.get("label") or "").strip()
                ts = int(t)
                img_path = Path(BOOK_DIR) / f"book_{ts}.jpg"
                meta_path = Path(BOOK_DIR) / f"book_{ts}.json"
                try: cv2.imwrite(str(img_path), frame)
                except Exception as e: print("Bookmark save failed:", e)

                # thumbnail embedding from current vision feature
                try:
                    thumb128 = (vfeat.detach().cpu().numpy() @ proj_thumb).astype(np.float32)  # [128]
                    thumb128 = thumb128 / (np.linalg.norm(thumb128) + 1e-9)
                    thumb128 = thumb128.tolist()
                except Exception:
                    thumb128 = None

                note = {
                    "t": float(t),
                    "mem_size": int(hip.mem.K.shape[0]),
                    "mode": r_v["mode"],
                    "novelty": float(r_v["novelty"]),
                    "path": str(img_path),
                    "label": label or None,
                    "thumb128": thumb128,
                }
                meta_path.write_text(json.dumps(note))
                note["action"] = "bookmark"
                print(json.dumps(note)); log_event(note)

            elif ctype == "save_snapshot":
                from hippocampus.memory_io import save_memory
                save_memory(hip, cmd["path"])
                print(json.dumps({"action":"save_snapshot","path":cmd["path"]}))
            elif ctype == "load_snapshot":
                from hippocampus.memory_io import load_memory
                load_memory(hip, cmd["path"])
                print(json.dumps({"action":"load_snapshot","path":cmd["path"]}))

        # status ~1 Hz
        if t - last_print > 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] mem={hip.mem.K.shape[0]} "
                  f"v_mode={r_v['mode']} v_nov={r_v['novelty']:.3f} "
                  f"a_mode={r_a['mode']} a_nov={r_a['novelty']:.3f}")
            last_print = t

        time.sleep(0.2)

finally:
    try:
        stream.stop(); stream.close()
    except Exception:
        pass
    cam.release()
