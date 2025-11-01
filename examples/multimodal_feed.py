"""
Multimodal live feed (vision + audio) into a single Hippocampus instance.
- Vision: MobileNetV3-Small 576-D features
- Audio: 256-D embedding from log-spectrogram (numpy FFT + random projection)
- EMA-based novelty + gating per modality
- Telemetry to HIPPO_LOG and live frame to HIPPO_FRAME
- Optional spectrogram image to HIPPO_SPEC
"""

import os, time, json, threading, queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import cv2
import torch, torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hippocampus import Hippocampus, Event
from hippocampus.telemetry import log_event

# ---------- Paths / env ----------
LOG_PATH   = os.environ.get("HIPPO_LOG",  "/tmp/hippo.jsonl")
FRAME_PATH = os.environ.get("HIPPO_FRAME","/tmp/hippo_latest.jpg")
SPEC_PATH  = os.environ.get("HIPPO_SPEC","/tmp/hippo_spec.png")

# ---------- Hippocampus ----------
hip = Hippocampus(
    input_dims={"vision": 576, "auditory": 256},
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

# For embedding: window 512, hop 256, n_frames ~ 32, n_bins 128 -> flatten -> proj(256)
FFT = 512
HOP = 256
N_BINS = 128
EMB_DIM = 256
rng = np.random.default_rng(7)
proj = rng.standard_normal((N_BINS * 32, EMB_DIM)).astype(np.float32) / np.sqrt(N_BINS * 32)

# Spectrogram queue for visualization (drop frames if slow)
spec_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)

def compute_audio_embedding(wave: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (256-D embedding, rms, spectrogram[H,W])."""
    # Take last ~0.5s
    seg_len = min(len(wave), SR // 2)
    x = wave[-seg_len:].astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2) + 1e-9))
    # STFT (magnitude)
    frames = []
    for start in range(0, max(1, seg_len - FFT), HOP):
        win = x[start:start+FFT]
        if len(win) < FFT:
            pad = np.zeros(FFT, dtype=np.float32)
            pad[:len(win)] = win
            win = pad
        w = np.hanning(FFT).astype(np.float32)
        spec = np.fft.rfft(win * w).astype(np.complex64)
        mag = np.abs(spec)  # [FFT/2+1] -> 257
        frames.append(mag[:N_BINS])  # take 128 lowest bins
        if len(frames) >= 32:  # cap time frames
            break
    if not frames:
        frames = [np.zeros(N_BINS, dtype=np.float32)]
    S = np.stack(frames, axis=1)  # [N_BINS, T]
    # log compression
    S_log = np.log1p(S)
    # fixed-size [128,32]
    if S_log.shape[1] < 32:
        pad = np.zeros((N_BINS, 32 - S_log.shape[1]), dtype=np.float32)
        S_log = np.concatenate([S_log, pad], axis=1)
    elif S_log.shape[1] > 32:
        S_log = S_log[:, :32]

    # Embedding via fixed random projection
    emb = (S_log.reshape(-1) @ proj).astype(np.float32)  # [256]

    return emb, rms, S_log

def audio_callback(indata, frames, timeinfo, status):
    global ring, ring_pos, audio_rms
    if status:
        print("Audio status:", status)
    # mono
    x = indata.copy().astype(np.float32).mean(axis=1, keepdims=False)
    with ring_lock:
        n = len(x)
        end = ring_pos + n
        if end <= RING_LEN:
            ring[ring_pos:end] = x
        else:
            k = RING_LEN - ring_pos
            ring[ring_pos:] = x[:k]
            ring[:n-k] = x[k:]
        ring_pos = (ring_pos + n) % RING_LEN
        audio_rms = float(np.sqrt(np.mean(ring**2) + 1e-9))

# Start audio stream
stream = sd.InputStream(channels=1, samplerate=SR, blocksize=BLOCK, dtype="float32", callback=audio_callback)
stream.start()

# EMA prediction per modality
ema_vision = None
ema_audio = None
EMA_ALPHA_V = 0.90
EMA_ALPHA_A = 0.90
ENC_COOLDOWN = 1.0
last_encode_t = 0.0
last_frame_save = 0.0
last_spec_save = 0.0

def now_sec() -> float:
    return time.time()

try:
    last_print = 0.0
    while True:
        t = now_sec()

        # ===== Vision =====
        ok, frame = cam.read()
        if not ok:
            break
        x = vision_pre(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            vfeat = vision_backbone(x).squeeze(0).float().clone()  # [576]

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

        if ema_audio is None:
            ema_audio = a_emb_t.clone()
        else:
            ema_audio = EMA_ALPHA_A * ema_audio + (1.0 - EMA_ALPHA_A) * a_emb_t
        apred = ema_audio.clone()
        a_l2 = float(torch.norm(a_emb_t - apred, p=2))

        # Policy: if both low motion and within cooldown -> retrieve; else None (auto gate)
        elapsed = t - last_encode_t
        force_retrieve = ((v_l2 < 0.5) and (a_l2 < 0.5) and (elapsed < ENC_COOLDOWN))
        mode = None if not force_retrieve else "retrieve"

        # First ever write forces encode without prediction
        if hip.mem.K.shape[0] == 0:
            # write vision then audio in the same time window
            r_v = hip(Event("vision", vfeat, t=t, prediction=None), mode="encode")
            r_a = hip(Event("auditory", a_emb_t, t=t+0.05, prediction=None), mode="encode")
        else:
            r_v = hip(Event("vision", vfeat, t=t, prediction=vpred), mode=mode)
            r_a = hip(Event("auditory", a_emb_t, t=t+0.05, prediction=apred), mode=mode)

        # Save frame & spectrogram periodically
        if t - last_frame_save > 0.5:
            try:
                cv2.imwrite(FRAME_PATH, frame)
            except Exception as e:
                print("Could not save frame:", e)
            last_frame_save = t

        if t - last_spec_save > 1.0:
            try:
                plt.figure(figsize=(4,2), dpi=100)
                plt.imshow(S_log, aspect="auto", origin="lower")
                plt.title("Log-spectrogram")
                plt.tight_layout()
                plt.savefig(SPEC_PATH)
                plt.close()
            except Exception as e:
                print("Could not save spectrogram:", e)
            last_spec_save = t

        # Track last encode
        if r_v["mode"] == "encode" or r_a["mode"] == "encode":
            last_encode_t = t

        # Telemetry lines (one per modality tick)
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

        # Status ~ 1 Hz
        if t - last_print > 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] mem={hip.mem.K.shape[0]} "
                  f"v_mode={r_v['mode']} v_nov={r_v['novelty']:.3f} vL2={v_l2:.3f} | "
                  f"a_mode={r_a['mode']} a_nov={r_a['novelty']:.3f} aL2={a_l2:.3f} aRMS={a_rms:.3f}")
            last_print = t

        time.sleep(0.2)  # ~5 Hz outer loop

finally:
    stream.stop(); stream.close()
    cam.release()
