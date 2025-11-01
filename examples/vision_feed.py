import time, cv2, torch, torchvision
from hippocampus import Hippocampus, Event

import os, json
from hippocampus.telemetry import log_event

FRAME_PATH = os.environ.get("HIPPO_FRAME", "/tmp/hippo_latest.jpg")

# --- Hippocampus config (windowing + temporal prior help a lot) ---
hip = Hippocampus(
    input_dims={"vision": 576},
    shared_dim=192,
    time_dim=64,
    capacity=1024,         # larger so you can watch behavior
    sparsity=0.04,
    novelty_threshold=0.25, # tweak 0.2–0.35; higher = stricter "new"
    window_size=0.50,       # fuse frames within 500 ms as one episode
    tau=0.1,
    read_topk=1,
    temporal_beta=2.0,
    temporal_sigma=0.6,
)

# --- MobileNetV3-Small feature extractor ---
m = torchvision.models.mobilenet_v3_small(weights="DEFAULT").eval()
backbone = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten())

pre = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

torch.set_grad_enabled(False)

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "No camera found"

# --- EMA predictor state ---
ema_feat = None
ema_alpha = 0.9          # higher = slower change; try 0.85–0.95
encode_cooldown = 1.0    # seconds after last encode before allowing next encode
last_encode_t = 0.0

try:
    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        x = pre(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            feat = backbone(x).squeeze(0).float()

        t = time.time()

        # --- Build a prediction from EMA of prior features ---
        if ema_feat is None:
            ema_feat = feat.clone()
        else:
            ema_feat = ema_alpha * ema_feat + (1.0 - ema_alpha) * feat

        pred = ema_feat.clone()  # prediction for novelty comparator

        # Optional: simple motion gate to avoid spam-encoding on static scenes
        # If the L2 distance to pred is tiny and cooldown not elapsed, force retrieve
        l2 = torch.norm((feat - pred), p=2).item()
        elapsed = t - last_encode_t
        force_retrieve = (l2 < 0.5 and elapsed < encode_cooldown)

        mode = None if not force_retrieve else "retrieve"

        res = hip(Event("vision", feat.clone(), t=t, prediction=pred), mode=mode)

        # Track when we actually encoded
        if res["mode"] == "encode":
            last_encode_t = t

        if t - last_print > 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] mode={res['mode']} "
                  f"novelty={res['novelty']:.3f} mem={hip.mem.K.shape[0]} "
                  f"l2_to_pred={l2:.3f}")
            last_print = t

        # ~5 Hz
        time.sleep(0.2)

finally:
    cap.release()
