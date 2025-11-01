import time, cv2, torch, torchvision
import torch.nn.functional as F
from hippocampus import Hippocampus, Event

# 1) Build hippocampus (vision-only for this demo)
hip = Hippocampus(
    input_dims={"vision": 576},  # MobileNetV3-Small feature dim
    shared_dim=192,
    time_dim=64,
    capacity=512,
    sparsity=0.04,
    novelty_threshold=0.2,
    window_size=0.35,
    tau=0.1,
    read_topk=1,
    temporal_beta=2.0,
    temporal_sigma=0.5,
)

# 2) MobileNetV3-Small feature extractor
m = torchvision.models.mobilenet_v3_small(weights="DEFAULT").eval()
# penultimate features: [1, 576]
backbone = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten())

pre = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

# Optional: disable grads globally for this runtime
torch.set_grad_enabled(False)

# 3) Camera loop
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "No camera found"

try:
    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # (A) feature extraction WITHOUT inference_mode (use no_grad)
        x = pre(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            feat = backbone(x).squeeze(0).float()  # [576]

        # IMPORTANT: clone to ensure not an inference tensor
        feat = feat.clone()

        # (B) send event
        t = time.time()
        res = hip(Event("vision", feat, t=t), mode=None)  # novelty gate auto-selects

        # (C) light console status ~1 Hz
        if t - last_print > 1.0:
            mode = res["mode"]
            nov = float(res["novelty"])
            mem = int(hip.mem.K.shape[0])
            print(f"[{time.strftime('%H:%M:%S')}] mode={mode} novelty={nov:.3f} mem={mem}")
            last_print = t

        # throttle ~5 Hz
        time.sleep(0.2)

finally:
    cap.release()
