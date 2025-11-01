import time, cv2, torch, torchvision
import torch.nn.functional as F
from hippocampus import Hippocampus, Event

# 1) Build hippocampus
hip = Hippocampus(
    input_dims={"vision": 576},  # MobileNetV3-Small penultimate layer size
    shared_dim=192, time_dim=64, capacity=512, sparsity=0.04,
    novelty_threshold=0.2, window_size=0.35, tau=0.1, read_topk=1,
    temporal_beta=2.0, temporal_sigma=0.5
)

# 2) MobileNetV3-Small feature extractor
m = torchvision.models.mobilenet_v3_small(weights="DEFAULT").eval()
# Remove final classifier to get features
backbone = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten())  # -> [1, 576]
pre = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
])

# 3) Camera loop
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "No camera found"
try:
    last_write_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # (A) feature
        x = pre(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.inference_mode():
            feat = backbone(x).squeeze(0).float()  # [576]

        # (B) event
        t = time.time()
        hip(Event("vision", feat, t=t), mode=None)  # auto gate encode/retrieve

        # (C) simple visualization
        if t - last_write_t > 1.0:
            print(f"mem={hip.mem.K.shape[0]}  mode=?")
            last_write_t = t

        # throttle to ~5 Hz
        time.sleep(0.2)

finally:
    cap.release()
