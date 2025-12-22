import cv2
import time

def pipe(sensor_id, w=1280, h=720, fps=60):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
    )

cap0 = cv2.VideoCapture(pipe(0), cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(pipe(1), cv2.CAP_GSTREAMER)

if not cap0.isOpened():
    raise RuntimeError("Failed to open CSI cam 0")
if not cap1.isOpened():
    raise RuntimeError("Failed to open CSI cam 1")

# warm up
time.sleep(1)

ok0, f0 = cap0.read()
ok1, f1 = cap1.read()

ts = int(time.time())
if ok0:
    fn0 = f"csi0_{ts}.jpg"
    cv2.imwrite(fn0, f0)
    print("Saved", fn0)
else:
    print("Failed to read cam 0 frame")

if ok1:
    fn1 = f"csi1_{ts}.jpg"
    cv2.imwrite(fn1, f1)
    print("Saved", fn1)
else:
    print("Failed to read cam 1 frame")

cap0.release()
cap1.release()
