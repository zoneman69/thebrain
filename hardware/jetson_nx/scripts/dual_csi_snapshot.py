import cv2
import time

def pipe(sensor_id: int, w=1280, h=720, fps=60) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
    )

def snap(sensor_id: int):
    cap = cv2.VideoCapture(pipe(sensor_id), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open CSI cam sensor-id={sensor_id}")
    time.sleep(0.8)
    ok, frame = cap.read()
    cap.release()
    return ok, frame

if __name__ == "__main__":
    ts = int(time.time())
    for sid in (0, 1):
        ok, frame = snap(sid)
        if ok:
            fn = f"csi{sid}_{ts}.jpg"
            cv2.imwrite(fn, frame)
            print("Saved", fn)
        else:
            print("Failed to read cam", sid)
