import cv2
import time

# Adjust sensor-id if you have multiple CSI cams
SENSOR_ID = 0

# 1280x720@30 is a safe starting point on many CSI modules
pipeline = (
    f"nvarguscamerasrc sensor-id={SENSOR_ID} ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Could not open CSI camera (check nvargus + camera wiring)")

print("Press 's' to save frame, 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed")
        time.sleep(0.1)
        continue

    cv2.imshow("CSI Camera", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        fn = f"csi_capture_{int(time.time())}.jpg"
        cv2.imwrite(fn, frame)
        print("Saved", fn)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
