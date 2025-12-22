import time
import cv2
from smbus2 import SMBus

# ---------- PCA9685 (servos) ----------
I2C_BUS = 8          # IMPORTANT: your working PCA9685 bus
PCA_ADDR = 0x40

MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06

def r(bus, reg): return bus.read_byte_data(PCA_ADDR, reg)
def w(bus, reg, val): bus.write_byte_data(PCA_ADDR, reg, val & 0xFF)

def set_pwm(bus, ch, on, off):
    reg = LED0_ON_L + 4 * ch
    w(bus, reg + 0, on & 0xFF)
    w(bus, reg + 1, (on >> 8) & 0xFF)
    w(bus, reg + 2, off & 0xFF)
    w(bus, reg + 3, (off >> 8) & 0x0F)

def set_freq(bus, hz=50):
    osc = 25_000_000.0
    prescale = int((osc / (4096.0 * hz)) - 1.0 + 0.5)
    old = r(bus, MODE1)
    w(bus, MODE1, (old & 0x7F) | 0x10)     # sleep
    w(bus, PRESCALE, prescale)
    w(bus, MODE1, old & 0xEF)             # wake
    time.sleep(0.005)
    w(bus, MODE1, (old & 0xEF) | 0x80)    # restart
    time.sleep(0.005)
    w(bus, MODE2, 0x04)                   # OUTDRV (totem pole)

def us_to_ticks(us, hz=50):
    period = 1_000_000 / hz
    return max(0, min(4095, int(us / period * 4096)))

def set_servo_us(bus, ch, us):
    set_pwm(bus, ch, 0, us_to_ticks(us))

# ---------- CSI cameras ----------
def pipe(sensor_id: int, w=1280, h=720, fps=60) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
    )

def open_cam(sensor_id: int):
    cap = cv2.VideoCapture(pipe(sensor_id), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open CSI cam sensor-id={sensor_id}")
    return cap

if __name__ == "__main__":
    # SG90-safe pulses (you can widen later)
    poses = [("L", 1000), ("C", 1500), ("R", 2000), ("C", 1500)]
    servo_channels = [0, 1, 2, 3]

    cap0 = open_cam(0)
    cap1 = open_cam(1)
    time.sleep(1.0)  # camera warm-up

    with SMBus(I2C_BUS) as bus:
        set_freq(bus, 50)

        ts = int(time.time())
        for label, us in poses:
            print(f"Pose {label}: {us}us")
            for ch in servo_channels:
                set_servo_us(bus, ch, us)

            time.sleep(0.6)

            ok0, f0 = cap0.read()
            ok1, f1 = cap1.read()

            if ok0:
                fn0 = f"pose_{label}_cam0_{ts}.jpg"
                cv2.imwrite(fn0, f0)
                print("Saved", fn0)
            else:
                print("Cam0 frame failed")

            if ok1:
                fn1 = f"pose_{label}_cam1_{ts}.jpg"
                cv2.imwrite(fn1, f1)
                print("Saved", fn1)
            else:
                print("Cam1 frame failed")

    cap0.release()
    cap1.release()
    print("Done.")
