import time
import board
import busio
from adafruit_pca9685 import PCA9685

# I2C bus
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)

# Servo frequency
pca.frequency = 50

CHANNEL = 0  # servo plugged into channel 0

def set_servo_pulse_us(channel: int, pulse_us: int):
    # PCA9685 is 12-bit (0-4095) over one period
    period_us = 1_000_000 / pca.frequency  # 20,000us at 50Hz
    ticks = int((pulse_us / period_us) * 4096)
    ticks = max(0, min(4095, ticks))
    pca.channels[channel].duty_cycle = int(ticks / 4095 * 65535)

try:
    print("PCA9685 servo test. Ctrl+C to stop.")
    while True:
        for name, us in [("left", 500), ("center", 1500), ("right", 2500), ("center", 1500)]:
            print(f"{name}: {us}us")
            set_servo_pulse_us(CHANNEL, us)
            time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    pca.deinit()
    print("Done.")
