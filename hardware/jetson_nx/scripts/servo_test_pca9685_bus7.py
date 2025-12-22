import time
from smbus2 import SMBus

# --------------------
# I2C configuration
# --------------------
BUS = 8
ADDR = 0x40

MODE1      = 0x00
MODE2      = 0x01
PRESCALE   = 0xFE
LED0_ON_L  = 0x06

# --------------------
# Low-level helpers
# --------------------
def write8(bus, reg, val):
    bus.write_byte_data(ADDR, reg, val & 0xFF)

def read8(bus, reg):
    return bus.read_byte_data(ADDR, reg)

def set_pwm(bus, channel, on, off):
    reg = LED0_ON_L + 4 * channel
    bus.write_byte_data(ADDR, reg + 0, on & 0xFF)
    bus.write_byte_data(ADDR, reg + 1, (on >> 8) & 0xFF)
    bus.write_byte_data(ADDR, reg + 2, off & 0xFF)
    bus.write_byte_data(ADDR, reg + 3, (off >> 8) & 0xFF)

# --------------------
# PCA9685 setup
# --------------------
def set_pwm_freq(bus, freq_hz=50):
    osc = 25_000_000.0
    prescaleval = (osc / (4096.0 * freq_hz)) - 1.0
    prescale = int(prescaleval + 0.5)

    oldmode = read8(bus, MODE1)

    # Sleep
    write8(bus, MODE1, (oldmode & 0x7F) | 0x10)
    write8(bus, PRESCALE, prescale)

    # Wake
    write8(bus, MODE1, oldmode & 0xEF)
    time.sleep(0.005)

    # Restart
    write8(bus, MODE1, (oldmode & 0xEF) | 0x80)
    time.sleep(0.005)

def set_servo_us(bus, channel, pulse_us, freq_hz=50):
    period_us = 1_000_000.0 / freq_hz  # 20,000us at 50Hz
    ticks = int((pulse_us / period_us) * 4096.0)
    ticks = max(0, min(4095, ticks))
    set_pwm(bus, channel, 0, ticks)

# --------------------
# Main test
# --------------------
if __name__ == "__main__":
    with SMBus(BUS) as bus:
        # Force wake in case chip boots asleep
        mode1 = read8(bus, MODE1)
        write8(bus, MODE1, mode1 & 0xEF)
        time.sleep(0.01)

        set_pwm_freq(bus, 50)

        CHANNEL = 0  # servo on channel 0

        positions = [
            ("left",   1000),
            ("center", 1500),
            ("right",  2000),
            ("center", 1500),
        ]

        print("Servo test running (Ctrl+C to stop)")
        try:
            while True:
                for name, us in positions:
                    print(f"{name}: {us}us")
                    set_servo_us(bus, CHANNEL, us)
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping servo test")
