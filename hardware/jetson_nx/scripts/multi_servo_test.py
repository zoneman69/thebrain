import time
from smbus2 import SMBus

BUS = 8
ADDR = 0x40

MODE1      = 0x00
MODE2      = 0x01
PRESCALE   = 0xFE
LED0_ON_L  = 0x06

def r(bus, reg): return bus.read_byte_data(ADDR, reg)
def w(bus, reg, val): bus.write_byte_data(ADDR, reg, val & 0xFF)

def set_pwm(bus, ch, on, off):
    reg = LED0_ON_L + 4*ch
    w(bus, reg+0, on & 0xFF)
    w(bus, reg+1, (on >> 8) & 0xFF)
    w(bus, reg+2, off & 0xFF)
    w(bus, reg+3, (off >> 8) & 0x0F)

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
    w(bus, MODE2, 0x04)                   # OUTDRV

def servo_us_to_ticks(us, hz=50):
    period = 1_000_000 / hz
    return max(0, min(4095, int(us / period * 4096)))

if __name__ == "__main__":
    with SMBus(BUS) as bus:
        set_freq(bus, 50)

        chans = [0, 1, 2, 3]
        poses = [("L", 1000), ("C", 1500), ("R", 2000), ("C", 1500)]

        print("Multi-servo test (0-3). Ctrl+C to stop.")
        try:
            while True:
                for label, us in poses:
                    ticks = servo_us_to_ticks(us)
                    print(label, us)
                    for ch in chans:
                        set_pwm(bus, ch, 0, ticks)
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
