import time
from smbus2 import SMBus

BUS=7
ADDR=0x40

LED0_ON_L = 0x06

def r(bus, reg): return bus.read_byte_data(ADDR, reg)
def w(bus, reg, val): bus.write_byte_data(ADDR, reg, val & 0xFF)

def set_ch0_off(bus, ticks):
    # ON=0, OFF=ticks
    w(bus, LED0_ON_L+0, 0)          # ON_L
    w(bus, LED0_ON_L+1, 0)          # ON_H
    w(bus, LED0_ON_L+2, ticks & 0xFF)       # OFF_L
    w(bus, LED0_ON_L+3, (ticks >> 8) & 0x0F)# OFF_H

with SMBus(BUS) as bus:
    print("Writing CH0 OFF ticks and reading back. Ctrl+C to stop.")
    try:
        while True:
            for ticks in (205, 307, 410):  # approx 1.0ms, 1.5ms, 2.0ms at 50Hz
                set_ch0_off(bus, ticks)
                back = [r(bus, LED0_ON_L+i) for i in range(4)]
                print("ticks", ticks, "regs", [hex(x) for x in back])
                time.sleep(1)
    except KeyboardInterrupt:
        pass
