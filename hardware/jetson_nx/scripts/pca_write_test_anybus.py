from smbus2 import SMBus
import time

BUS = 7      # <-- change this
ADDR = 0x40
MODE1 = 0x00

with SMBus(BUS) as bus:
    before = bus.read_byte_data(ADDR, MODE1)
    bus.write_byte_data(ADDR, MODE1, before ^ 0x01)  # flip bit0
    time.sleep(0.01)
    after = bus.read_byte_data(ADDR, MODE1)
    print("before", hex(before), "after", hex(after))
