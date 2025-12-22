from smbus2 import SMBus
import time

BUS=7
ADDR=0x40
MODE1=0x00

with SMBus(BUS) as bus:
    before = bus.read_byte_data(ADDR, MODE1)
    print(f"MODE1 before: 0x{before:02X}")

    # Try a simple write: clear sleep
    bus.write_byte_data(ADDR, MODE1, before & 0xEF)
    time.sleep(0.01)

    after = bus.read_byte_data(ADDR, MODE1)
    print(f"MODE1 after : 0x{after:02X}")
