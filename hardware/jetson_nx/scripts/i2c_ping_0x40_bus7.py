from smbus2 import SMBus

BUS = 7
ADDR = 0x40

with SMBus(BUS) as bus:
    # PCA9685 MODE1 register is 0x00
    mode1 = bus.read_byte_data(ADDR, 0x00)
    print(f"Read MODE1 from 0x{ADDR:02X} on bus {BUS}: 0x{mode1:02X}")
