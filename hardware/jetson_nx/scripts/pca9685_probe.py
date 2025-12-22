from smbus2 import SMBus

BUS=7
ADDR=0x40

MODE1=0x00
MODE2=0x01
SUBADR1=0x02
SUBADR2=0x03
SUBADR3=0x04
ALLCALLADR=0x05
PRESCALE=0xFE

def r(bus, reg): return bus.read_byte_data(ADDR, reg)

with SMBus(BUS) as bus:
    regs = {
        "MODE1": r(bus, MODE1),
        "MODE2": r(bus, MODE2),
        "SUBADR1": r(bus, SUBADR1),
        "SUBADR2": r(bus, SUBADR2),
        "SUBADR3": r(bus, SUBADR3),
        "ALLCALLADR": r(bus, ALLCALLADR),
        "PRESCALE": r(bus, PRESCALE),
    }
    for k,v in regs.items():
        print(f"{k}: 0x{v:02X}")
