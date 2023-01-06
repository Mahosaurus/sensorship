import time

from typing import Tuple

import smbus


address = 0x38 #Put your device's address here

try:
    i2cbus = smbus.SMBus(1)
except (FileNotFoundError, PermissionError):
    print("No sensor connected / permission error.")
time.sleep(0.5)

def get_sensor_data() -> Tuple[str, str]:
    """ Get real sensor data """
    data = i2cbus.read_i2c_block_data(address,0x71,1)
    if (data[0] | 0x08) == 0:
        print('Initialization error')

    i2cbus.write_i2c_block_data(address,0xac,[0x33,0x00])
    time.sleep(0.1)

    data = i2cbus.read_i2c_block_data(address,0x71,7)

    temp_raw_value = ((data[3] & 0xf) << 16) + (data[4] << 8) + data[5]
    temperature = round(200*float(temp_raw_value)/2**20 - 50, 2)

    humid_raw_value = ((data[3] & 0xf0) >> 4) + (data[1] << 12) + (data[2] << 4)
    humidity = round(100*float(humid_raw_value)/2**20, 2)

    return str(temperature), str(humidity)
