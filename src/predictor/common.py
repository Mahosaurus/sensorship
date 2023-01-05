import pandas as pd
from src.utils.io_interaction import read_as_pandas_from_disk

def load_data():
    return read_as_pandas_from_disk("training-data.txt")

def parse_data(data):
    # Convert temp and humid to numeric
    data["temperature"] = data["temperature"].astype(float)
    data["humidity"] = data["humidity"].astype(float)
    # Last row is empty
    data = data[:-1]
    # Convert to datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"])    
    return data