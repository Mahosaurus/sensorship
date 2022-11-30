import datetime

from typing import Tuple, List

def format_timestamp(timestamp: str) -> str:
    """ Convert ts to human readable """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def compile_data_point(timestamp: str, temperature: str, humidity: str) -> str:
    """ Compiles the information for one data point for text file"""
    formatted_timestamp = format_timestamp(timestamp)
    out_str = f"{formatted_timestamp}, TO_REMOVE, {temperature}, {humidity}\n"
    return out_str

def parse_data_points(data: str) -> Tuple[List[str], List[float], List[float]]:
    data = data.split("\n")[:-1]
    timestamp = [val.split(",")[0] for val in data] # Extract values
    temperature = [float(val.split(",")[2]) for val in data] # Extract values
    rel_humidity = [float(val.split(",")[3]) for val in data] # Extract values
    return timestamp, temperature, rel_humidity
