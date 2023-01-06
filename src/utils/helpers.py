import datetime
import os

from typing import Tuple, List

def format_timestamp(timestamp: str) -> str:
    """ Convert ts to human readable """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def compile_data_point(timestamp: int, temperature: str, humidity: str) -> str:
    """ Compiles the information for one data point for text file"""
    assert isinstance(temperature, str) and isinstance(humidity, str)
    formatted_timestamp = format_timestamp(timestamp)
    out_str = f"{formatted_timestamp}, {temperature}, {humidity}\n"
    return out_str

def parse_data_points(data: str) -> Tuple[List[str], List[float], List[float]]:
    data = data.split("\n")[:-1]
    timestamp = [val.split(",")[0] for val in data] # Extract values
    temperature = [float(val.split(",")[1]) for val in data] # Extract values
    rel_humidity = [float(val.split(",")[2]) for val in data] # Extract values
    return timestamp, temperature, rel_humidity

def get_repo_root() -> str:
    """
    Returns the absolute path to the root of the  repository
    specific to the system the code is run on.
    """
    path_to_this_file = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/"
    return path_to_this_file
