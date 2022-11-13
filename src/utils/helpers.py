import datetime

def format_timestamp(timestamp: str) -> str:
    """ Convert ts to human readable """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def compile_data_point(timestamp: str, temperature: str, humidity: str) -> str:
    """ Compiles the information for one data point for text file"""
    formatted_timestamp = format_timestamp(timestamp)
    out_str = f"{formatted_timestamp}, TO_REMOVE, {temperature}, {humidity}\n"
    return out_str
