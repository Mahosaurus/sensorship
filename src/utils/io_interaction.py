import os
import pandas as pd

from src.config import DATA_COLUMNS, DATA_SEP

# Custom Format (no header) str vs. Pandas DF
# 2022-11-21 19:00:00, 18.75, 56.84

def read_as_str_from_disk(path: str) -> str:
    """ Read data as str from disk, empty str, if not file"""
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    return data

def save_str_data_to_disk(data: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as filehandle:
        data = filehandle.write(data)

def read_as_pandas_from_disk(path: str) -> pd.DataFrame:
    """ Read data as pdDf from disk, empty df, if not file"""
    if not os.path.isfile(path):
        return pd.DataFrame(columns=DATA_COLUMNS)
    data = pd.read_table(path, names=DATA_COLUMNS, sep=DATA_SEP)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def write_pandas_data_to_disk(data: pd.DataFrame, path: str) -> None:
    """ Writes Pandas Data Frame as str to path """
    data = pandas_to_str(data)
    save_str_data_to_disk(data, path)

def pandas_to_str(data: pd.DataFrame):
    """ Converts Pandas representation to string """
    out_data = ""
    for i in data.itertuples():
        out_data += f"{i.timestamp}, {i.temperature}, {i.humidity}\n"
    return out_data

def str_to_pandas(data: str) -> pd.DataFrame:
    """ Converts string representation to Pandas """
    return read_as_pandas_from_disk(data) #TODO: Check if that works
