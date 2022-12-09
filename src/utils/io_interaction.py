import pandas as pd

from src.config import DATA_COLUMNS, DATA_SEP

def read_data(path: str) -> str:
    with open(path, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    return data

def write_data(data: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as filehandle:
        data = filehandle.write(data)
    
def read_as_pandas(path: str) -> pd.DataFrame:
    data = pd.read_table(path, names=DATA_COLUMNS, sep=DATA_SEP)
    return data
