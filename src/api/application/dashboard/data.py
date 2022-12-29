
"""Prepare data for Plotly Dash."""
import math
import pandas as pd
from src.utils.io_interaction import read_as_pandas_from_disk
from src.utils.predictor import Predictor

def load_and_prepare_data(server):
    # Load DataFrame
    data = read_as_pandas_from_disk(server.config["DATA_PATH"])    
    # Add predicted data
    pred_data = read_as_pandas_from_disk(server.config["DATA_PATH"])
    predictor = Predictor(pred_data)
    pred_data = predictor.make_lstm_prediction()
    # Concat the two
    data = data.append(pred_data, ignore_index=True)
    # Add Abs Humidity
    convert_rel_to_abs_humidity = lambda x: (6.112*math.exp((17.67*x["temperature"])/(x["temperature"] + 243.5)) * x["humidity"] * 2.1674) / (273.15+x["temperature"])
    data["abs_humidity"] = data.apply(convert_rel_to_abs_humidity, axis=1)
    # Need this, as Dash cannot deal with objects
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y/%m/%d %H:%M:%S')
    return data
