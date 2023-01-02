
"""Prepare data for Plotly Dash."""
import math
import numpy as np
import pandas as pd    

import plotly.graph_objects as go

from src.utils.io_interaction import *

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
    # Need this for slider
    data['slider'] = data['timestamp'].astype(np.int64) // 1e9
    return data, len(pred_data)


def generate_plot(data: pd.DataFrame, style: str, len_pred: int):
    x = data["timestamp"]
    y = data[style]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            connectgaps=False,
            mode='lines+markers',
            marker=dict()
        )
    )
    fig.update_layout(
        title_text=style.capitalize()
    )
    # TODO: Makes lines at timestamp not at prediction
    fig.add_vline(
        x=data.at[len(data)-len_pred, "timestamp"],
        line_color="red")
    fig.add_vrect(
        x0=data.at[len(data)-len_pred, "timestamp"],
        x1=data.at[len(data)-1, "timestamp"],
        line_width=0,
        fillcolor="red",
        opacity=0.2)
    return fig