
"""Prepare data for Plotly Dash."""
import math
import numpy as np
import pandas as pd

import plotly.graph_objects as go

from src.utils.io_interaction import *

from src.utils.predictor import Predictor

class PlotDashboard():

    def __new__(cls, *args, **kwargs):
        """ Singleton implementation
        https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons/33201#33201
        """
        instances = cls.__dict__.get("__instances__")
        if instances is not None:
            return instances
        cls.__instances__ = instances = object.__new__(cls)
        instances.__init__(*args, **kwargs)
        return instances

    def __init__(self, server=None):
        if server is not None:
            self.data_path = server.config.get("DATA_PATH")
        self.colormap = {'Night': 'darkolivegreen',
                         'Morning': 'teal',
                         'Day': 'indigo',
                         'Afternoon':'maroon',
                         'Evening': "purple"}

    def load_and_prepare_data(self):
        "kk"
        # Load DataFrame
        data = read_as_pandas_from_disk(self.data_path)
        # Add predicted data
        pred_data = read_as_pandas_from_disk(self.data_path)
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


    def generate_plot(self, data: pd.DataFrame, style: str, len_pred: int):
        "lkj"
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