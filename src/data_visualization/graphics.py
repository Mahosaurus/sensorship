import datetime
import math

from typing import Tuple, List

from matplotlib.figure import Figure
import matplotlib.dates as dates
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.data_prediction.predictor import Predictor
from src.utils.helpers import parse_data_points
from src.data_handling.io_interaction import *

class PlotSensor():
    def __init__(self, data):
        self.data = data
        self.colormap = {'Night': 'darkolivegreen',
                         'Morning': 'teal',
                         'Day': 'indigo',
                         'Afternoon':'maroon',
                         'Evening': "purple"}

    def parse_data(self) -> Tuple[List[str], List[str], List[float], List[float], List[float]]:
        """ Parse data from sensor """
        timestamp, temperature, rel_humidity = parse_data_points(self.data)
        time_of_day = [self.map_time_to_time_of_day(ts) for ts in timestamp] # Extract values

        rel_to_abs_humidity = lambda temp, humidity: (6.112*math.exp((17.67*temp)/(temp + 243.5)) * humidity * 2.1674) / (273.15+temp)
        abs_humidity = [rel_to_abs_humidity(temp, humidity) for humidity, temp in zip (rel_humidity, temperature)]
        return timestamp, time_of_day, temperature, rel_humidity, abs_humidity

    @staticmethod
    def determine_minor_x_axis_interval(timestamp: List[str], steps: int=16) -> int:
        """ Determine the interval to be 16 steps, as this fits font size with plot """
        if len(timestamp) < 2: raise ValueError
        mini = datetime.datetime.strptime(min(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        maxi = datetime.datetime.strptime(max(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        time_diff_mins = (maxi-mini)/60 # Minute Locator
        interval = math.ceil(time_diff_mins/steps)
        return interval

    @staticmethod
    def determine_major_x_axis_interval(timestamp: List[str], steps: int=16) -> int:
        """ Determine the interval to be 16 steps, as this fits font size with plot """
        if len(timestamp) < 2: raise ValueError
        mini = datetime.datetime.strptime(min(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        maxi = datetime.datetime.strptime(max(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        time_diff_days = (maxi-mini)/(60*60*24) # Day Locator
        interval = math.ceil(time_diff_days/steps)
        return interval

    @staticmethod
    def map_time_to_time_of_day(timestamp: str) -> str:
        """ Extract time of day from ts """
        hour = int(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%H"))
        class Switch(dict):
            """ Helper class to emulate switch statement """
            def __getitem__(self, item):
                for key in self.keys():                 # iterate over the intervals
                    if item in key:                     # if the argument is in that interval
                        return super().__getitem__(key) # return its associated value
                raise KeyError(item)                    # if not in any interval, raise KeyError
        switch = Switch({
            range(0, 8): 'Night',
            range(8, 19): 'Day',
            range(19, 24): 'Night'
        })
        time_of_day = switch[hour]
        return time_of_day

    def create_figure(self, len_pred_data):
        """ Creates figure from outcome.txt content """
        timestamp, time_of_day, temperature, rel_humidity, abs_humidity = self.parse_data()
        idx = [dates.datestr2num(idx) for idx in timestamp] # Conversion to proper timestamp

        # Shorten time range to last 100 entries
        timestamp    = timestamp[-100:]
        time_of_day  = time_of_day[-100:]
        temperature  = temperature[-100:]
        rel_humidity = rel_humidity[-100:]
        abs_humidity = abs_humidity[-100:]
        idx          = idx[-100:]

        interval_minor = self.determine_minor_x_axis_interval(timestamp)
        interval_major = self.determine_major_x_axis_interval(timestamp)

        # Start plotting
        fig = Figure(figsize=(13, 8))
        fig.subplots_adjust(hspace=0.4, top=0.95)

        # Subplots
        temperature_axis = fig.add_subplot(3, 1, 1)
        rel_humidity_axis = fig.add_subplot(3, 1, 2)
        abs_humidity_axis = fig.add_subplot(3, 1, 3)
        self.generic_plot("Temperature", temperature_axis, temperature, idx, time_of_day, interval_minor, interval_major)
        self.generic_plot("Rel Humidity", rel_humidity_axis, rel_humidity, idx, time_of_day, interval_minor, interval_major)
        self.generic_plot("Abs Humidity", abs_humidity_axis, abs_humidity, idx, time_of_day, interval_minor, interval_major)
        # If short or no predicted data -> -1
        cutoff = -1 if len(idx) < 25 or len_pred_data == 0 else -24

        temperature_axis.axvline(x=idx[cutoff], c='r', linestyle='--')
        rel_humidity_axis.axvline(x=idx[cutoff], c='r', linestyle='--')
        abs_humidity_axis.axvline(x=idx[cutoff], c='r', linestyle='--')

        return fig

    def generic_plot(self, title: str, axis, variable, idx: int, time_of_day, interval_minor, interval_major):
        """ Creates generic plot """
        for label in set(time_of_day):
            x = [i if tod == label else np.nan for i, tod in zip(idx, time_of_day)]
            y = [temp if tod == label else np.nan  for temp, tod in zip(variable, time_of_day)]
            axis.plot(x, y,
                        linestyle="--", dash_joinstyle="bevel", color=self.colormap[label], linewidth=0.6,
                        marker=".", markerfacecolor=self.colormap[label], markeredgewidth=0.2,
                        fillstyle="full")
        axis.set_title(title, fontdict={"fontweight": "bold", "color": "darkblue"})

        axis.xaxis.set_minor_locator(dates.MinuteLocator(interval=interval_minor))   # every x mins
        axis.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

        axis.xaxis.set_major_locator(dates.DayLocator(interval=interval_major))    # every day
        axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))

class PlotDashboard():
    """Prepare data for Plotly Dash."""

    def __new__(cls, *args, **kwargs):
        """ Singleton implementation
        https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons/33201#33201
        We need the singleton, as the main instance knows the data path, which cannot be shared with the
        callbacks for dash (as that would require passing of server instance)
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
        self.data = read_as_pandas_from_disk(self.data_path)
        self.pred_data = None
        self.length_prediction = 0

    def update_data(self):
        """ Updates with the latest data from disk """
        self.data = read_as_pandas_from_disk(self.data_path)

    def update_prediction(self):
        """ Updates prediction based on self.data """
        predictor = Predictor(self.data)
        self.pred_data = predictor.make_lstm_prediction()
        self.length_prediction = len(self.pred_data)

    def concat_data(self):
        """ Concats data and prediction """
        return pd.concat([self.data, self.pred_data], ignore_index=True)

    def append_to_data(self, data):
        """ Appends and lints data """
        # Add Abs Humidity
        convert_rel_to_abs_humidity = lambda x: (6.112*math.exp((17.67*x["temperature"])/(x["temperature"] + 243.5)) * x["humidity"] * 2.1674) / (273.15+x["temperature"])
        data["abs_humidity"] = data.apply(convert_rel_to_abs_humidity, axis=1)
        # Need this, as Dash cannot deal with objects
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y/%m/%d %H:%M:%S')
        # Need this for slider
        data['slider'] = data['timestamp'].astype(np.int64) // 1e9
        return data

    def generate_plot(self, style: str):
        "Runs the steps to generate a generic scatter plot in plotly"
        self.update_data()
        if len(self.data) == 0:
            return go.Figure()

        self.update_prediction()
        data = self.concat_data()
        data = self.append_to_data(data)


        x = data["timestamp"]
        y = data[style]
        _lower_bound = min(y) - 0.5
        _upper_bound = max(y) + 0.5

        fig = go.Figure()
        # Adding scatter lines
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                connectgaps=False,
                mode='lines+markers'
            )
        )
        fig.update_layout(
            title_text=style.capitalize(),
            # Adding fixed scale
            yaxis=dict(
                range=[_lower_bound, _upper_bound]
            ),
            # White Background
            plot_bgcolor='white'
        )

        # Adding back lines
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        # Adding back lines
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )

        # Prediction marker
        length_for_prediction = 24
        prediction_horizon = 24
        if len(data) > length_for_prediction + prediction_horizon:
            fig.add_vline(
                x=data.at[len(data)-self.length_prediction , "timestamp"],
                line_color="red")
            fig.add_vrect(
                x0=data.at[len(data)-self.length_prediction , "timestamp"],
                x1=data.at[len(data)-1, "timestamp"],
                line_width=0,
                fillcolor="red",
                opacity=0.2)
        return fig
