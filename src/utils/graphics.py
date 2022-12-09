import datetime
import math

from matplotlib.figure import Figure
import matplotlib.dates as dates
import pandas as pd
import numpy as np

from src.utils.io_interaction import read_data
from src.utils.helpers import parse_data_points

class PlotSensor():
    def __init__(self, source_path):
        self.source_path = source_path
        self.colormap = {'Night': 'darkolivegreen',
                         'Morning': 'teal',
                         'Day': 'indigo',
                         'Afternoon':'maroon',
                         'Evening': "purple"}

    def parse_data(self, data):
        """ Parse date from sensor """
        timestamp, temperature, rel_humidity = parse_data_points(data)
        time_of_day = [self.map_time_to_time_of_day(ts) for ts in timestamp] # Extract values

        conv_to_abs_humidity = lambda temp, humidity: (6.112*math.exp((17.67*temp)/(temp + 243.5)) * humidity * 2.1674) / (273.15+temp)
        abs_humidity = [conv_to_abs_humidity(temp, humidity) for humidity, temp in zip (rel_humidity, temperature)]
        return timestamp, time_of_day, temperature, rel_humidity, abs_humidity

    @staticmethod
    def determine_minor_x_axis_interval(timestamp, steps=16) -> int:
        """ Determine the interval to be 16 steps, as this fits font size with plot """
        mini = datetime.datetime.strptime(min(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        maxi = datetime.datetime.strptime(max(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        time_diff_mins = (maxi-mini)/60 # Minute Locator
        interval = math.ceil(time_diff_mins/steps)
        return interval

    @staticmethod
    def determine_major_x_axis_interval(timestamp, steps=16) -> int:
        """ Determine the interval to be 16 steps, as this fits font size with plot """
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

    def create_figure(self):
        """ Creates figure from outcome.txt content """
        data = read_data(self.source_path)
        timestamp, time_of_day, temperature, rel_humidity, abs_humidity = self.parse_data(data)
        idx = [dates.datestr2num(idx) for idx in timestamp] # Conversion to proper timestamp

        interval_minor = self.determine_minor_x_axis_interval(timestamp)
        interval_major = self.determine_major_x_axis_interval(timestamp)

        # Start plotting
        fig = Figure(figsize=(13, 8))
        fig.subplots_adjust(hspace=0.4, top=0.95)

        # Subplots
        temperature_axis = fig.add_subplot(3, 1, 1)
        rel_humidity_axis = fig.add_subplot(3, 1, 2)
        abs_humidity_axis = fig.add_subplot(3, 1, 3)
        self.create_plot("Temperature", temperature_axis, temperature, idx, time_of_day, interval_minor, interval_major)
        self.create_plot("Rel Humidity", rel_humidity_axis, rel_humidity, idx, time_of_day, interval_minor, interval_major)
        self.create_plot("Abs Humidity", abs_humidity_axis, abs_humidity, idx, time_of_day, interval_minor, interval_major)

        return fig

    def create_plot(self, title: str, axis, variable, idx: int, time_of_day, interval_minor, interval_major):
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

class PlotPrediction():
    
    def __init__(self, data: pd.DataFrame):
        self.data_subset = data[["timestamp", "predictions"]]
        self.data_subset = self.data_subset.to_dict(orient="list")

    def create_figure(self):
        fig = Figure(figsize=(13, 8))
        axis = fig.add_subplot(1, 1, 1)
        axis.plot(self.data_subset["timestamp"], self.data_subset["predictions"],
                    linestyle="--", dash_joinstyle="bevel", linewidth=0.6,
                    marker=".", markeredgewidth=0.2,
                    fillstyle="full")  
        axis.xaxis.set_major_locator(dates.HourLocator(interval=6))    # every day
        axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y %H:%M'))                    
        axis.set_title("Predictions temperature next 24h", fontdict={"fontweight": "bold", "color": "darkblue"})      
        return fig



