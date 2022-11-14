import datetime
import math

from matplotlib.figure import Figure
import matplotlib.dates as dates
import numpy as np

class PlotSensor():
    def __init__(self, source_path):
        self.source_path = source_path
        self.colormap = {'Night': 'darkolivegreen',
                         'Morning': 'teal',
                         'Day': 'indigo',
                         'Afternoon':'maroon',
                         'Evening': "purple"}

    def load_data(self):
        with open(self.source_path, "r", encoding="utf-8") as filehandle:
            data = filehandle.read()
        data = data.split("\n")[:-1]
        return data

    def parse_data(self, data):
        """ Parse date from sensor """
        timestamp = [val.split(",")[0] for val in data] # Extract values
        time_of_day = [self.map_time_to_time_of_day(ts) for ts in timestamp] # Extract values
        temperature = [float(val.split(",")[2]) for val in data] # Extract values
        rel_humidity = [float(val.split(",")[3]) for val in data] # Extract values
        abs_humidity = [humidity/(288.68 * (1.098 + temp/100)**8.02) for humidity, temp in zip (rel_humidity, temperature)]
        return timestamp, time_of_day, temperature, rel_humidity, abs_humidity

    @staticmethod
    def determine_x_axis_interval(timestamp):
        """ Determine the interval to be 16 steps """
        mini = datetime.datetime.strptime(min(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        maxi = datetime.datetime.strptime(max(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
        time_diff_mins = (maxi-mini)/60
        interval = math.ceil(time_diff_mins/16)
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
            range(0, 6): 'Night',
            range(6, 10): 'Morning',
            range(10, 14): 'Day',
            range(14, 17): 'Afternoon',
            range(17, 21): 'Evening',
            range(21, 24): 'Night'
        })
        time_of_day = switch[hour]
        return time_of_day

    def create_figure(self):
        """ Creates figure from outcome.txt content """
        data = self.load_data()
        timestamp, time_of_day, temperature, rel_humidity, abs_humidity = self.parse_data(data)
        idx = [dates.datestr2num(idx) for idx in timestamp] # Conversion to proper timestamp

        interval = self.determine_x_axis_interval(timestamp)

        # Start plotting
        fig = Figure(figsize=(13, 8))
        fig.subplots_adjust(hspace=0.4, top=0.95)

        # Subplots
        temperature_axis = fig.add_subplot(3, 1, 1)
        rel_humidity_axis = fig.add_subplot(3, 1, 2)
        abs_humidity_axis = fig.add_subplot(3, 1, 3)
        self.create_plot(temperature_axis, temperature, idx, time_of_day, interval)
        self.create_plot(rel_humidity_axis, rel_humidity, idx, time_of_day, interval)
        self.create_plot(abs_humidity_axis, abs_humidity, idx, time_of_day, interval)

        return fig

    def create_plot(self, axis, variable, idx, time_of_day, interval):
        """ Creates generic plot """
        for label in set(time_of_day):
            x = [i if tod == label else np.nan for i, tod in zip(idx, time_of_day)]
            y = [temp if tod == label else np.nan  for temp, tod in zip(variable, time_of_day)]
            axis.plot(x, y,
                        linestyle="--", dash_joinstyle="bevel", color=self.colormap[label], linewidth=0.6,
                        marker=".", markerfacecolor=self.colormap[label], markeredgewidth=0.2,
                        fillstyle="full")
        axis.set_title("Temperature", fontdict={"fontweight": "bold", "color": "darkblue"})

        axis.xaxis.set_minor_locator(dates.MinuteLocator(interval=interval))   # every x mins
        axis.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

        axis.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
        axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))
