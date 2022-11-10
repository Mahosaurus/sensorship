import datetime
import math

from matplotlib.figure import Figure
import matplotlib.dates as dates
import numpy as np

INPUT_PATH = "outcome_remote.txt"

def load_data(path):
    with open(path, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    data = data.split("\n")[:-1]
    return data

def parse_data(data):
    """ Parse date from sensor """
    timestamp = [val.split(",")[0] for val in data] # Extract values
    time_of_day = [val.split(",")[1].strip() for val in data] # Extract values
    temperature = [float(val.split(",")[2]) for val in data] # Extract values
    rel_humidity = [float(val.split(",")[3]) for val in data] # Extract values
    abs_humidity = [humidity/(288.68 * (1.098 + temp/100)**8.02) for humidity, temp in zip (rel_humidity, temperature)]
    return timestamp, time_of_day, temperature, rel_humidity, abs_humidity

def determine_x_axis_interval(timestamp):
    """ Determine the interval to be 16 steps """
    mini = datetime.datetime.strptime(min(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
    maxi = datetime.datetime.strptime(max(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
    time_diff_mins = (maxi-mini)/60
    interval = math.ceil(time_diff_mins/16)
    return interval

def create_figure():
    """ Creates figure from outcome.txt content """
    data = load_data(INPUT_PATH)
    timestamp, time_of_day, temperature, rel_humidity, abs_humidity = parse_data(data)
    idx = [dates.datestr2num(idx) for idx in timestamp] # Conversion to proper timestamp
    colormap = {'Night': 'black', 'Morning': 'orangered', 'Day': 'indigo', 'Afternoon':'maroon', 'Evening': "cyan"}
    interval = determine_x_axis_interval(timestamp)

    # Start plotting
    fig = Figure(figsize=(13, 8))
    fig.subplots_adjust(hspace=0.4, top=0.95)

    # Temperature
    temperature_axis = fig.add_subplot(3, 1, 1)

    for label in set(time_of_day):
        x = [i if tod == label else np.nan for i, tod in zip(idx, time_of_day)]
        y = [temp if tod == label else np.nan  for temp, tod in zip(temperature, time_of_day)]
        temperature_axis.plot(x, y,
                    linestyle="--", dash_joinstyle="bevel", color=colormap[label], linewidth=0.6,
                    marker=".", markerfacecolor="maroon", markeredgewidth=0.2,
                    fillstyle="full")
    temperature_axis.set_title("Temperature", fontdict={"fontweight": "bold", "color": "darkblue"})

    temperature_axis.xaxis.set_minor_locator(dates.MinuteLocator(interval=interval))   # every x mins
    temperature_axis.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

    temperature_axis.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
    temperature_axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))

    # Rel Humidity
    rel_humidity_axis = fig.add_subplot(3, 1, 2)

    for label in set(time_of_day):
        x = [i if tod == label else np.nan for i, tod in zip(idx, time_of_day)]
        y = [rel_hum if tod == label else np.nan for rel_hum, tod in zip(rel_humidity, time_of_day)]
        rel_humidity_axis.plot(x, y,
                    linestyle="--", dash_joinstyle="bevel", color=colormap[label], linewidth=0.6,
                    marker=".", markerfacecolor="maroon", markeredgewidth=0.2,
                    fillstyle="full")

    rel_humidity_axis.set_title("Relative Humidity", fontdict={"fontweight": "bold", "color": "darkblue"})

    rel_humidity_axis.xaxis.set_minor_locator(dates.MinuteLocator(interval=interval))   # every x mins
    rel_humidity_axis.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

    rel_humidity_axis.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
    rel_humidity_axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))

    # Abs Humidity
    abs_humidity_axis = fig.add_subplot(3, 1, 3)

    for label in set(time_of_day):
        x = [i if tod == label else np.nan for i, tod in zip(idx, time_of_day)]
        y = [abs_hum if tod == label else np.nan for abs_hum, tod in zip(abs_humidity, time_of_day)]
        abs_humidity_axis.plot(x, y,
                                linestyle="--", dash_joinstyle="bevel", color=colormap[label], linewidth=0.6,
                                marker=".", markerfacecolor="maroon", markeredgewidth=0.2,
                                fillstyle="full")

    abs_humidity_axis.set_title("Absolute Humidity", fontdict={"fontweight": "bold", "color": "darkblue"})

    abs_humidity_axis.xaxis.set_minor_locator(dates.MinuteLocator(interval=interval))   # every x mins
    abs_humidity_axis.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

    abs_humidity_axis.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
    abs_humidity_axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))
    return fig
