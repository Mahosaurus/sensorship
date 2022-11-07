import datetime
import math
from matplotlib.figure import Figure
import matplotlib.dates as dates

def load_data(path):
    with open(path, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    data = data.split("\n")[:-1]
    return data

def parse_data(data):
    timestamp = [val.split(",")[0] for val in data] # Extract values
    temperature = [float(val.split(",")[2]) for val in data] # Extract values
    humidity = [float(val.split(",")[3]) for val in data] # Extract values
    return timestamp, temperature, humidity

def create_figure():
    """ Creates figure from outcome.txt content """
    data = load_data("outcome_local.txt")
    timestamp, temperature, humidity = parse_data(data)
    idx = [dates.datestr2num(idx) for idx in  timestamp]

    # Determine interval
    mini = datetime.datetime.strptime(min(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
    maxi = datetime.datetime.strptime(max(timestamp), "%Y-%m-%d %H:%M:%S").timestamp()
    time_diff_mins = (maxi-mini)/60
    interval = math.ceil(time_diff_mins/16)

    fig = Figure(figsize=(10, 8))

    temperature_axis = fig.add_subplot(2, 1, 1)
    temperature_axis.plot_date(idx, temperature,
                 linestyle="--", dash_joinstyle="bevel", color="salmon", linewidth=0.6,
                 markerfacecolor="maroon", markeredgewidth=0.3,
                 fillstyle="full")
    temperature_axis.set_title("Temperature", fontdict={"fontweight": "bold", "color": "darkblue"})

    temperature_axis.xaxis.set_minor_locator(dates.MinuteLocator(interval=interval))   # every x mins
    temperature_axis.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

    temperature_axis.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
    temperature_axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))

    humidity_axis = fig.add_subplot(2, 1, 2)
    humidity_axis.plot_date(idx, humidity,
                 linestyle="--", dash_joinstyle="bevel", color="salmon", linewidth=0.6,
                 markerfacecolor="maroon", markeredgewidth=0.3,
                 fillstyle="full")

    humidity_axis.set_title("Humidity", fontdict={"fontweight": "bold", "color": "darkblue"})

    humidity_axis.xaxis.set_minor_locator(dates.MinuteLocator(interval=interval))   # every x mins
    humidity_axis.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

    humidity_axis.xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
    humidity_axis.xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))
    return fig
