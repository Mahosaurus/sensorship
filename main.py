import datetime
import io
import random
import time

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def get_sensor_data() -> str:
    """ Sensor data retrieval """
    with open("random_data.txt", "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    data = data.split("\n")
    choice = random.choice(data)
    return choice

def map_time_to_time_of_day(timestamp: str) -> str:
    """ Extract time of day from ts """
    hour = int(datetime.datetime.fromtimestamp(timestamp).strftime('%H'))
    class Switch(dict):
        """ Helper class to emulate switch statement """
        def __getitem__(self, item):
            for key in self.keys():                 # iterate over the intervals
                if item in key:                     # if the argument is in that interval
                    return super().__getitem__(key) # return its associated value
            raise KeyError(item)                    # if not in any interval, raise KeyError
    switch = Switch({
        range(0, 8): 'Night',
        range(8, 12): 'Morning',
        range(12, 19): 'Afternoon',
        range(18, 24): 'Night'
    })
    time_of_day = switch[hour]
    return time_of_day

def format_timestamp(timestamp: str) -> str:
    """ Convert ts to human readable """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def compile_data_point(timestamp, sensor_value) -> str:
    """ Compiles the information for one data point """
    formatted_timestamp = format_timestamp(timestamp)
    time_of_day = map_time_to_time_of_day(timestamp)
    out_str = f"{formatted_timestamp}, {time_of_day}, {sensor_value}\n"
    return out_str

def create_figure():
    """ Creates figure from outcome.txt content """
    with open("outcome.txt", "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    data = data.split("\n")[:-1]
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    ys = [int(val.split(",")[2]) for val in data] # Extract values
    xs = [i for i in range(len(ys))] # Generic x-axis (should be timestamp)
    axis.plot(xs, ys)
    return fig

def main():
    """ Function for test purposes. """
    sensor_value, current_ts = get_sensor_data(), time.time()
    out_str = compile_data_point(current_ts, sensor_value)
    with open("outcome.txt", "a", encoding="utf-8") as filehandle:
        filehandle.write(out_str)

sched = BackgroundScheduler(daemon=True)
sched.add_job(main, 'interval', seconds=1)
sched.start()

app = Flask(__name__)

@app.route("/")
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run()
