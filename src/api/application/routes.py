

from flask import Response, render_template
from flask import current_app as app

from src.utils.io_interaction import read_as_str_from_disk, read_as_pandas_from_disk, write_pandas_data_to_disk, pandas_to_str

@app.route("/")
def home():
    """Landing page."""
    return render_template(
        "index.jinja2",
        title="Plotly Dash Flask Tutorial",
        description="Embed Plotly Dash into your Flask applications.",
        template="home-template",
        body="This is a homepage served with Flask.",
    )


@app.route("/text-data")
def text_data():
    data = read_as_str_from_disk(app.config["DATA_PATH"])
    return data.split("\n")

from src.utils.aggregator import aggregate

@app.route("/aggregate-data")
def aggregate_data():
    data = read_as_pandas_from_disk(app.config["DATA_PATH"])
    aggregated_data = aggregate(data)
    write_pandas_data_to_disk(aggregated_data, app.config["DATA_PATH"])
    return "Success"

import io
from matplotlib.backends.backend_agg import FigureCanvasAgg
from src.utils.graphics import PlotSensor
from src.utils.predictor import Predictor

@app.route("/show-data")
def plot_png():
    # Read existing data
    data = read_as_str_from_disk(app.config["DATA_PATH"])    
    # Add predicted data
    pred_data = read_as_pandas_from_disk(app.config["DATA_PATH"])
    predictor = Predictor(pred_data)
    result = predictor.make_lstm_prediction()
    pred_data = pandas_to_str(result)
    # Concat
    data = data + pred_data
    # Plot
    plotter = PlotSensor(data)    
    fig = plotter.create_figure(len(pred_data))
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

import os
import time
from src.utils.helpers import get_repo_root

def save_to_file(out_str):
    """ Write metrics to file """
    with open(app.config["DATA_PATH"], "a", encoding="utf-8") as filehandle:
        filehandle.write(out_str)

from src.utils.helpers import compile_data_point
from src.utils.tempsensor import get_mock_data

def main():
    """ Function that gets called by scheduler. """
    temperature, humidity = get_mock_data()
    current_ts = time.time()
    out_str = compile_data_point(current_ts, temperature, humidity)
    save_to_file(out_str)    
