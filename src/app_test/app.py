import io
import os
import time

from typing import Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.utils.aggregator import aggregate
from src.utils.graphics import PlotSensor
from src.utils.tempsensor import get_mock_data
from src.utils.helpers import compile_data_point
from src.utils.predictor import make_prediction
from src.predictor.startnet import StartNet
from src.config import get_repo_root
from src.config import APP_TEST_DATA_PATH


app = Flask(__name__)

def get_data() -> Tuple[str, str]:
    """ Data ingestion switch """
    return get_mock_data()

def save_to_file(out_str):
    """ Write metrics to file """
    with open(os.path.join(get_repo_root(), "outcome_local.txt"), "a", encoding="utf-8") as filehandle:
        filehandle.write(out_str)

def main():
    """ Function that gets called by scheduler. """
    temperature, humidity = get_data()
    current_ts = time.time()
    out_str = compile_data_point(current_ts, temperature, humidity)
    save_to_file(out_str)

@app.route("/")
def plot_png():
    fig = plotter.create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route("/text-data")
def text_data():
    with open(APP_TEST_DATA_PATH, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    return data.split("\n")

@app.route("/aggregate-data")
def aggregate_data():
    with open(APP_TEST_DATA_PATH, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    aggregated_data = aggregate(data)
    with open(APP_TEST_DATA_PATH, "w", encoding="utf-8") as filehandle:
        data = filehandle.write(aggregated_data)
    return "Success"

@app.route("/predict-data")
def predict():
    result = make_prediction()
    return result

if __name__ == "__main__":
    main() # Call it once to test it works, even with long scheduler
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(main, 'interval', seconds=5)
    sched.start()
    plotter = PlotSensor(APP_TEST_DATA_PATH)
    app.run(host="localhost", port=8000, debug=True)
