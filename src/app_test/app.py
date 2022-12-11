import io
import os
import time

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.utils.aggregator import aggregate
from src.utils.graphics import PlotSensor, PlotPrediction
from src.utils.io_interaction import read_as_str_from_disk, read_as_pandas_from_disk, write_pandas_data_to_disk, pandas_to_str
from src.utils.tempsensor import get_mock_data
from src.utils.helpers import compile_data_point
from src.utils.predictor import Predictor
from src.config import get_repo_root
from src.config import APP_TEST_DATA_PATH


app = Flask(__name__)

def save_to_file(out_str):
    """ Write metrics to file """
    with open(os.path.join(get_repo_root(), "outcome_local.txt"), "a", encoding="utf-8") as filehandle:
        filehandle.write(out_str)

def main():
    """ Function that gets called by scheduler. """
    temperature, humidity = get_mock_data()
    current_ts = time.time()
    out_str = compile_data_point(current_ts, temperature, humidity)
    save_to_file(out_str)

@app.route("/")
def plot_png():
    # First aggregate
    data = read_as_pandas_from_disk(APP_TEST_DATA_PATH)
    aggregated_data = aggregate(data)
    write_pandas_data_to_disk(aggregated_data, APP_TEST_DATA_PATH)    
    # Add predicted data
    pred_data = read_as_pandas_from_disk(APP_TEST_DATA_PATH)
    predictor = Predictor(pred_data)
    result = predictor.make_lstm_prediction()
    pred_data = pandas_to_str(result)
    # Read existing data
    data = read_as_str_from_disk(APP_TEST_DATA_PATH)
    # Concat
    data = data + pred_data
    # Plot
    plotter = PlotSensor(data)    
    fig = plotter.create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route("/text-data")
def text_data():
    data = read_as_str_from_disk(APP_TEST_DATA_PATH)
    return data.split("\n")

@app.route("/aggregate-data")
def aggregate_data():
    data = read_as_pandas_from_disk(APP_TEST_DATA_PATH)
    aggregated_data = aggregate(data)
    write_pandas_data_to_disk(aggregated_data, APP_TEST_DATA_PATH)
    return "Success"

@app.route("/predict-data")
def predict():
    data = read_as_pandas_from_disk(APP_TEST_DATA_PATH)
    predictor = Predictor(data)
    result = predictor.make_lstm_prediction()
    pred_plotter = PlotPrediction(result)
    fig = pred_plotter.create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    
if __name__ == "__main__":
    #main() # Call it once to test it works, even with long scheduler
    #sched = BackgroundScheduler(daemon=True)
    #sched.add_job(main, 'interval', seconds=5)
    #sched.start()
    app.run(host="localhost", port=8000, debug=True)
