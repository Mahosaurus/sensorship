import io
import os

from flask import Flask, render_template, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.utils.aggregator import aggregate
from src.utils.io_interaction import read_data, write_data
from src.utils.graphics import PlotSensor, PlotPrediction
from src.utils.predictor import Predictor
from src.config import API_DATA_PATH

app = Flask(__name__)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/sensor-data', methods=['PUT'])
def get_data():
    if request.method == 'PUT':
        data = request.json
        with open(API_DATA_PATH, "a", encoding="utf-8") as filehandle:
            filehandle.write(data["data"])
        return f"Received {data}"

@app.route("/show-data")
def plot_png():
    plotter = PlotSensor(API_DATA_PATH)
    fig = plotter.create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route("/del-data")
def del_data():
    data = read_data(API_DATA_PATH)
    os.remove(API_DATA_PATH)
    return "Deleted:\n" + data

@app.route("/text-data")
def text_data():
    data = read_data(API_DATA_PATH)
    return data.split("\n")

@app.route("/aggregate-data")
def aggregate_data():
    data = read_data(API_DATA_PATH)
    aggregated_data = aggregate(data)
    write_data(aggregated_data, API_DATA_PATH)
    return "Success"

@app.route("/predict-data")
def predict():
    predictor = Predictor(API_DATA_PATH)
    result = predictor.make_lstm_prediction()
    pred_plotter = PlotPrediction(result)
    fig = pred_plotter.create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
if __name__ == '__main__':
    app.run()
