import io
import os

from flask import Flask, render_template, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.utils.aggregator import aggregate
from src.utils.graphics import PlotSensor
from src.predictor.startnet import StartNet
from src.utils.predictor import make_prediction
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
    with open(API_DATA_PATH, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    os.remove(API_DATA_PATH)
    return "Deleted:\n" + data

@app.route("/text-data")
def text_data():
    with open(API_DATA_PATH, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    return data.split("\n")

@app.route("/aggregate-data")
def aggregate_data():
    with open(API_DATA_PATH, "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    aggregated_data = aggregate(data)
    with open(API_DATA_PATH, "w", encoding="utf-8") as filehandle:
        data = filehandle.write(aggregated_data)
    return "Success"

@app.route("/predict-data")
def predict():
    result = make_prediction()
    return result    

if __name__ == '__main__':
    startnet = StartNet()
    print(startnet)    
    app.run()
