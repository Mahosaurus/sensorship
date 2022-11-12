import io
import os
from flask import Flask, render_template, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.utils.graphics import PlotSensor
from src.config import API_DATA_PATH

app = Flask(__name__)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/sensor-data', methods=['PUT', 'DELETE'])
def get_data():
    if request.method == 'PUT':
        data = request.json
        with open(API_DATA_PATH, "a", encoding="utf-8") as filehandle:
            filehandle.write(data["data"])
        return f"Received {data}"
    if request.method == 'DELETE':
        os.remove(API_DATA_PATH)
        return "Removed sensor data"

@app.route("/show-data")
def plot_png():
    fig = plotter.create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    plotter = PlotSensor(API_DATA_PATH)
    app.run()
