import io
import os
from flask import Flask, render_template, request, send_from_directory, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from graphics import create_figure

app = Flask(__name__)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/sensor-data', methods=['PUT', 'DELETE'])
def get_data():
    if request.method == 'PUT':
        data = request.json
        with open("outcome_remote.txt", "a", encoding="utf-8") as filehandle:
            filehandle.write(data["data"])
        return f"Received {data}"
    if request.method == 'DELETE':
        os.remove("outcome_remote.txt")
        return "Removed sensor data"

@app.route("/show-data")
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
   app.run()
