
import io
import os

from flask import Response, render_template, request, send_file
from flask import current_app as app
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.utils.aggregator import aggregate
from src.utils.graphics import PlotSensor
from src.utils.predictor import Predictor

from src.utils.io_interaction import read_as_str_from_disk, read_as_pandas_from_disk, write_pandas_data_to_disk, pandas_to_str

@app.route("/")
def home():
    """Landing page."""
    return render_template(
        "index.jinja2",
        title="Entry page",
        template="home-template",
        body="Entry page for room condition dashboard",
    )

@app.route('/sensor-data', methods=['PUT'])
def get_data():
    if request.method == 'PUT':
        data = request.json
        with open(app.config["DATA_PATH"], "a", encoding="utf-8") as filehandle:
            filehandle.write(data["data"])
        return f"Received {data}"

@app.route("/text-data")
def text_data():
    return send_file(app.config["DATA_PATH"], as_attachment=True)

@app.route("/del-data")
def del_data():
    data = read_as_str_from_disk(app.config["DATA_PATH"])
    os.remove(app.config["DATA_PATH"])
    return "Deleted:\n" + data

@app.route("/aggregate-data")
def aggregate_data():
    data = read_as_pandas_from_disk(app.config["DATA_PATH"])
    aggregated_data = aggregate(data)
    write_pandas_data_to_disk(aggregated_data, app.config["DATA_PATH"])
    return "Success"

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
