
import io
import os

from flask import Response, render_template, request, send_file
from flask import current_app as app
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd

from src.data_visualization.graphics import PlotSensor
from src.data_prediction.predictor import Predictor
from src.data_handling.io_interaction import pandas_to_str
from src.data_storage.postgres_interaction import PostgresInteraction

postgres = PostgresInteraction(os.getenv("POSTGRES_HOST"),
                               os.getenv("POSTGRES_DBNAME"),
                               os.getenv("POSTGRES_USER"),
                               os.getenv("POSTGRES_PASSWORD"))

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
    """Receives data from sensor and writes it to disk."""
    if request.method == 'PUT':
        data = request.json
        if data['secret_key'] == app.config["SECRET_KEY"]:
            postgres.send_data_to_postgres(data)
            return f"Received {data}"
    return f"Received {data}, could not process using {data['secret_key']}"

@app.route("/show-data")
def plot_png():
    """ Shows data as plot."""
    # Read existing data
    history = postgres.load_data()

    # Add predicted data
    predictor = Predictor(history)
    prediction = predictor.make_lstm_prediction()

    # Merge the two dataframes
    data = pd.concat([history, prediction], ignore_index=True)
    # Transform dataframe to string
    # TODO: This is very clunky, find a better way!
    data = pandas_to_str(data)

    # Plot
    plotter = PlotSensor(data)
    fig = plotter.create_figure(len(data))
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route("/text-data", methods=['GET'])
def text_data():
    """ Shows data as text. """
    # https://sensorndf.azurewebsites.net/text-data?secret_key=
    secret_key = request.args.get("secret_key")
    if secret_key == app.config["SECRET_KEY"]:
        data = postgres.load_data()
        daten = pd.DataFrame(data, columns=['id', 'timestamp', 'temperature', 'humidity'])
        daten.drop(columns=['id'], inplace=True)
        # Sort data by timestamp
        daten = daten.sort_values(by=['timestamp'])
        daten.to_csv(app.config["DATA_PATH"], index=False, header=False)
        return send_file(app.config["DATA_PATH"], as_attachment=True)
    else:
        return render_template(
            "index.jinja2",
            title="Entry page",
            template="home-template",
            body="Entry page for room condition dashboard")
