import io
import os
import time

from typing import Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
import requests

from src.utils.graphics import PlotSensor
from src.utils.tempsensor import get_mock_data
from src.utils.helpers import compile_data_point
from src.config import get_repo_root

import src.config as cfg

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

# Call it once to test it works, even with long scheduler
main()
sched = BackgroundScheduler(daemon=True)
sched.add_job(main, 'interval', seconds=5)
sched.start()

app = Flask(__name__)

@app.route("/")
def plot_png():
    fig = plotter.create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    plotter = PlotSensor(os.path.join(get_repo_root(), "outcome_local.txt"))
    app.run(host="localhost", port=8000, debug=True)
