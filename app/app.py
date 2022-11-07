import io
import os
import time

from typing import Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
import requests

from graphics import create_figure
from tempsensor import get_mock_data, get_sensor_data
from helpers import compile_data_point

import config as cfg

def get_data() -> Tuple[str, str]:
    """ Data ingestion switch """
    if cfg.TEST_MODE:
        return get_mock_data()
    else:
        return get_sensor_data()

def save_to_file(out_str):
    """ Write metrics to file """
    with open("outcome_local.txt", "a", encoding="utf-8") as filehandle:
        filehandle.write(out_str)

# def send_to_app(out_str):
#     """ Try to send metrics to App """
#     try:
#         dict_to_send = {"data": out_str}
#         res = requests.put(os.environ['LINK'], json=dict_to_send, verify=False)
#         print(res, res.text)
#     except Exception as exc:
#         print(f"Error in sending metrics to App: {exc}")

def main():
    """ Function that gets called by scheduler. """
    temperature, humidity = get_data()
    current_ts = time.time()
    out_str = compile_data_point(current_ts, temperature, humidity)
    save_to_file(out_str)
<<<<<<< Updated upstream
    if not cfg.TEST_MODE:
        send_to_app(out_str)
=======
    #send_to_app(out_str)
>>>>>>> Stashed changes

# Call it once to test it works, even with long scheduler
main()
sched = BackgroundScheduler(daemon=True)
<<<<<<< Updated upstream
sched.add_job(main, 'interval', seconds=cfg.PERIOD)
=======
sched.add_job(main, 'interval', seconds=1)
>>>>>>> Stashed changes
sched.start()

app = Flask(__name__)

@app.route("/")
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
