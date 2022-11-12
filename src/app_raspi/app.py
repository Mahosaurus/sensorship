import os
import time

from typing import Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
import requests

from src.utils.tempsensor import get_sensor_data
from src.utils.helpers import compile_data_point

import src.config as cfg

def get_data() -> Tuple[str, str]:
    """ Data ingestion switch """
    return get_sensor_data()

def send_to_app(out_str):
    """ Try to send metrics to App """
    try:
        dict_to_send = {"data": out_str}
        res = requests.put(os.environ['LINK'], json=dict_to_send, verify=False)
        print(res, res.text)
    except Exception as exc:
        print(f"Error in sending metrics to App: {exc}")

def main():
    """ Function that gets called by scheduler. """
    temperature, humidity = get_data()
    current_ts = time.time()
    out_str = compile_data_point(current_ts, temperature, humidity)
    send_to_app(out_str)

# Call it once to test it works, even with long scheduler
main()
sched = BackgroundScheduler(daemon=True)
sched.add_job(main, 'interval', seconds=60*10)
sched.start()

app = Flask(__name__)

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)