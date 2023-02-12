import os
import time

from typing import Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
import requests

from src.data_collection.tempsensor import get_sensor_data
from src.utils.helpers import compile_data_point

def get_data() -> Tuple[str, str]:
    """ Data ingestion switch """
    return get_sensor_data()

def send_to_app(out_str):
    """ Try to send metrics to App """
    try:
        dict_to_send = {"data": out_str}
        dict_to_send = {'secret_key': os.environ.get("SECRET_KEY")}
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
app = Flask(__name__)

if __name__ == "__main__":
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(main, 'interval', seconds=60*60)
    sched.start()
    # Debug Mode False, as otherwise there will be two instances of the scheduler
    app.run(host="localhost", port=8000, debug=False)
