import numpy as np
import pandas as pd
from more_itertools import run_length

from src.utils.helpers import parse_data_points

def aggregate(data: str, base: str="hours") -> str:
    timestamp, temperature, rel_humidity = parse_data_points(data)
    if base == "hours":
        # Convert to hours
        hours = [value.split(" ")[0] + " " + value.split(" ")[1][0:2] + ":00:00" for value in timestamp]
        # Aggregate indices
        # Logic is that we count the occurence of hours like this:
        # 9, 9, 10, 11, 11, 9 => [(9, 2), (10, 1), (11, 2), (9, 1)]
        # And calculate the mean according to a runner index that relates to that element
        indices = list(run_length.encode(hours))
        aggregated_timestamps = []
        aggregated_temperature = []
        aggregated_humidity = []
        runner = 0
        for idx in indices:
            aggregated_timestamps.append(idx[0])
            aggregated_temperature.append(round(np.mean(temperature[runner:runner+idx[1]]), 2))
            aggregated_humidity.append(round(np.mean(rel_humidity[runner:runner+idx[1]]), 2))
            runner += idx[1] # Move runner idx forward by idx[1] elements (next hour)
        # Make it a text list again
        out_data = ""
        for i in range(len(indices)):
            out_data += f"{aggregated_timestamps[i]}, TO_REMOVE, {aggregated_temperature[i]}, {aggregated_humidity[i]}\n"
        return out_data

    else:
        print("not implemented")
        return data