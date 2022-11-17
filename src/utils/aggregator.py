import numpy as np
from more_itertools import run_length

def aggregate(data, base="hours"):
    if base == "hours":
        # TODO: Put to data parsing class
        data = data.split("\n")[:-1]
        timestamp = [val.split(",")[0] for val in data] # Extract values
        temperature = [float(val.split(",")[2]) for val in data] # Extract values
        rel_humidity = [float(val.split(",")[3]) for val in data] # Extract values

        # Convert to hours
        hours = [value.split(" ")[0] + " " + value.split(" ")[1][0:2] + ":00:00" for value in timestamp]
        # Aggregate indices
        indices = list(run_length.encode(hours))
        aggregated_timestamps = []
        aggregated_temperature = []
        aggregated_humidity = []
        runner = 0
        for idx in indices:
            aggregated_timestamps.append(idx[0])
            aggregated_temperature.append(int(np.mean(temperature[runner:runner+idx[1]])))
            aggregated_humidity.append(int(np.mean(rel_humidity[runner:runner+idx[1]])))
            runner += idx[1]
        out_data = ""
        for i in range(len(indices)):
            out_data += f"{aggregated_timestamps[i]}, TO_REMOVE, {aggregated_temperature[i]}, {aggregated_humidity[i]}\n"
        return out_data

    else:
        print("not implemented")
        return data