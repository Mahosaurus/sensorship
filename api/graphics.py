from matplotlib.figure import Figure
import matplotlib.ticker as mticker

def create_figure():
    """ Creates figure from outcome.txt content """
    with open("outcome_remote.txt", "r", encoding="utf-8") as filehandle:
        data = filehandle.read()
    data = data.split("\n")[:-1]

    timestamp = [val.split(",")[0] for val in data] # Extract values
    temperature = [float(val.split(",")[2]) for val in data] # Extract values
    humidity = [float(val.split(",")[3]) for val in data] # Extract values

    fig = Figure(figsize=(10, 8))
    xs = list(range(len(timestamp))) # Generic x-axis

    temperature_axis = fig.add_subplot(2, 1, 1)
    temperature_axis.set_title("Temperature")
    # Set correct lables on x-axis
    temperature_axis.set_xticklabels(timestamp)
    # Rotate by 45°
    fig.autofmt_xdate(rotation=45)
    # Make font smaller
    temperature_axis.tick_params(axis='both', which='major', labelsize=7)
    temperature_axis.plot(xs, temperature)

    humidity_axis = fig.add_subplot(2, 1, 2)
    humidity_axis.set_title("Humidity")
    # Set correct lables on x-axis
    humidity_axis.set_xticklabels(timestamp)
    # Rotate by 45°
    fig.autofmt_xdate(rotation=45)
    # Make font smaller
    humidity_axis.tick_params(axis='both', which='major', labelsize=7)
    humidity_axis.plot(xs, humidity)
    return fig
