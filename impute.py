"""
Script to impute data
"""
import datetime
import os
import time
import sys
import pandas as pd
from data_storage.postgres_interaction import send_data_to_postgres, load_data, delete_database
from src.data_prediction.predictor import Predictor

##############################################################################################################
# Delete and Backfill with existing data
##############################################################################################################
SET_UP_DATABASE_ANEW = False
if SET_UP_DATABASE_ANEW:
    print("!WARNING! You are about to delete the database and backfill it with data from example_data/outcome_local.txt")
    print("Killing database in 10 seconds...")
    time.sleep(10)
    delete_database(os.getenv("POSTGRES_HOST"),
                    os.getenv("POSTGRES_DBNAME"),
                    os.getenv("POSTGRES_USER"),
                    os.getenv("POSTGRES_PASSWORD"))

    data = pd.read_csv("./src/example_data/outcome_local.txt", header=None)
    for row in data.values:
        data = {"data": f"{row[0]}, {row[1]}, {row[2]}"}
        send_data_to_postgres(os.getenv("POSTGRES_HOST"),
                              os.getenv("POSTGRES_DBNAME"),
                              os.getenv("POSTGRES_USER"),
                              os.getenv("POSTGRES_PASSWORD"), data)
    sys.exit(0)

data = load_data(os.getenv("POSTGRES_HOST"),
                        os.getenv("POSTGRES_DBNAME"),
                        os.getenv("POSTGRES_USER"),
                        os.getenv("POSTGRES_PASSWORD"))
data = pd.DataFrame(data, columns=['id', 'timestamp', 'temperature', 'humidity'])
print(data)
df

###############################################################################################################
# Actual imputation
###############################################################################################################
impute_max_date = '2023-10-03' # The date until which we want to impute data
# BE CAREFUL WITH THE LIMIT HERE!

for i in range(1, 34):
    data = load_data(os.getenv("POSTGRES_HOST"),
                         os.getenv("POSTGRES_DBNAME"),
                         os.getenv("POSTGRES_USER"),
                         os.getenv("POSTGRES_PASSWORD"))

    data = pd.DataFrame(data, columns=['id', 'timestamp', 'temperature', 'humidity'])
    # Sort data by timestamp
    data = data.sort_values(by=['timestamp'])
    # Keep all rows where timestamp is smaller than impute_max_date
    data = data[data['timestamp'] < impute_max_date]

    model = Predictor(data)
    rows_of_prediction = model.make_lstm_prediction()

    for ix, prediction in enumerate(rows_of_prediction.values):
        # Get last timestamp in data as datetime object
        last_timestamp = data['timestamp'].iloc[-1]
        last_timestamp = datetime.datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S')
        # Add one hour to last timestamp
        last_timestamp = last_timestamp + datetime.timedelta(hours=ix+1)
        # Convert last timestamp to string
        last_timestamp = last_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        # Create datapoint
        data = {"data": f"{last_timestamp}, {prediction[1]}, {prediction[2]}"}
        send_data_to_postgres(os.getenv("POSTGRES_HOST"),
                              os.getenv("POSTGRES_DBNAME"),
                              os.getenv("POSTGRES_USER"),
                              os.getenv("POSTGRES_PASSWORD"), data)
