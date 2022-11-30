import datetime

import os

import torch

import pandas as pd

from src.predictor.startnet import StartNet
from src.config import get_repo_root


def load_model():
    path_to_model = os.path.join(get_repo_root(), "predictor", "startnet_temperature.model")
    model = StartNet()
    if not os.path.isfile(path_to_model):
        print("Model state dict not found")
        return None
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model

def make_24hrs():
    current_time = datetime.datetime.now()
    next_24hrs = [current_time]
    for i in range(1, 25):
        next_24hrs.append(current_time + datetime.timedelta(hours=i))
    data_parsed = pd.DataFrame.from_dict(next_24hrs)      
    data_parsed.columns = ["timestamp"]
    return data_parsed

def get_features(timestamp_df):
    # Add hour
    timestamp_df["hour"] = timestamp_df["timestamp"].dt.hour
    # Add day of year
    timestamp_df["day_of_year"] = timestamp_df["timestamp"].dt.day_of_year
    # Add weekday
    timestamp_df["weekday"] = timestamp_df["timestamp"].dt.weekday 
    return timestamp_df   

def make_prediction() -> pd.DataFrame:
    timestamp_df = make_24hrs()
    features = get_features(timestamp_df)
    model = load_model()
    predictions = []
    for _, row in features.iterrows():
        row["hour"] = float(row["hour"]) # Convert to float
        prediction = model(torch.tensor([row["hour"], row["weekday"], row["day_of_year"]]))
        predictions.append(round(prediction.tolist()[0], 2))
    features["predictions"] = predictions
    return features

if __name__ == "__main__":
    load_model()


