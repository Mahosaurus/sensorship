import datetime
import os

from collections import deque

import joblib
import torch

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.predictor.temperature_ffn_model import FFNModel
from src.predictor.temperature_lstm_model import LSTMModel
from src.config import get_repo_root

def load_ffn_temp_model():
    path_to_model = os.path.join(get_repo_root(), "predictor", "temperature_ffn.model")
    model = FFNModel()
    if not os.path.isfile(path_to_model):
        print("Model state dict not found")
        return None
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model

def load_lstm_temp_model():
    path_to_model = os.path.join(get_repo_root(), "predictor", "temperature_lstm.model")
    path_to_preproc = os.path.join(get_repo_root(), "predictor", "lstm_preproc.joblib")
    model = LSTMModel()
    if not os.path.isfile(path_to_model):
        print("Model state dict not found")
        return None
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    preprocessor = joblib.load(path_to_preproc) # Imports Fitted and Transformed Min Max Scaler
    return model, preprocessor

def make_24hrs():
    """ Create timestamps for next 24hrs from now on"""
    current_time = datetime.datetime.now()
    next_24hrs = [current_time]
    for i in range(1, 25):
        next_24hrs.append(current_time + datetime.timedelta(hours=i))
    data_parsed = pd.DataFrame.from_dict(next_24hrs)      
    data_parsed.columns = ["timestamp"]
    return data_parsed

def get_features(timestamp_df):
    """ Extracts features from timestamp column """
    # Add hour
    timestamp_df["hour"] = timestamp_df["timestamp"].dt.hour
    # Add day of year
    timestamp_df["day_of_year"] = timestamp_df["timestamp"].dt.day_of_year
    # Add weekday
    timestamp_df["weekday"] = timestamp_df["timestamp"].dt.weekday 
    return timestamp_df   

def make_ffn_prediction() -> pd.DataFrame:
    timestamps_next_24hrs = make_24hrs()
    features = get_features(timestamps_next_24hrs)
    model = load_ffn_temp_model()
    predictions = []
    for _, row in features.iterrows():
        row["hour"] = float(row["hour"]) # Convert to float
        prediction = model(torch.tensor([row["hour"], row["weekday"], row["day_of_year"]]))
        predictions.append(round(prediction.tolist()[0], 2))
    features["predictions"] = predictions
    return features

def make_lstm_prediction() -> pd.DataFrame:
    model, sc = load_lstm_temp_model()
    # This is just to 
    features = make_24hrs()
    
    cache = deque([], maxlen=24)
    # TODO replace them by last hourly values
    start_values = [18, 18, 19, 19, 20, 20, 21, 21, 20, 20, 19, 19, 18, 17, 18, 19, 20, 19, 19, 19, 19, 20, 19, 19, 19] 
    for value in start_values:
        cache.append(value)

    predictions = []
    for _, _ in features.iterrows():
        transformed_cache = [sc.transform([[cache[0]]]), sc.transform([[cache[1]]]), sc.transform([[cache[2]]]), sc.transform([[cache[3]]])]
        prediction = model(torch.Tensor(transformed_cache))
        prediction_transformed = sc.inverse_transform(prediction.detach().numpy())[0][0]
        predictions.append(prediction_transformed)
        print("Cache", cache)
        cache.append(prediction_transformed)

    features["predictions"] = predictions
    return features



