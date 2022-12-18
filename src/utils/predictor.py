import datetime
import os

from collections import deque

import joblib
import torch

import pandas as pd

from src.predictor.temperature_ffn_model import FFNModel
from src.predictor.temperature_lstm_model import LSTMModel
from src.config import get_repo_root
from src.config import LSTM_INPUT_HISTORY, DATA_COLUMNS

class Predictor():
    def __init__(self, data):
        self.data = data

    @staticmethod
    def load_ffn_temp_model():
        path_to_model = os.path.join(get_repo_root(), "predictor", "temperature_ffn.model")
        model = FFNModel()
        if not os.path.isfile(path_to_model):
            print("Model state dict not found")
            return None
        model.load_state_dict(torch.load(path_to_model))
        model.eval()
        return model

    @staticmethod
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

    @staticmethod
    def make_24hrs() -> pd.DataFrame:
        """ Create timestamps for next 24hrs from now on"""
        current_time = datetime.datetime.now()
        next_24hrs = []
        for i in range(1, 25):
            next_24hrs.append((current_time + datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"))
        data_parsed = pd.DataFrame.from_dict(next_24hrs)
        data_parsed.columns = ["timestamp"]
        return data_parsed

    @staticmethod
    def add_features(timestamp_df: pd.DataFrame) -> pd.DataFrame:
        """ Extracts features from timestamp column """
        # Add hour
        timestamp_df["hour"] = timestamp_df["timestamp"].dt.hour
        # Add day of year
        timestamp_df["day_of_year"] = timestamp_df["timestamp"].dt.day_of_year
        # Add weekday
        timestamp_df["weekday"] = timestamp_df["timestamp"].dt.weekday
        return timestamp_df

    def make_ffn_prediction(self) -> pd.DataFrame:
        timestamps_next_24hrs = self.make_24hrs()
        features = self.add_features(timestamps_next_24hrs)
        model = self.load_ffn_temp_model()
        predictions = []
        for _, row in features.iterrows():
            row["hour"] = float(row["hour"]) # Convert to float
            prediction = model(torch.tensor([row["hour"], row["weekday"], row["day_of_year"]]))
            predictions.append(round(prediction.tolist()[0], 2))
        features["predictions"] = predictions
        return features

    def get_last_24hourly_avg(self) -> list:
        # Get last 24 hourly averages
        times    = pd.DatetimeIndex(self.data.timestamp)
        agg_data = self.data.groupby([times.date, times.hour]).temperature.mean()
        values   = agg_data.to_list()
        return values[-LSTM_INPUT_HISTORY:]

    def make_lstm_prediction(self) -> pd.DataFrame:
        model, sc = self.load_lstm_temp_model()
        # This is just to make a data frame with right timestamps
        features = self.make_24hrs()

        # Get values
        past_24hrs_values = self.get_last_24hourly_avg()
        cache = deque([], maxlen=LSTM_INPUT_HISTORY)

        for value in past_24hrs_values:
            cache.append([value])
        cache = [sc.transform(cache).tolist()]

        if len(cache[0]) < LSTM_INPUT_HISTORY:
            print("History too short")
            return pd.DataFrame(columns=DATA_COLUMNS)

        predictions = []
        prediction = model(torch.Tensor(cache))
        for val in prediction[0]:
            prediction_transformed = sc.inverse_transform([[val.item()]])[0][0]
            predictions.append(prediction_transformed)

        features["temperature"] = predictions
        features["humidity"] = 55
        return features
