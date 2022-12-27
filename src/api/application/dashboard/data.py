
"""Prepare data for Plotly Dash."""
import os

import numpy as np
import pandas as pd

from src.utils.helpers import get_repo_root

def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    path = os.path.join(get_repo_root(), "api", "data", "311-calls.csv")
    df = pd.read_csv(path, parse_dates=["created"])
    df["created"] = df["created"].dt.date
    df.drop(columns=["incident_zip"], inplace=True)
    num_complaints = df["complaint_type"].value_counts()
    to_remove = num_complaints[num_complaints <= 30].index
    df.replace(to_remove, np.nan, inplace=True)
    return df