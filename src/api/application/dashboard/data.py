
"""Prepare data for Plotly Dash."""
import os

import numpy as np
import pandas as pd

def create_dataframe(path):
    """Create Pandas DataFrame from local CSV."""
    df = pd.read_csv(path, parse_dates=["created"])
    df["created"] = df["created"].dt.date
    df.drop(columns=["incident_zip"], inplace=True)
    num_complaints = df["complaint_type"].value_counts()
    to_remove = num_complaints[num_complaints <= 30].index
    df.replace(to_remove, np.nan, inplace=True)
    return df