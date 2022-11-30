import datetime
import pandas as pd

from src.predictor.startnet import StartNet
from src.utils.predictor import make_24hrs, get_features

def test_make_24hrs():
    result = make_24hrs()
    assert isinstance(result, pd.core.frame.DataFrame)
    assert result.columns == "timestamp"
    assert len(result) == 25

def test_get_features():
    input = [datetime.datetime(2022, 11, 29, 13, 5, 17, 820026), datetime.datetime(2022, 11, 29, 14, 5, 17, 820026)]
    data_parsed = pd.DataFrame.from_dict(input)      
    data_parsed.columns = ["timestamp"]
    result = get_features(data_parsed)
    assert "hour" in result.columns.values
    assert "day_of_year" in result.columns.values
    assert "weekday" in result.columns.values
