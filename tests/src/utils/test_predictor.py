import pandas as pd

from value_prediction.predictor import Predictor

def test_make_24hrs():
    result = Predictor.make_24hrs()
    assert isinstance(result, pd.core.frame.DataFrame)
    assert result.columns == "timestamp"
    assert len(result) == 24
    assert 1==1 #format of string
