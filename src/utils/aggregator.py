import pandas as pd

def aggregate(data: pd.DataFrame, base: str="hours") -> pd.DataFrame:
    """ Aggregates temperature and humidity """
    if base == "hours":
        data["date"] = data["timestamp"].dt.date
        data["hour"] = data["timestamp"].dt.hour
        agg_data = data.groupby([data.date, data.hour])["temperature", "humidity"].apply(lambda x : round(x.astype(float).mean(), 2))
        agg_data = agg_data.reset_index()
        # Adding back expected columns
        agg_data["timestamp"] = agg_data["date"].astype(str) + " " + agg_data["hour"].astype(str) + ":00:00"
        agg_data['remove'] = 'TO_REMOVE'
        agg_data.drop(["date", "hour"], axis=1, inplace=True)
        return agg_data
    else:
        print("not implemented")
        return data
