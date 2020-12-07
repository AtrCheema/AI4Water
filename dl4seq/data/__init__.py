import pandas as pd


def load_30min():
    df = pd.read_csv("data_30min.csv", index_col="Date_Time2")
    df.index=pd.to_datetime(df.index)
    return df
