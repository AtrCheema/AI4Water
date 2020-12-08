import pandas as pd
import os

def load_30min():
    fpath = os.path.join(os.path.dirname(__file__), "data_30min.csv")
    df = pd.read_csv(fpath, index_col="Date_Time2")
    df.index=pd.to_datetime(df.index)
    return df

def load_u1():
    """loads 1d data that can be used fo regression and classification"""
    fpath = os.path.join(os.path.dirname(__file__), "input_target_u1.csv")
    df = pd.read_csv(fpath)
    return df