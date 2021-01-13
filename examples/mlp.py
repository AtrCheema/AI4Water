# this file shows how to build a simple dense layer based model
# the input_features and outputs are columns and are present in the file

import pandas as pd
import os

from dl4seq import Model

if __name__ == "__main__":
    input_features = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
                  'input11']
    # column in dataframe to bse used as output/target
    outputs = ['target7']

    fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dl4seq/data/data_30min.csv")
    df = pd.read_csv(fname)
    df.index = pd.to_datetime(df['Date_Time2'])

    model = Model(data=df,
                  batch_size=16,
                  lookback=1,
                  inputs=input_features,
                  outputs=outputs,
                  lr=0.0001
                  )

    history = model.fit(indices='random')

    y, obs = model.predict(st=0, use_datetime_index=False, marker='.', linestyle='')
    model.view_model(st=0)
