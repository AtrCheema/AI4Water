from utils import make_model
from models import NBeatsModel, DualAttentionModel

import pandas as pd


def make_and_run(input_model, _layers=None, lookback=12, epochs=4, **kwargs):


    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=lookback,
                                                         lr=0.001,
                                                         epochs=epochs,
                                                         **kwargs)
    nn_config['layers'] = _layers

    df = pd.read_csv("../data/nasdaq100_padding.csv")

    _model = input_model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  intervals=total_intervals
                  )

    _model.build_nn()

    _ = _model.train_nn(indices='random')

    _ = _model.predict(use_datetime_index=False)

    return _model

##
# NBeats based model
model = make_and_run(NBeatsModel)

##
# DualAttentionModel based model
make_and_run(DualAttentionModel)
