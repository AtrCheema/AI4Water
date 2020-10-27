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
                         #intervals=total_intervals
                         )

    _model.build_nn()

    _ = _model.train_nn(indices='random')

    _ = _model.predict(use_datetime_index=False)

    return _model

lookback = 12
exo_ins = 81
layers = {
    "Input": {"config": {"shape": (lookback, 1), "name": "prev_inputs"}},
    "Input_Exo": {"config": {"shape": (lookback, exo_ins),"name": "exo_inputs"}},
    "NBeats": {"config": {"backcast_length":lookback, "input_dim":1, "exo_dim":81, "forecast_length":1,
                            "stack_types":('generic', 'generic'), "nb_blocks_per_stack":2, "thetas_dim":(4,4),
                            "share_weights_in_stack":True, "hidden_layer_units":62},
                 "inputs": "prev_inputs",
                 "call_args": {"exo_inputs": "exo_inputs"}},
    "Flatten": {"config": {}},
}
##
# NBeats based model
model = make_and_run(NBeatsModel, _layers=layers, lookback=lookback)

##
# DualAttentionModel based model
make_and_run(DualAttentionModel)
