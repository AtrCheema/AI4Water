from utils import make_model
from models import NBeatsModel, DualAttentionModel

import pandas as pd
import numpy as np


def make_and_run(input_model, _layers=None, lookback=12, epochs=4, **kwargs):

    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=lookback,
                                                         lr=0.001,
                                                         epochs=epochs,
                                                         **kwargs)
    nn_config['layers'] = _layers

    df = pd.read_csv("../data/nasdaq100_padding.csv")

    model = input_model(data_config=data_config,
                         nn_config=nn_config,
                         data=df,
                         #intervals=total_intervals
                         )

    model.build_nn()

    _ = model.train_nn(indices='random')

    _, pred_y = model.predict(use_datetime_index=False)

    return pred_y

lookback = 12
exo_ins = 81
forecsat_length = 4 # predict next 4 values
layers = {
    "Input": {"config": {"shape": (lookback, 1), "name": "prev_inputs"}},
    "Input_Exo": {"config": {"shape": (lookback, exo_ins),"name": "exo_inputs"}},
    "NBeats": {"config": {"backcast_length":lookback, "input_dim":1, "exo_dim":exo_ins, "forecast_length":forecsat_length,
                            "stack_types":('generic', 'generic'), "nb_blocks_per_stack":2, "thetas_dim":(4,4),
                            "share_weights_in_stack":True, "hidden_layer_units":62},
                 "inputs": "prev_inputs",
                 "call_args": {"exo_inputs": "exo_inputs"}},
    "Flatten": {"config": {}},
}
##
# NBeats based model
predictions = make_and_run(NBeatsModel, _layers=layers, lookback=lookback, forecast_length=forecsat_length)
np.testing.assert_almost_equal(float(predictions[0].sum().values.sum()), 85065.516, decimal=3)

##
# DualAttentionModel based model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
preds = make_and_run(DualAttentionModel)
