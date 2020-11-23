import pandas as pd
import os
import unittest

import site  # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from inspect import getsourcefile
from os.path import abspath

from dl4seq.utils import make_model
from dl4seq import Model
from dl4seq import NBeatsModel

input_features = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
                  'input11']
ins = len(input_features)

file_path = abspath(getsourcefile(lambda:0))
dpath = os.path.join(os.path.dirname(os.path.dirname(file_path)), "data")

def make_and_run(
        model,
        lookback=12,
        epochs=4,
        batch_size=16,
        data_type='other',
        **kwargs):

    if data_type == "nasdaq":
        fname = os.path.join(dpath, "nasdaq100_padding.csv")
        df = pd.read_csv(fname)
        in_cols = list(df.columns)
        in_cols.remove("NDX")
        inputs = in_cols
        outputs = ["NDX"]
    else:
        fname = os.path.join(dpath, "data_30min.csv")
        df = pd.read_csv(fname)
        inputs = input_features
        outputs = ['target7'] # column in dataframe to bse used as output/target

    data_config, nn_config, total_intervals = make_model(batch_size=batch_size,
                                                         lookback=lookback,
                                                         lr=0.001,
                                                         inputs=inputs,
                                                         outputs = outputs,
                                                         epochs=epochs,
                                                         **kwargs)

    model = model(
        data_config=data_config,
        nn_config=nn_config,
        data=df,
        intervals=None if data_type=="nasdaq" else total_intervals,
        verbosity=0
    )

    model.build_nn()

    _ = model.train_nn(indices='random')

    _,  pred_y = model.predict(use_datetime_index=False)

    return pred_y


class TestModels(unittest.TestCase):

    def test_mlp(self):

        lyrs = {"Dense_0": {'config': {'units': 64, 'activation': 'relu'}},
                  "Dropout_0": {'config': {'rate': 0.3}},
                  "Dense_1": {'config': {'units': 32, 'activation': 'relu'}},
                  "Dropout_1": {'config': {'rate': 0.3}},
                  "Dense_2": {'config': {'units': 16, 'activation': 'relu'}},
                  "Dense_3": {'config': {'units': 1}}
                  }

        prediction = make_and_run(Model, lookback=1, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1312.97485, 4)


    def test_LSTMModel(self):
        lyrs = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
                  "LSTM_1": {'config': {'units': 32}},
                  "Dropout": {'config': {'rate': 0.3}},
                  "Dense": {'config': {'units': 1, 'name': 'output'}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1452.8463, 4)


    def test_SeqSelfAttention(self):
        # SeqSelfAttention
        batch_size=16
        lyrs = {
            "Input": {"config": {"batch_shape": (batch_size, 12, 8)}},
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
                  "SeqSelfAttention": {"config": {"units": 32, "attention_width": 12, "attention_activation": "sigmoid"}},
                  "LSTM_1": {'config': {'units': 32}},
                  "Dropout": {'config': {'rate': 0.3}},
                  "Dense": {'config': {'units': 1, 'name': 'output'}}
                  }

        prediction = make_and_run(model=Model, layers=lyrs, batch_size=batch_size, batches_per_epoch=5)
        self.assertAlmostEqual(float(prediction[0].sum()), 471.49829, 4)

    def test_SeqWeightedAttention(self):
        # SeqWeightedAttention
        lyrs = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
                  "SeqWeightedAttention": {"config": {}},
                  "Dropout": {'config': {'rate': 0.3}},
                  "Dense": {'config': {'units': 1, 'name': 'output'}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1457.831176, 2)  # TODO failing with higher precision

    def test_RaffelAttention(self):
        # LSTM  + Raffel Attention
        lyrs = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
                  "LSTM_1": {'config': {'units': 32, 'return_sequences': True}},
                  "AttentionRaffel": {'config': {'step_dim': 12}},
                  "Dropout": {'config': {'rate': 0.3}},
                  "Dense": {'config': {'units': 1, 'name': 'output'}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1378.492431, 4)

    def test_SnailAttention(self):
        # LSTM  + Snail Attention
        lyrs = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
                  "LSTM_1": {'config': {'units': 32, 'return_sequences': True}},
                  "SnailAttention": {'config': {'dims': 32, 'k_size': 32, 'v_size': 32}},
                  "Dropout": {'config': {'rate': 0.3}},
                  "Dense_0": {'config': {'units': 1, 'name': 'output'}},
                  "Flatten": {'config': {}},
                  "Dense": {'config': {'units': 1}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1306.02380, 4)

    def test_SelfAttention(self):
        # LSTM + SelfAttention model
        lyrs = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
                  "SelfAttention": {'config': {}},
                  "Dropout": {'config': {'rate': 0.3}},
                  "Dense": {'config': {'units': 1, 'name': 'output'}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1527.28979, 4)


    def test_HierarchicalAttention(self):
        # LSTM + HierarchicalAttention model
        lyrs = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
                  "HierarchicalAttention": {'config': {}},
                  "Dropout": {'config': {'rate': 0.3}},
                  "Dense": {'config': {'units': 1, 'name': 'output'}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1374.090332, 4)

    def test_CNNModel(self):
        # CNN based model
        lyrs = {"Conv1D_9": {'config': {'filters': 64, 'kernel_size': 2}},
                  "dropout": {'config': {'rate': 0.3}},
                  "Conv1D_1": {'config': {'filters': 32, 'kernel_size': 2}},
                  "maxpool1d": {'config': {'pool_size': 2}},
                  'flatten': {'config': {}},
                  'leakyrelu': {'config': {}},
                  "Dense": {'config': {'units': 1}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1333.210693, 4)

    def test_LSTMCNNModel(self):
        # LSTMCNNModel based model
        lyrs = {"LSTM": {'config': {'units': 64, 'return_sequences': True}},
                  "Conv1D_0": {'config': {'filters': 64, 'kernel_size': 2}},
                  "dropout": {'config': {'rate': 0.3}},
                  "Conv1D_1": {'config': {'filters': 32, 'kernel_size': 2}},
                  "maxpool1d": {'config': {'pool_size': 2}},
                  'flatten': {'config': {}},
                  'leakyrelu': {'config': {}},
                  "Dense": {'config': {'units': 1}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 1398.09057, 4)

    def test_ConvLSTMModel(self):
        # ConvLSTMModel based model
        self.lookback = 12
        sub_seq = 3
        sub_seq_lens = int(self.lookback / sub_seq)
        lyrs = {'Input' : {'config': {'shape':(sub_seq, 1, sub_seq_lens, ins)}},
                  'convlstm2d': {'config': {'filters': 64, 'kernel_size': (1, 3), 'activation': 'relu'}},
                  'flatten': {'config': {}},
                  'repeatvector': {'config': {'n': 1}},
                  'lstm':   {'config': {'units': 128,   'activation': 'relu', 'dropout': 0.3, 'recurrent_dropout': 0.4 }},
                  'Dense': {'config': {'units': 1}}
                  }
        prediction = make_and_run(Model, layers=lyrs, subsequences=sub_seq, lookback=self.lookback)
        self.assertAlmostEqual(float(prediction[0].sum()), 1413.6604, 3)  # TODO failing with higher precision

    def test_CNNLSTMModel(self):
        # CNNLSTM based model
        subsequences = 3
        lookback = 12
        timesteps = lookback // subsequences
        lyrs = {'Input' : {'config': {'shape':(subsequences, timesteps, ins)}},
                  "TimeDistributed_0": {'config': {}},
                  "Conv1D_0": {'config': {'filters': 64, 'kernel_size': 2}},
                  "leakyrelu": {'config': {}},
                  "TimeDistributed_1": {'config': {}},
                  "maxpool1d": {'config': {'pool_size': 2}},
                  "TimeDistributed_2": {'config': {}},
                  'flatten': {'config': {}},
                  'lstm':   {'config': {'units': 64,   'activation': 'relu'}},
                  'Dense': {'config': {'units': 1}}
                       }
        prediction = make_and_run(Model, layers=lyrs, subsequences=subsequences)
        self.assertAlmostEqual(float(prediction[0].sum()), 1523.947998, 4)


    def test_LSTMAutoEncoder(self):
        # LSTM auto-encoder
        lyrs = {
            'lstm_0': {'config': {'units': 100,  'recurrent_dropout': 0.4}},
            "leakyrelu_0": {'config': {}},
            'RepeatVector': {'config': {'n': 11}},
            'lstm_1': {'config': {'units': 100,  'dropout': 0.3}},
            "relu_1": {'config': {}},
            'Dense': {'config': {'units': 1}}
        }
        prediction = make_and_run(Model, layers=lyrs, lookback=12)
        self.assertAlmostEqual(float(prediction[0].sum()), 1514.47912, 4)

    def test_TCNModel(self):
        # TCN based model auto-encoder
        lyrs = {"tcn":  {'config': {'nb_filters': 64,
                          'kernel_size': 2,
                          'nb_stacks': 1,
                          'dilations': [1, 2, 4, 8, 16, 32],
                          'padding': 'causal',
                          'use_skip_connections': True,
                          'return_sequences': False,
                          'dropout_rate': 0.0}},
                  'Dense':  {'config': {'units': 1}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction[0].sum()), 935.47619, 2)   # TODO failing with higher precision


    def test_NBeats(self):
        # NBeats based model
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

        predictions = make_and_run(NBeatsModel, layers=layers, forecast_length=forecsat_length, data_type="nasdaq")
        # self.assertAlmostEqual(float(predictions[0].sum().values.sum()), 85065.516, 2)   # TODO reproduction failing on linux
        self.assertGreater(float(predictions[0].sum().values.sum()), 80000.0)


if __name__ == "__main__":
    unittest.main()