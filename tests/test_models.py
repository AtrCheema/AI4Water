import pandas as pd
import os
import unittest
from sys import platform

import tensorflow as tf

import site  # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from inspect import getsourcefile
from os.path import abspath

from dl4seq import Model
from dl4seq import NBeatsModel

PLATFORM = ''.join(tf.__version__.split('.')[0:2]) + '_' + os.name

input_features = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
                  'input11']
ins = len(input_features)
outs = 1

file_path = abspath(getsourcefile(lambda:0))
dpath = os.path.join(os.path.join(os.path.dirname(os.path.dirname(file_path)), "dl4seq"), "data")

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
        # column in dataframe to bse used as output/target
        outputs = ['target7']
        kwargs['intervals'] = ((0, 146,),
                              (145, 386,),
                              (385, 628,),
                              (625, 821,),
                              (821, 1110),
                              (1110, 1447))

    model = model(
        data=df,
        verbosity=0,
        batch_size=batch_size,
        lookback=lookback,
        lr=0.001,
        inputs=inputs,
        outputs = outputs,
        epochs=epochs,
        model={'layers': kwargs.pop('layers')},
        **kwargs
    )

    _ = model.fit(indices='random')

    _,  pred_y = model.predict(use_datetime_index=False)

    return pred_y


class TestModels(unittest.TestCase):

    """
    Tests that most of the models run with some basic configuration.
    Also checks reproducibility in most ases
    """
    def test_mlp(self):

        lyrs = {
            "Dense_0": {'config': {'units': 64, 'activation': 'relu'}},
            "Dropout_0": {'config': {'rate': 0.3}},
            "Dense_1": {'config': {'units': 32, 'activation': 'relu'}},
            "Dropout_1": {'config': {'rate': 0.3}},
            "Dense_2": {'config': {'units': 16, 'activation': 'relu'}},
            "Dense_3": {'config': {'units': outs}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }

        prediction = make_and_run(Model, lookback=1, layers=lyrs)
        trues = {
            '21_posix': 1312.3688450753175,
            '115_posix': 1265.676495072539,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1289.3845434825826), 3)
        return

    def test_LSTMModel(self):
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "LSTM_1": {'config': {'units': 32}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
                  }
        prediction = make_and_run(Model, layers=lyrs)
        trues = {
            '21_posix': 1408.9016057021054,
            '115_posix': 1327.6904604995418,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1434.2028425805552), 3)
        return

    def test_SeqSelfAttention(self):
        # SeqSelfAttention
        batch_size=16
        lyrs = {
            "Input": {"config": {"batch_shape": (batch_size, 12, 8)}},
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "SeqSelfAttention": {"config": {"units": 32, "attention_width": 12, "attention_activation": "sigmoid"}},
            "LSTM_1": {'config': {'units': 32}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        if int(''.join(tf.__version__.split('.')[0:2])) <= 23:
            prediction = make_and_run(model=Model, layers=lyrs, batch_size=batch_size, batches_per_epoch=5)
            self.assertAlmostEqual(float(prediction.sum()), 474.3257980449273, 4)
        else:
            pass

    def test_SeqWeightedAttention(self):
        # SeqWeightedAttention
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "SeqWeightedAttention": {"config": {}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction.sum()), 1483.734244648907, 2)  # TODO failing with higher precision

    def test_RaffelAttention(self):
        # LSTM  + Raffel Attention
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "LSTM_1": {'config': {'units': 32, 'return_sequences': True}},
            "AttentionRaffel": {'config': {'step_dim': 12}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        trues = {
            '21_posix': 1361.6870130712944,
            '115_posix':  1443.1860088206834,
        }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1353.11274522034), 4)

    def test_SnailAttention(self):  # TODO failing to save model to h5 file on linux
        # LSTM  + Snail Attention
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True, "name": "first_lstm"}},
            "LSTM_1": {'config': {'units': 32, 'return_sequences': True, "name": "second_lstm"}},
            "SnailAttention": {'config': {'dims': 32, 'k_size': 32, 'v_size': 32, "name": "snail"}},
            "Dropout": {'config': {'rate': 0.3, "name": "FirstDropout"}},
            "Dense_0": {'config': {'units': 1, 'name': 'FirstDense'}},
            "Flatten": {'config': {"name": "FirstFlatten"}},
            "Dense": {'config': {'units': outs, "name": "prediction"}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
                  }
        prediction = make_and_run(Model, layers=lyrs, save_model=False)
        trues = {
            '21_posix': 1327.5073743917194,
            '115_posix': 1430.7282310875908,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1356.0140036362777), 2)  # TODO failing with higher precision

    def test_SelfAttention(self):
        # LSTM + SelfAttention model
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "SelfAttention": {'config': {}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        prediction = make_and_run(Model, layers=lyrs)
        trues = {
            '21_posix': 1522.6872986943176,
            '115_posix': 1549.689167207415,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1534.6699074814169), 4)

    def test_HierarchicalAttention(self):
        # LSTM + HierarchicalAttention model
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "HierarchicalAttention": {'config': {}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction.sum()), 1376.5340244296872, 3)

    def test_CNNModel(self):
        # CNN based model
        lyrs = {
            "Conv1D_9": {'config': {'filters': 64, 'kernel_size': 2}},
            "dropout": {'config': {'rate': 0.3}},
            "Conv1D_1": {'config': {'filters': 32, 'kernel_size': 2}},
            "maxpool1d": {'config': {'pool_size': 2}},
            'flatten': {'config': {}},
            'leakyrelu': {'config': {}},
            "Dense": {'config': {'units': outs}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        prediction = make_and_run(Model, layers=lyrs)
        trues = {
            '21_posix': 1347.4325338505837,
            '115_posix': 1272.8471532368762,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1347.498777542553), 4)

    def test_channel_attn(self):
        lyrs = {
            "Conv1D_9": {'config': {'filters': 8, 'kernel_size': 2}},
            "ChannelAttention": {'config': {'conv_dim': '1d', 'in_planes': 32}},
            'flatten': {'config': {}},
            'leakyrelu': {'config': {}},
            "Dense": {'config': {'units': outs}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        prediction = make_and_run(Model, layers=lyrs)
        trues = {
            '21_posix': 1548.395502996973,
            '115_posix': 673.8151633572088,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1559.654005092702), 4)
        return

    def test_spatial_attn(self):
        lyrs = {
            "Conv1D_9": {'config': {'filters': 8, 'kernel_size': 2}},
            "SpatialAttention": {'config': {'conv_dim': '1d'}},
            'flatten': {'config': {}},
            'leakyrelu': {'config': {}},
            "Dense": {'config': {'units': outs}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
                  }
        try:
            import tcn
            prediction = make_and_run(Model, layers=lyrs)
            trues = {
                '23_nt': 1459.3191080708255
            }
            self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 970.6771222840335), 2)  # TODO failing with higher precision
        except ModuleNotFoundError:
            print("tcn based model can not be tested as it is not found.")
    #
    # def test_NBeats(self):
    #     # NBeats based model
    #     lookback = 12
    #     exo_ins = 81
    #     forecsat_length = 4 # predict next 4 values
    #     layers = {
    #         "Input": {"config": {"shape": (lookback, 1), "name": "prev_inputs"}},
    #         "Input_Exo": {"config": {"shape": (lookback, exo_ins),"name": "exo_inputs"}},
    #         "NBeats": {"config": {"backcast_length":lookback,
    #                               "input_dim":1,
    #                               "exo_dim":exo_ins,
    #                               "forecast_length":forecsat_length,
    #                               "stack_types":('generic', 'generic'),
    #                               "nb_blocks_per_stack":2,
    #                               "thetas_dim":(4,4),
    #                               "share_weights_in_stack":True,
    #                               "hidden_layer_units":62},
    #                      "inputs": "prev_inputs",
    #                      "call_args": {"exo_inputs": "exo_inputs"}},
    #         "Flatten": {"config": {}},
    #         "Reshape": {"config": {"target_shape": (1, forecsat_length)}}
    #     }
    #
    #     predictions = make_and_run(NBeatsModel, layers=layers, forecast_length=forecsat_length, data_type="nasdaq")
    #     if platform.upper() in ["WIN32"]:
    #         self.assertAlmostEqual(float(predictions.sum()), 781079416.2836407, 4)  # 780862696.2410148
    #     else:
    #         self.assertGreater(float(predictions.sum()), 80000.0)  # TODO reproduction failing on linux


if __name__ == "__main__":
    unittest.main()