
import os
import site  # so that AI4Water directory is in path
import unittest
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import numpy as np
import tensorflow as tf

from AI4Water import Model
from AI4Water.functional import Model as FModel
from AI4Water import NBeatsModel
from AI4Water.utils.datasets import arg_beach, load_nasdaq

PLATFORM = ''.join(tf.__version__.split('.')[0:2]) + '_' + os.name

#tf.compat.v1.disable_eager_execution()

input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                  'rel_hum']
ins = len(input_features)
outs = 1


def make_and_run(
        model,
        lookback=12,
        epochs=4,
        batch_size=16,
        data_type='other',
        return_model=False,
        **kwargs):

    if data_type == "nasdaq":
        df = load_nasdaq()
        in_cols = list(df.columns)[0:-1]
        inputs = in_cols
        outputs = ["NDX"]
    else:
        outputs = ['blaTEM_coppml']
        df = arg_beach(input_features, target=outputs)
        inputs = input_features
        # column in dataframe to bse used as output/target
        df['blaTEM_coppml'] = np.log10(df['blaTEM_coppml'])
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
        input_features=inputs,
        output_features = outputs,
        epochs=epochs,
        model={'layers': kwargs.pop('layers')},
        **kwargs
    )

    _ = model.fit(indices='random')

    _,  pred_y = model.predict(use_datetime_index=False)

    if return_model:
        return pred_y, model

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
            '25_nt': 158.6142,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1405.3555436921633), 3)
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
            '25_nt': 131.04134,
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
            self.assertAlmostEqual(float(prediction.sum()), 519.6375288994583, 4)
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
        trues = {
            '25_nt': 77.85861968
        }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1483.734244648907), 2)  # TODO failing with higher precision

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
            '25_nt': 118.26399,
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
            '25_nt': 66.52947,
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
        prediction, model = make_and_run(FModel, layers=lyrs, return_model=True)
        trues = {
            '21_posix': 1522.6872986943176,
            '115_posix': 1549.689167207415,
            '21_nt_functional': 165.6065673828125,
            '25_nt': 131,
            '25_nt_functional': 222.57863,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(f'{PLATFORM}_{model.api}', 1409.7687666416527), 4)

    def test_HierarchicalAttention(self):
        # LSTM + HierarchicalAttention model
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "HierarchicalAttention": {'config': {}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
            "Reshape": {"config": {"target_shape": (outs, 1)}}
        }
        trues = {
            '21_posix': 1376.5340244296872,
            '115_posix': 1376.5340244296872,
            '25_nt': 80.872604,
        }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction.sum()),  trues.get(PLATFORM, 1376.5340244296872), 3)

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
            '25_nt': 163.878662109375,
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
            '25_nt': 63.53640365600586,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1475.2777905818857), 4)
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
                '23_nt': 1372.9329818539659,
                '25_nt': 1.849501132965088,
            }
            self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 970.6771222840335), 2)  # TODO failing with higher precision
        except ModuleNotFoundError:
            print("tcn based model can not be tested as it is not found.")

    def test_NBeats(self):
        # NBeats based model
        lookback = 12
        exo_ins = 81
        forecsat_length = 4 # predict next 4 values
        layers = {
            "Input": {"config": {"shape": (lookback, 1), "name": "prev_inputs"}},
            "Input_Exo": {"config": {"shape": (lookback, exo_ins),"name": "exo_inputs"}},
            "NBeats": {"config": {"backcast_length":lookback,
                                  "input_dim":1,
                                  "exo_dim":exo_ins,
                                  "forecast_length":forecsat_length,
                                  "stack_types":('generic', 'generic'),
                                  "nb_blocks_per_stack":2,
                                  "thetas_dim":(4,4),
                                  "share_weights_in_stack":True,
                                  "hidden_layer_units":62},
                         "inputs": "prev_inputs",
                         "call_args": {"exo_inputs": "exo_inputs"}},
            "Flatten": {"config": {}},
            "Reshape": {"config": {"target_shape": (1, forecsat_length)}}
        }

        predictions = make_and_run(NBeatsModel, layers=layers, forecast_length=forecsat_length, data_type="nasdaq",
                                   transformation=None)

        trues = {
            '25_nt': 780467456
        }

        if PLATFORM.upper() in ["WIN32"]:
            self.assertAlmostEqual(float(predictions.sum()), 781079416.2836407, 4)  # 780862696.2410148
        else:
            self.assertAlmostEqual(float(predictions.sum()), trues.get(PLATFORM, 80000.0))  # TODO reproduction failing on linux


if __name__ == "__main__":
    unittest.main()