import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import numpy as np
import tensorflow as tf

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.functional import Model as FModel
from ai4water.datasets import busan_beach

PLATFORM = ''.join(tf.__version__.split('.')[0:2]) + '_' + os.name

#tf.compat.v1.disable_eager_execution()

input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                  'rel_hum']
ins = len(input_features)
outs = 1

class CustomLayer(tf.keras.layers.Dense):
    pass

def make_and_run(
        model,
        lookback=12,
        epochs=4,
        batch_size=16,
        return_model=False,
        **kwargs):

    outputs = ['blaTEM_coppml']
    df = busan_beach(input_features, target=outputs)
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
        verbosity=0,
        batch_size=batch_size,
        ts_args={'lookback':lookback},
        lr=0.001,
        input_features=inputs,
        output_features = outputs,
        epochs=epochs,
        model={'layers': kwargs.pop('layers')},
        split_random=True,
        x_transformation='minmax',
        y_transformation="minmax",
        **kwargs
    )

    _ = model.fit(data=df)

    pred_y = model.predict(process_results=False)

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
        }

        make_and_run(Model, lookback=1, layers=lyrs, return_model=True)

        return

    def test_LSTMModel(self):
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "LSTM_1": {'config': {'units': 32}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
                  }
        make_and_run(Model, layers=lyrs, return_model=True)

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
        }
        trues = {
            '21_nt': 204.87033081054688,
            '23_nt': 108.09015740001126,
            '27_nt': 195.22941257807892,
        }
        if int(''.join(tf.__version__.split('.')[0:2])) <= 23:
            prediction = make_and_run(model=Model, layers=lyrs, batch_size=batch_size,
                                      batches_per_epoch=5, drop_remainder=True)
            self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 519.6375288994583), 4)
        else:
            pass


    def test_SelfAttention(self):
        # LSTM + SelfAttention model
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "SelfAttention": {'config': {'return_attention_weights': False}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
        }
        make_and_run(FModel, layers=lyrs, return_model=True)

        return

    def test_CNNModel(self):
        # CNN based model
        lyrs = {
            "Conv1D_9": {'config': {'filters': 64, 'kernel_size': 2}},
            "Dropout": {'config': {'rate': 0.3}},
            "Conv1D_1": {'config': {'filters': 32, 'kernel_size': 2}},
            "MaxPool1D": {'config': {'pool_size': 2}},
            'Flatten': {'config': {}},
            'LeakyReLU': {'config': {}},
            "Dense": {'config': {'units': outs}},
        }
        make_and_run(Model, layers=lyrs)

        return

    def test_channel_attn(self):
        lyrs = {
            "Conv1D_9": {'config': {'filters': 8, 'kernel_size': 2}},
            "ChannelAttention": {'config': {'conv_dim': '1d', 'in_planes': 32}},
            'Flatten': {'config': {}},
            'LeakyReLU': {'config': {}},
            "Dense": {'config': {'units': outs}},
        }
        prediction = make_and_run(Model, layers=lyrs)
        trues = {
            '21_posix': 1548.395502996973,
            '26_posix': 1548.395502996973,
            '115_posix': 673.8151633572088,
            '21_nt': 311.81280517578125,
            '23_nt': 192.28048198313545,
            '25_nt': 310.38372802734375,
            '26_nt': 192.28047997186326,
            '27_nt': 310.38372802734375,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1475.2777905818857), 4)
        return

    def test_spatial_attn(self):
        lyrs = {
            "Conv1D_9": {'config': {'filters': 8, 'kernel_size': 2}},
            "SpatialAttention": {'config': {'conv_dim': '1d'}},
            'Flatten': {'config': {}},
            'LeakyReLU': {'config': {}},
            "Dense": {'config': {'units': outs}},
                  }
        try:
            import tcn
            prediction = make_and_run(Model, layers=lyrs)
            trues = {
                '21_nt': 303.4326477050781,
                '23_nt': 197.4112326089435,
                '25_nt': 305.4326171875,
                '27_nt': 305.4326171875,
            }
            self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 970.6771222840335), 2)  # TODO failing with higher precision
        except ModuleNotFoundError:
            print("tcn based model can not be tested as it is not found.")
        return

    def test_custom_layer(self):

        outputs = ['blaTEM_coppml']
        df = busan_beach(input_features, target=outputs)

        lyrs = {
            "Input": {"shape": (ins, )},
            CustomLayer: 1
        }

        model = FModel(model={"layers": lyrs},
                       epochs=1,
                       verbosity=0)
        model.fit(data=df)
        return

if __name__ == "__main__":
    unittest.main()