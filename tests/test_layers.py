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

    pred_y = model.predict()

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

        prediction, _model = make_and_run(Model, lookback=1, layers=lyrs, return_model=True)
        trues = {
            '21_posix': 1312.3688450753175,
            '115_posix': 1265.676495072539,
            '26_posix': 1312.3688450753175,
            '21_nt': 426.52874755859375,
            '23_nt': 286.2128473956241,
            '25_nt': 428.7085876464844,
            '26_nt': 286.21284595900113,
            '27_nt': 428.7085876464844,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1405.3555436921633), 3)
        return

    def test_LSTMModel(self):
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "LSTM_1": {'config': {'units': 32}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
                  }
        prediction, _model = make_and_run(Model, layers=lyrs, return_model=True)
        trues = {
            '21_posix': 1408.9016057021054,
            '115_posix': 1327.6904604995418,
            '26_posix': 1408.9016057021054,
            '21_nt': 299.75152587890625,
            '23_nt': 195.22941200342964,
            '25_nt': 312.6127014160156,
            '26_nt': 195.22941257807892,
            '27_nt': 312.6127014160156,
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

    def test_SeqWeightedAttention(self):
        # SeqWeightedAttention
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "SeqWeightedAttention": {"config": {}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
        }
        trues = {
            '21_nt': 317.4256286621094,
            '23_nt': 204.1259977159385,
            '25_nt': 313.1969909667969,
            '26_nt': 204.12599872157463,
            '27_nt': 313.1969909667969,
        }
        prediction, _model = make_and_run(Model, layers=lyrs, return_model=True)

        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1483.734244648907), 2)  # TODO failing with higher precision
        return

    def test_RaffelAttention(self):
        # LSTM  + Raffel Attention
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "LSTM_1": {'config': {'units': 32, 'return_sequences': True}},
            "AttentionRaffel": {'config': {'step_dim': 12}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
        }
        trues = {
            '21_posix': 1361.6870130712944,
            '26_posix': 1361.6870130712944,
            '115_posix':  1443.1860088206834,
            '21_nt': 290.931396484375,
            '23_nt': 205.64932981643847,
            '25_nt': 313.2801818847656,
            '26_nt': 205.64933154038607,
            '27_nt': 313.2801818847656,
        }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1353.11274522034), 4)
        return

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
                  }
        prediction = make_and_run(Model, layers=lyrs, save_model=False)
        trues = {
            '21_posix': 1327.5073743917194,
            '26_posix': 1327.5073743917194,
            '115_posix': 1430.7282310875908,
            '21_nt': 284.6064147949219,
            '23_nt': 197.95141420462951,
            '25_nt': 287.0279846191406,
            '26_nt': 197.95141865816086,
            '27_nt': 287.0279846191406,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 1356.0140036362777), 2)  # TODO failing with higher precision
        return

    def test_SelfAttention(self):
        # LSTM + SelfAttention model
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "SelfAttention": {'config': {}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
        }
        prediction, model = make_and_run(FModel, layers=lyrs, return_model=True)
        trues = {
            '21_posix_functional': 1522.6872986943176,
            '115_posix': 1549.689167207415,
            '26_posix_functional': 1549.689167207415,
            '21_nt_functional': 298.0132141113281,
            '27_nt_functional': 294.6787414550781,
            '25_nt': 131,
            '23_nt_functional': 182.70824960252654,
            '25_nt_functional': 294.6787414550781,
            '26_nt_functional': 182.7082507518249,
            #'27_nt_nt': 197.26560974121094,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(f'{PLATFORM}_{model.api}', 1409.7687666416527), 4)
        return

    def test_HierarchicalAttention(self):
        # LSTM + HierarchicalAttention model
        lyrs = {
            "LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
            "HierarchicalAttention": {'config': {}},
            "Dropout": {'config': {'rate': 0.3}},
            "Dense": {'config': {'units': outs, 'name': 'output'}},
        }
        trues = {
            '21_posix': 1376.5340244296872,
            '26_posix': 1376.5340244296872,
            '115_posix': 1376.5340244296872,
            '21_nt': 291.0807189941406,
            '23_nt': 197.72353275519052,
            '25_nt': 302.0905456542969,
            '26_nt': 197.72353304251516,
            '27_nt': 302.0905456542969,
        }
        prediction = make_and_run(Model, layers=lyrs)
        self.assertAlmostEqual(float(prediction.sum()),  trues.get(PLATFORM, 198.62783813476562), 3)
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
        prediction = make_and_run(Model, layers=lyrs)
        trues = {
            '21_posix': 1347.4325338505837,
            '26_posix': 1347.4325338505837,
            '115_posix': 1272.8471532368762,
            '21_nt': 311.875732421875,
            '23_nt': 188.44924334159703,
            '25_nt': 299.3736877441406,
            '26_nt': 188.449236,
            '27_nt': 299.3736877441406,
        }
        self.assertAlmostEqual(float(prediction.sum()), trues.get(PLATFORM, 193.57937622070312), 4)
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
        h = model.fit(data=df)
        return

if __name__ == "__main__":
    unittest.main()