import os
import unittest



import site  # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq import Model

import numpy as np
import pandas as pd

examples = 200
lookback = 10
inp_1d = ['prec', 'temp']
inp_2d = ['lai', 'etp']
w = 5
h = 5

# input for 2D data, radar images
pcp = np.random.randint(0, 5, examples*w*h).reshape(examples, w, h)  # 5 rows 5 columns, and 100 images
lai = np.random.randint(6, 10, examples*w*h).reshape(examples, w, h)

data_2d = np.full((len(pcp), len(inp_2d), w, h), np.nan) #np.shape(data)==(100,5,5,5); 5 rows, 5 columns, 5 variables, and 100 images in each variable

##output
flow = np.full((len(pcp), 1), np.nan)  # np.shape(flow)==(100,1)

for i in range(len(flow)): #len(flow)==100
    flow[i] = pcp[i, :].sum() + lai[i, :].sum()
    data_2d[i, :] = np.stack([pcp[i], lai[i]])

def make_1d(outputs):
    data_1d = np.random.random((examples, len(inp_1d) + len(outputs)))
    # output of model
    if len(outputs) == 1:
        data_1d[:, -1] = np.add(np.add(np.add(data_1d[:, 0], data_1d[:, 1]), data_1d[:, 2]), flow.reshape(-1,))
    else:
        data_1d[:, -1] = np.add(np.add(np.add(data_1d[:, 0], data_1d[:, 1]), data_1d[:, 2]), flow.reshape(-1, ))
        data_1d[:, -2] = np.multiply(np.add(np.add(data_1d[:, 0], data_1d[:, 1]), data_1d[:, 2]), flow.reshape(-1, ))
    data_1d = pd.DataFrame(data_1d, columns=inp_1d+ outputs)
    return data_1d

def make_layers(outs):
    layers = {
        "Input_1d": {"config": {"shape": (lookback, len(inp_1d)), "name": "inp_1d"}},
        "LSTM_1d": {"config": {"units": 12}},

        "Input_2d": {"config": {"shape": (lookback, len(inp_2d), w, h), "name": "inp_2d"}},
        "ConvLSTM2d": {"config": {"filters": 32, "kernel_size": (3,3), "data_format": "channels_first"}},
        "Flatten": {"config": {},
                    "inputs": "ConvLSTM2d"},
        "Concat": {"config": {},
                   "inputs": ["LSTM_1d", "Flatten"]},
        "Dense": {"config": {"units": outs}},
        "Reshape": {"config": {"target_shape": (outs, 1)}}
    }
    return layers

def build_and_run(outputs, transformation=None, indices=None):
    model = Model(
        model={"layers": make_layers(len(outputs['inp_1d']))},
        lookback=lookback,
        inputs = {"inp_1d": inp_1d, "inp_2d": inp_2d},
        outputs=outputs,
        data={'inp_1d': make_1d(outputs['inp_1d']), 'inp_2d': data_2d},
        transformation = transformation,
        epochs=2,
        verbosity=0
    )

    model.fit(indices=indices)
    return model.predict(indices=model.test_indices if indices else None)

class test_MultiInputModels(unittest.TestCase):

    def test_with_no_transformation(self):
        outputs = {"inp_1d": ['flow']}
        build_and_run(outputs, transformation=None)
        return

    def test_with_transformation(self):
        outputs = {"inp_1d": ['flow']}
        transformation={'inp_1d': 'minmax', 'inp_2d': None}
        build_and_run(outputs, transformation=transformation)
        return

    def test_with_random_sampling(self):
        outputs = {"inp_1d": ['flow']}
        build_and_run(outputs, indices="random")
        return

    def test_with_multiple_outputs(self):
        outputs = {'inp_1d': ['flow', 'suro']}
        build_and_run(outputs)
        return

    def test_add_output_layer1(self):
        # check if it adds both dense and reshapes it correctly or not
        model = Model(model={'layers': {'lstm': {'config': {'units': 64}}}})

        self.assertEqual(model._model.outputs[0].shape[1], model.outs)
        self.assertEqual(model._model.outputs[0].shape[-1], model.forecast_len)

        return

    def test_add_output_layer2(self):
        # check if it reshapes the output correctly
        model = Model(model={'layers': {'lstm': {'config': {'units': 64}},
                                        'Dense': {'config': {'units': 1}}}})

        self.assertEqual(model._model.outputs[0].shape[1], model.outs)
        self.assertEqual(model._model.outputs[0].shape[-1], model.forecast_len)
        return

    def test_add_no_output_layer(self):
        # check if it does not add layers when it does not have to
        model = Model(model={'layers': {'lstm': {'config': {'units': 64}},
                                        'Dense': {'config': {'units': 1}},
                                        'Reshape': {'config': {'target_shape': (1,1)}}}})

        self.assertEqual(model._model.outputs[0].shape[1], model.outs)
        self.assertEqual(model._model.outputs[0].shape[-1], model.forecast_len)
        return

if __name__ == "__main__":
    unittest.main()