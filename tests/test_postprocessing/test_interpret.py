import time
import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.datasets import busan_beach
from ai4water.postprocessing import Interpret
from ai4water import InputAttentionModel, DualAttentionModel

default_model = {
    'layers': {
        "Dense_0": {'units': 64, 'activation': 'relu'},
        "Flatten": {},
        "Dense_3": 1,
    }
}


examples = 2000
ins = 5
outs = 1
in_cols = ['input_'+str(i) for i in range(5)]
out_cols = ['output']
cols=  in_cols + out_cols
data1 = pd.DataFrame(np.arange(int(examples*len(cols))).reshape(-1, examples).transpose(),
                    columns=cols,
                    index=pd.date_range("20110101", periods=examples,
                                        freq="D"))


beach_data = busan_beach()
input_features=beach_data.columns.tolist()[0:-1]
output_features=beach_data.columns.tolist()[-1:]


def build_model(**kwargs):

    model = Model(
        verbosity=0,
        batch_size=16,
        epochs=1,
        **kwargs
    )

    return model


def da_lstm_model(data,
                  **kwargs):

    model = DualAttentionModel(
        verbosity=0,
        **kwargs
    )

    _ = model.fit(data=data)

    return model


def ia_lstm_model(**kwargs):

    model = InputAttentionModel(verbosity=0, **kwargs)

    _ = model.fit(data=beach_data)

    return model


def lstm_with_SelfAtten(**kwargs):
    return




class TestInterpret(unittest.TestCase):

    def test_plot_feature_importance(self):

        time.sleep(1)
        model = build_model(
            ts_args={'lookback':7},
            input_features=in_cols,
            output_features=out_cols,
            model=default_model
        )

        model.fit(data=data1.astype(np.float32))

        Interpret(model, show=False).plot_feature_importance(
            np.random.randint(1, 10, 5))

        return

    def test_da_interpret(self):
        m = da_lstm_model(data='CAMELS_AUS',
                          ts_args={'lookback': 14},
                          input_features = ['precipitation_AWAP',
                          'evap_pan_SILO'],
        output_features = ['streamflow_MLd_inclInfilled'],
        intervals =  [("20000101", "20011231")],
        dataset_args = {'stations': 1},)
        m.interpret(data='CAMELS_AUS', show=False)
        return

    def test_da_without_prevy_interpret(self):
        # todo, only working in tensorflow 1
        m = da_lstm_model(teacher_forcing=False, drop_remainder=True,
                          data=beach_data,
                          input_features=input_features,
                          batch_size=8,
                          ts_args={'lookback':14},
                          output_features=output_features)
        x,y = m.training_data()

        m.interpret(data_type='training', data=beach_data, show=False)
        return

    def test_ia_interpret(self):
        m = ia_lstm_model(input_features=input_features,
                          ts_args={'lookback': 14},
                          output_features=output_features)
        m.interpret(data=beach_data, show=False)
        m.plot_act_along_inputs(data=beach_data, feature="pcp_mm")
        return

    def test_ml(self):
        time.sleep(1)
        for m in ['XGBRegressor', 'RandomForestRegressor',
                  'GradientBoostingRegressor', 'LinearRegression']:

            model = build_model(model=m)
            model.fit(data=beach_data)
            model.interpret(show=False)
        return

    def test_xgb_f_imp_comparison(self):
        model = Model(model="XGBRegressor")
        model.fit(data=busan_beach(inputs=["tide_cm", "rel_hum"]))
        interpreter = Interpret(model, show=False)
        interpreter.compare_xgb_f_imp()
        return


if __name__ == "__main__":
    unittest.main()