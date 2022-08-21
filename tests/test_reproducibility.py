
import unittest
import os

import numpy as np
import tensorflow as tf

from ai4water import Model
from ai4water.datasets import busan_beach
from ai4water.preprocessing import DataSet
from SeqMetrics import RegressionMetrics

PLATFORM = ''.join(tf.__version__.split('.')[0:2]) + '_' + os.name

data = busan_beach()
inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]

def get_model(model_name, **kwargs):
    model = Model(model=model_name,
                  verbosity=0,
                  **kwargs)

    model.fit(data=data)

    p = model.predict_on_test_data(data=data)

    return p


class TestML(unittest.TestCase):

    def test_xgboost(self):

        p = get_model("XGBRegressor")
        assert np.allclose(p.sum(), 85075350.0), p.sum()

        return

    def test_catboost(self):

        p = get_model("CatBoostRegressor")
        assert np.allclose(p.sum(), 118372625.26412539), p.sum()

        return

    def test_lgbm(self):
        p = get_model("LGBMRegressor")
        assert np.allclose(p.sum(), 276302360.69196635), p.sum()

        return

    def test_rf(self):

        p = get_model("RandomForestRegressor")
        assert np.allclose(p.sum(), 51240889.60922757), p.sum()

        return

    def test_transformation(self):

        p = get_model("RandomForestRegressor",
                      x_transformation="minmax",
                      y_transformation="log")
        assert np.allclose(p.sum(), 833541.2080729741), p.sum()

        return

    def test_transformation1(self):
        """reproducibility with complex transformations"""
        p = get_model("RandomForestRegressor",
                      x_transformation=[
            {
                "method": "sqrt",
                "features": [
                    "tide_cm"
                ],
                "treat_negatives": True
            },
            {
                "method": "zscore",
                "features": [
                    "wat_temp_c"
                ]
            },
            {
                "method": "robust",
                "features": [
                    "sal_psu"
                ]
            },
            {
                "method": "minmax",
                "features": [
                    "air_temp_c"
                ]
            },
            {
                "method": "zscore",
                "features": [
                    "pcp_mm"
                ]
            },
            {
                "method": "sqrt",
                "features": [
                    "pcp3_mm"
                ],
                "treat_negatives": True
            },
            {
                "method": "sqrt",
                "features": [
                    "pcp6_mm"
                ],
                "treat_negatives": True
            },
            {
                "method": "log2",
                "features": [
                    "pcp12_mm"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "robust",
                "features": [
                    "wind_dir_deg"
                ]
            },
            {
                "method": "minmax",
                "features": [
                    "wind_speed_mps"
                ]
            },
            {
                "method": "log2",
                "features": [
                    "air_p_hpa"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "log",
                "features": [
                    "rel_hum"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            }
        ],
                      y_transformation=[
            {
                "method": "log10",
                "features": outputs,
                "treat_negatives": True,
                "replace_zeros": True
            }
        ])
        assert np.allclose(p.sum(), 1017556.4220493606), p.sum()

        return

    def test_lstm_tf(self):
        # this test does not pass in tensorflow<2
        ds = DataSet(data, ts_args={'lookback': 14}, train_fraction=1.0)
        train_x, train_y = ds.training_data()
        val_x, val_y = ds.validation_data()

        _model = Model(model={"layers": {
            "LSTM": {"units": 34},
            "Activation": "tanh",
            "Dense": 1
        }},
            ts_args={"lookback": 14},
            input_features=data.columns.tolist()[0:-1],
            output_features=data.columns.tolist()[-1],
            train_fraction=1.0,
            epochs=100,
            verbosity=0
        )

        _model.seed_everything(313)

        # train model
        _model.fit(x=train_x, y=train_y)

        # evaluate model
        p = _model.predict(val_x)
        val_score = RegressionMetrics(val_y, p).mse()
        trues = {
            '21_posix': 60611115352064.0,
            '115_posix': 60611115352064.0,
            '26_posix': 0,
            '21_nt': 60611115352064.0,
            '23_nt': 60611115352064.0,
            '25_nt': 60611144712192.0,
            '26_nt': 60611115352064.0,
            '27_nt': 60611144712192.0,
        }

        self.assertAlmostEqual(val_score, trues[PLATFORM], 1)
        return



if __name__ == "__main__":
    unittest.main()