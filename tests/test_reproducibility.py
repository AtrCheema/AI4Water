
import unittest

import numpy as np

from ai4water import Model
from ai4water.datasets import busan_beach

data = busan_beach()
inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]

def get_model(model_name, **kwargs):
    model = Model(model=model_name,
                  verbosity=0,
                  **kwargs)

    model.fit(data=data)

    p = model.predict()

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
        assert np.allclose(p.sum(), 19376295.67998046), p.sum()

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
        assert np.allclose(p.sum(), 5953105.679834714), p.sum()

        return



if __name__ == "__main__":
    unittest.main()