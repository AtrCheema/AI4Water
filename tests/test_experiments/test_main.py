
import os
import site
import unittest

import pandas as pd

ai4_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(ai4_dir)

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from ai4water.datasets import busan_beach
from ai4water.utils.utils import dateandtime_now
from ai4water.experiments import MLRegressionExperiments


input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm',
                  'pcp3_mm', 'pcp12_mm', 'air_p_hpa']
# column in dataframe to bse used as output/target
outputs = ['blaTEM_coppml']

df = busan_beach(input_features, outputs)


class TestExperiments(unittest.TestCase):

    def test_build_predict_from_configs(self):
        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            exp_name=f"BestMLModels_{dateandtime_now()}",
            verbosity=0
        )
        include = ['GaussianProcessRegressor', 'RandomForestRegressor']

        comparisons.fit(data=df, run_type="dry_run", include=include)
        y = np.random.random(100)
        x = np.random.random((100, 8))
        comparisons._build_predict_from_configs(x, y)

        assert isinstance(comparisons.metrics['model_RandomForestRegressor']['test']['r2'], float)

        return


if __name__=="__main__":
    unittest.main()