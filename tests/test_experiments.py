import os
import unittest
import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import pandas as pd
import sklearn

from dl4seq.experiments import MLRegressionExperiments


input_features = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
                  'input11']
# column in dataframe to bse used as output/target
outputs = ['target7']

fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dl4seq/data/data_30min.csv")
df = pd.read_csv(fname)
df.index = pd.to_datetime(df['Date_Time2'])

class TestExperiments(unittest.TestCase):

    def test_dryrun(self):

        comparisons = MLRegressionExperiments(data=df, inputs=input_features, outputs=outputs,
                                              input_nans={'SimpleImputer': {'strategy': 'mean'}} )
        exclude = ['model_GammaRegressor', 'model_TPOTREGRESSOR']
        if int(sklearn.__version__.split('.')[1]) < 23:
            exclude += ['model_POISSONREGRESSOR', 'model_TWEEDIEREGRESSOR']
        comparisons.fit(run_type="dry_run", exclude=exclude)
        comparisons.compare_errors('r2')
        best_models = comparisons.compare_errors('r2', cutoff_type='greater', cutoff_val=0.1)
        self.assertGreater(len(best_models), 1)
        return

    def test_optimize(self):
        best_models = ['model_GaussianProcessRegressor',
                       'model_HISTGRADIENTBOOSTINGREGRESSOR',
                       'model_HISTGRADIENTBOOSTINGREGRESSOR',
                       'model_ADABOOSTREGRESSOR',
                       'model_RADIUSNEIGHBORSREGRESSOR',
                       #'model_XGBOOSTRFREGRESSOR'
            ]

        comparisons = MLRegressionExperiments(data=df, inputs=input_features, outputs=outputs,
                                              input_nans={'SimpleImputer': {'strategy': 'mean'}}, exp_name="BestMLModels")
        comparisons.num_samples = 2
        comparisons.fit(run_type="optimize", opt_method="grid", include=best_models, post_optimize='train_best')
        comparisons.compare_errors('r2')
        comparisons.plot_taylor()


if __name__=="__main__":
    unittest.main()

