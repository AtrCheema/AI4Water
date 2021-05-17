import os
import unittest
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import pandas as pd

from AI4Water.experiments import MLRegressionExperiments


input_features = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
                  'input11']
# column in dataframe to bse used as output/target
outputs = ['target7']

fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AI4Water", "utils", "datasets", "mts_30min.csv")
df = pd.read_csv(fname)
df.index = pd.to_datetime(df['Date_Time2'])

class TestExperiments(unittest.TestCase):

    def test_dryrun(self):

        comparisons = MLRegressionExperiments(data=df, inputs=input_features, outputs=outputs,
                                              input_nans={'SimpleImputer': {'strategy': 'mean'}} )
        exclude = ['TPOTRegressor']

        comparisons.fit(run_type="dry_run", exclude=exclude)
        comparisons.compare_errors('r2')
        best_models = comparisons.compare_errors('r2', cutoff_type='greater', cutoff_val=0.1)
        self.assertGreater(len(best_models), 1)
        return

    def test_optimize(self):
        best_models = ['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor',
                       'ADABoostRegressor',
                       #'RadiusNeighborsRegressor',  # todo error when using radusneighborsregressor
                       'XGBoostRFRegressor'
            ]

        comparisons = MLRegressionExperiments(data=df, inputs=input_features, outputs=outputs,
                                              input_nans={'SimpleImputer': {'strategy': 'mean'}}, exp_name="BestMLModels")
        comparisons.num_samples = 2
        comparisons.fit(run_type="optimize", opt_method="random", include=best_models, post_optimize='train_best')
        comparisons.compare_errors('r2')
        comparisons.plot_taylor()

    def test_from_config(self):

        exp = MLRegressionExperiments(data=df, inputs=input_features, outputs=outputs,
                                              input_nans={'SimpleImputer': {'strategy': 'mean'}}, exp_name="BestMLModels")
        exp.fit(run_type="dry_run",
                include=['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor'],
                post_optimize='train_best')

        exp2 = MLRegressionExperiments.from_config(os.path.join(exp.exp_path, "config.json"))

        self.assertEqual(exp2.exp_name, exp.exp_name)
        self.assertEqual(exp2.exp_path, exp.exp_path)

if __name__=="__main__":
    unittest.main()

