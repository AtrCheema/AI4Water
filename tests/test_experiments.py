import os
import unittest
import site   # so that ai4water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from ai4water.experiments import MLRegressionExperiments
from ai4water.utils.datasets import arg_beach


input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'pcp12_mm',
                  'air_p_hpa']
# column in dataframe to bse used as output/target
outputs = ['blaTEM_coppml']

df = arg_beach(input_features, outputs)

class TestExperiments(unittest.TestCase):

    def test_dryrun(self):

        comparisons = MLRegressionExperiments(
            data=df, input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args': {'strategy': 'mean'}, 'features': input_features}
        )
        exclude = []

        comparisons.fit(run_type="dry_run", exclude=exclude)
        comparisons.compare_errors('r2')
        best_models = comparisons.compare_errors('r2', cutoff_type='greater', cutoff_val=0.01)
        self.assertGreater(len(best_models), 1), len(best_models)
        return

    def test_optimize(self):
        best_models = ['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor',
                       'ADABoostRegressor',
                       #'RadiusNeighborsRegressor',  # todo error when using radusneighborsregressor
                       'XGBoostRFRegressor'
            ]

        comparisons = MLRegressionExperiments(
            data=df,
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args':  {'strategy': 'mean'}, 'features': input_features},
            exp_name="BestMLModels")
        comparisons.num_samples = 2
        comparisons.fit(run_type="optimize", opt_method="random", include=best_models, post_optimize='train_best')
        comparisons.compare_errors('r2')
        comparisons.taylor_plot()

    def test_from_config(self):

        exp = MLRegressionExperiments(
            data=df,
            input_features=input_features,
            output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'features': input_features, 'imputer_args': {'strategy': 'mean'}},
            exp_name="BestMLModels")
        exp.fit(run_type="dry_run",
                include=['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor'],
                post_optimize='train_best')

        exp2 = MLRegressionExperiments.from_config(os.path.join(exp.exp_path, "config.json"))

        self.assertEqual(exp2.exp_name, exp.exp_name)
        self.assertEqual(exp2.exp_path, exp.exp_path)

if __name__=="__main__":
    unittest.main()

