import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import warnings
warnings.filterwarnings("ignore")

from ai4water.experiments import MLRegressionExperiments, TransformationExperiments
from ai4water.datasets import arg_beach
from ai4water.hyperopt import Categorical, Integer, Real
from ai4water.utils.utils import dateandtime_now


input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'pcp12_mm',
                  'air_p_hpa']
# column in dataframe to bse used as output/target
outputs = ['blaTEM_coppml']

df = arg_beach(input_features, outputs)

class TestExperiments(unittest.TestCase):

    def test_dryrun(self):

        comparisons = MLRegressionExperiments(
            data=df, input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args': {'strategy': 'mean'}, 'features': input_features},
            verbosity=0
        )
        exclude = []

        comparisons.fit(run_type="dry_run", exclude=exclude)
        comparisons.compare_errors('r2', show=False)
        best_models = comparisons.compare_errors('r2', cutoff_type='greater', cutoff_val=0.01, show=False)
        self.assertGreater(len(best_models), 1), len(best_models)
        return

    def test_optimize(self):
        best_models = ['GaussianProcessRegressor',
                       #'HistGradientBoostingRegressor',
                       #'ADABoostRegressor',
                       #'RadiusNeighborsRegressor',  # todo error when using radusneighborsregressor
                       'XGBoostRFRegressor'
            ]

        comparisons = MLRegressionExperiments(
            data=df,
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args':  {'strategy': 'mean'}, 'features': input_features},
            exp_name="BestMLModels",
        verbosity=0)
        comparisons.num_samples = 2
        comparisons.fit(run_type="optimize", opt_method="random",
                        num_iterations=4,
                        include=best_models, post_optimize='train_best')
        comparisons.compare_errors('r2', show=False)
        comparisons.taylor_plot(show=False)
        return

    def test_cross_val(self):

        comparisons = MLRegressionExperiments(
            data=df,
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args':  {'strategy': 'mean'}, 'features': input_features},
            cross_validator = {"KFold": {"n_splits": 5}},
            exp_name="MLRegrCrossVal",
        verbosity=0)
        comparisons.fit(cross_validate=True, include=['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor',
                       'XGBoostRFRegressor'])
        comparisons.compare_errors('r2', show=False)
        comparisons.taylor_plot(show=False)
        comparisons.plot_cv_scores(show=False)
        comparisons.taylor_plot(show=False, include=['GaussianProcessRegressor', 'XGBoostRFRegressor'])
        comparisons.plot_cv_scores(show=False, include=['GaussianProcessRegressor', 'XGBoostRFRegressor'])
        return

    def test_from_config(self):

        exp = MLRegressionExperiments(
            data=df,
            input_features=input_features,
            output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'features': input_features, 'imputer_args': {'strategy': 'mean'}},
            exp_name=f"BestMLModels_{dateandtime_now()}",
        verbosity=0)
        exp.fit(run_type="dry_run",
                include=['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor'],
                post_optimize='train_best')

        exp2 = MLRegressionExperiments.from_config(os.path.join(exp.exp_path, "config.json"))

        self.assertEqual(exp2.exp_name, exp.exp_name)
        self.assertEqual(exp2.exp_path, exp.exp_path)

        return

    def test_transformation_experiments(self):

        class MyTransformationExperiments(TransformationExperiments):

            def update_paras(self, **kwargs):

                _layers = {
                "LSTM": {"config": {"units": int(kwargs['lstm_units'])}},
                "Dense": {"config": {"units": 1, "activation": kwargs['dense_actfn']}},
                "reshape": {"config": {"target_shape": (1, 1)}}
                }

                return {
                    'model': {'layers': _layers},
                    'lookback': int(kwargs['lookback']),
                    'batch_size': int(kwargs['batch_size']),
                    'lr': float(kwargs['lr']),
                    'transformation': kwargs['transformation']
                }

        cases = {'model_minmax': {'transformation': 'minmax'},
                 'model_zscore': {'transformation': 'zscore'}}
        search_space = [
            Integer(low=16, high=64, name='lstm_units', num_samples=2),
            Integer(low=3, high=15, name="lookback", num_samples=2),
            Categorical(categories=[4, 8, 12, 16, 24, 32], name='batch_size'),
            Real(low=1e-6, high=1.0e-3, name='lr', prior='log', num_samples=2),
            Categorical(categories=['relu', 'elu'], name='dense_actfn'),
        ]

        x0 = [4, 5, 32, 0.00029613, 'relu']
        experiment = MyTransformationExperiments(cases=cases,
                                                 input_features=input_features,
                                                 output_features = outputs,
                                                 data = df,
                                                 param_space=search_space,
                                                 x0=x0,
                                                 verbosity=0,
                                                 exp_name = f"testing_{dateandtime_now()}")
        experiment.num_samples = 2
        experiment.fit('optimize', opt_method='random', num_iterations=2)
        return

    def test_fit_with_tpot(self):
        exp = MLRegressionExperiments(data=arg_beach(),
                                      exp_name=f"tpot_{dateandtime_now()}",
                                      verbosity=0)

        exp.fit(include=[
            "XGBoostRegressor",
            "LGBMRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor"])

        exp.fit_with_tpot(2, generations=1, population_size=1)
        return

    def test_fit_with_tpot1(self):

        exp = MLRegressionExperiments(data=arg_beach(),
                                      exp_name=f"tpot_{dateandtime_now()}",
                                      verbosity=0)

        exp.fit_with_tpot(["LGBMRegressor",
                           "RandomForestRegressor"],
                          generations=1, population_size=1)
        return


if __name__=="__main__":
    unittest.main()

