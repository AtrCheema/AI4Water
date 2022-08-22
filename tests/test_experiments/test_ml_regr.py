
import os
import site
import unittest

import pandas as pd

ai4_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(ai4_dir)

import warnings
warnings.filterwarnings("ignore")

from ai4water.datasets import busan_beach
from ai4water.preprocessing import DataSet
from ai4water.utils.utils import dateandtime_now
from ai4water.hyperopt import Categorical, Integer, Real
from ai4water.experiments import MLRegressionExperiments, TransformationExperiments


input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm',
                  'pcp3_mm', 'pcp12_mm', 'air_p_hpa']
# column in dataframe to bse used as output/target
outputs = ['blaTEM_coppml']

df = busan_beach(input_features, outputs)


class TestExperiments(unittest.TestCase):

    def test_dryrun(self):

        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args': {'strategy': 'mean'},
                        'features': input_features},
            verbosity=0
        )
        exclude = []

        comparisons.fit(data=df, run_type="dry_run", exclude=exclude)
        comparisons.compare_errors('r2', show=False)
        best_models = comparisons.compare_errors('r2', cutoff_type='greater',
                                                 cutoff_val=0.01, show=False)
        comparisons.taylor_plot(show=False)
        self.assertGreater(len(best_models), 1), len(best_models)
        return

    def test_with_xy(self):

        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args': {'strategy': 'mean'},
                        'features': input_features},
            verbosity=0
        )

        ds = DataSet(df, input_features=input_features, output_features=outputs)
        x,y = ds.training_data()
        comparisons.fit(x=x, y=y,
                        validation_data=ds.validation_data(),
                        include=['GaussianProcessRegressor', 'XGBRFRegressor']
                        )

        return

    def test_optimize(self):
        best_models = ['GaussianProcessRegressor', 'XGBRFRegressor']

        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args':  {'strategy': 'mean'},
                        'features': input_features},
            exp_name="BestMLModels",
        verbosity=0)
        comparisons.num_samples = 2
        comparisons.fit(data=df, run_type="optimize", opt_method="random",
                        num_iterations=4,
                        include=best_models, post_optimize='eval_best')
        comparisons.compare_errors('r2', show=False)
        comparisons.taylor_plot(show=False)
        comparisons.plot_improvement('r2', save=False)
        comparisons.plot_improvement('mse', save=False)
        comparisons.compare_convergence()
        return

    def test_cross_val(self):

        comparisons = MLRegressionExperiments(
            input_features=input_features,
            output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args':  {'strategy': 'mean'},
                        'features': input_features},
            cross_validator = {"KFold": {"n_splits": 5}},
            exp_name="MLRegrCrossVal",
        verbosity=0)
        comparisons.fit(data=df,
                        cross_validate=True,
                        include=['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor',
                       'XGBRFRegressor'])
        comparisons.compare_errors('r2', show=False)
        comparisons.taylor_plot(show=False)
        comparisons.plot_cv_scores(show=False)
        comparisons.taylor_plot(show=False, include=['GaussianProcessRegressor',
                                                     'XGBRFRegressor'])
        comparisons.plot_cv_scores(show=False, include=['GaussianProcessRegressor',
                                                        'XGBRFRegressor'])

        models = comparisons.sort_models_by_metric('r2')
        assert isinstance(models, pd.DataFrame)

        return

    def test_from_config(self):

        exp = MLRegressionExperiments(
            input_features=input_features,
            output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'features': input_features,
                        'imputer_args': {'strategy': 'mean'}},
            exp_name=f"BestMLModels_{dateandtime_now()}",
        verbosity=0)
        exp.fit(data=df,
                run_type="dry_run",
                include=['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor'],
                post_optimize='train_best')

        exp2 = MLRegressionExperiments.from_config(os.path.join(exp.exp_path, "config.json"))

        self.assertEqual(exp2.exp_name, exp.exp_name)
        self.assertEqual(exp2.exp_path, exp.exp_path)
        self.assertEqual(len(exp.metrics), len(exp2.metrics))
        self.assertEqual(len(exp.features), len(exp2.features))

        return

    def test_transformation_experiments(self):

        class MyTransformationExperiments(TransformationExperiments):

            def update_paras(self, **kwargs):

                _layers = {
                "LSTM": {"config": {"units": int(kwargs['lstm_units'])}},
                "Dense": {"config": {"units": 1, "activation": kwargs['dense_actfn']}},
                }

                return {
                    'model': {'layers': _layers},
                    'ts_args': {'lookback': int(kwargs['lookback'])},
                    'batch_size': int(kwargs['batch_size']),
                    'lr': float(kwargs['lr']),
                    'x_transformation': kwargs['transformation']
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
                                                 param_space=search_space,
                                                 x0=x0,
                                                 verbosity=0,
                                                 exp_name = f"testing_{dateandtime_now()}")
        experiment.num_samples = 2
        experiment.fit(data = df, run_type='optimize', opt_method='random',
                       num_iterations=2)
        return

    def test_fit_with_tpot(self):
        exp = MLRegressionExperiments(
            exp_name=f"tpot_{dateandtime_now()}",
            verbosity=0)

        exp.fit(
            data=busan_beach(),
            include=[
            "XGBRegressor",
            "LGBMRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor"])

        exp.fit_with_tpot( data=busan_beach(), models=2, generations=1,
                           population_size=1)
        return

    def test_fit_with_tpot1(self):

        exp = MLRegressionExperiments(
            exp_name=f"tpot_{dateandtime_now()}",
            verbosity=0)

        exp.fit_with_tpot(
            data=busan_beach(),
            models = ["LGBMRegressor", "RandomForestRegressor"],
            generations=1,
            population_size=1)
        return


if __name__=="__main__":
    unittest.main()

