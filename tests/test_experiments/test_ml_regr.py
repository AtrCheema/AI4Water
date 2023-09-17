
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


def test_cross_val(scoring):
    comparisons = MLRegressionExperiments(
        input_features=input_features,
        output_features=outputs,
        nan_filler={'method': 'SimpleImputer', 'imputer_args': {'strategy': 'mean'},
                    'features': input_features},
        cross_validator={"KFold": {"n_splits": 5}},
        exp_name=f"MLRegrCrossVal_{dateandtime_now()}",
        verbosity=0,
        show=False, save=False)

    comparisons.fitcv(data=df,
                      include=[
                          'HistGradientBoostingRegressor',
                          'RandomForestRegressor'],
                      scoring=scoring)
    comparisons.compare_errors('r2', data=df)
    comparisons.taylor_plot(data=df)
    comparisons.plot_cv_scores()
    comparisons.plot_cv_scores(scoring='mae')
    comparisons.taylor_plot(data=df, include=['GaussianProcessRegressor',
                                              'RandomForestRegressor'])
    comparisons.plot_cv_scores(include=['GaussianProcessRegressor',
                                        'RandomForestRegressor'])

    models = comparisons.sort_models_by_metric('r2')
    assert isinstance(models, pd.DataFrame)

    comparisons.compare_regression_plots(data=df)
    comparisons.compare_residual_plots(data=df)
    comparisons.compare_edf_plots(data=df)
    return


class TestExperiments(unittest.TestCase):

    def test_dryrun(self):

        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args': {'strategy': 'mean'},
                        'features': input_features},
            exp_name=f"dryrun_{dateandtime_now()}",
            verbosity=0,
            show=False, save=False
        )
        exclude = [
            'model_RadiusNeighborsRegressor'  # nan predictions
                   ]

        comparisons.fit(data=df, run_type="dry_run", exclude=exclude)

        comparisons.compare_errors('r2', data=df)
        best_models = comparisons.compare_errors('r2', data=df, cutoff_type='greater',
                                                 cutoff_val=0.01)
        comparisons.taylor_plot(data=df)
        comparisons.compare_regression_plots(data=df)
        comparisons.compare_regression_plots(data=df, include=["RandomForestRegressor",
                                                               "DecisionTreeRegressor"])
        comparisons.compare_residual_plots(data=df)
        comparisons.compare_residual_plots(data=df, include=["RandomForestRegressor",
                                                               "DecisionTreeRegressor"])
        self.assertGreater(len(best_models), 1), len(best_models)
        return

    def test_with_xy(self):

        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args': {'strategy': 'mean'},
                        'features': input_features},
            verbosity=0,
            show=False, save=False
        )

        ds = DataSet(df, input_features=input_features, output_features=outputs)
        x,y = ds.training_data()
        comparisons.fit(x=x, y=y,
                        validation_data=ds.validation_data(),
                        include=['GaussianProcessRegressor', 'RandomForestRegressor']
                        )
        comparisons.compare_regression_plots(x=x, y=y)
        comparisons.compare_residual_plots(x=x, y=y)
        comparisons.compare_edf_plots(x=x, y=y)

        return

    def test_optimize(self):
        best_models = ['BaggingRegressor', 'ARDRegression', "LassoLarsIC", "NuSVR",
                       "SVR", "TheilSenRegressor", "SGDRegressor", "OneClassSVM"]

        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'imputer_args':  {'strategy': 'mean'},
                        'features': input_features},
            exp_name=f"BestMLModels_{dateandtime_now()}",
            verbosity=0,
            show=False, save=False)

        comparisons.num_samples = 2
        comparisons.fit(data=df, run_type="optimize", opt_method="random",
                        num_iterations=4,
                        include=best_models, post_optimize='eval_best',
                        hpo_kws=dict(process_results=False))
        comparisons.compare_errors('r2', data=df)
        comparisons.taylor_plot(data=df)
        comparisons.plot_improvement('r2')
        comparisons.plot_improvement('mse')
        comparisons.compare_convergence()

        comparisons.compare_regression_plots(data=df)
        comparisons.compare_residual_plots(data=df)
        comparisons.compare_edf_plots(data=df)

        return

    def test_cross_val_with_multiple_scoring(self):

        test_cross_val(['r2', 'r2_score', 'rmse', 'mae'])

        return

    def test_cross_val_with_single_scoring(self):

        test_cross_val('r2')

        return

    def test_from_config(self):

        exp = MLRegressionExperiments(
            input_features=input_features,
            output_features=outputs,
            nan_filler={'method': 'SimpleImputer', 'features': input_features,
                        'imputer_args': {'strategy': 'mean'}},
            exp_name=f"BestMLModels_{dateandtime_now()}",
            verbosity=0,
            show=False, save=False)
        exp.fit(data=df,
                run_type="optimize",
                opt_method="random",
                num_iterations=4,
                include=['GaussianProcessRegressor',
                       'HistGradientBoostingRegressor'],
                post_optimize='train_best')

        exp2 = MLRegressionExperiments.from_config(os.path.join(exp.exp_path, "config.json"))
        exp2.show = False
        exp2.save = False

        self.assertEqual(exp2.exp_name, exp.exp_name)
        self.assertEqual(exp2.exp_path, exp.exp_path)
        self.assertEqual(len(exp.metrics), len(exp2.metrics))
        self.assertEqual(len(exp.features), len(exp2.features))
        exp2.compare_errors('r2', data=df)

        exp2.taylor_plot(data=df)

        exp2.compare_regression_plots(data=df)
        exp2.compare_residual_plots(data=df)
        exp2.compare_edf_plots(data=df)

        return


class TestNonSKlearn(unittest.TestCase):

    def test_basic(self):
        comparisons = MLRegressionExperiments(
            input_features=input_features, output_features=outputs,
            verbosity=0,
            show=False, save=False
        )

        include = ["CatBoostRegressor", "XGBRegressor", "LGBMRegressor"]

        comparisons.fit(data=df, run_type="dry_run", include=include)

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
                    #'ts_args': {'lookback': int(kwargs['lookback'])},
                    'batch_size': int(kwargs['batch_size']),
                    'lr': float(kwargs['lr']),
                    'x_transformation': kwargs['transformation']
                }

        cases = {'model_minmax': {'transformation': 'minmax'},
                 'model_zscore': {'transformation': 'zscore'}}
        search_space = [
            Integer(low=16, high=64, name='lstm_units', num_samples=2),
            #Integer(low=3, high=15, name="lookback", num_samples=2),
            Categorical(categories=[4, 8, 12, 16, 24, 32], name='batch_size'),
            Real(low=1e-6, high=1.0e-3, name='lr', prior='log', num_samples=2),
            Categorical(categories=['relu', 'elu'], name='dense_actfn'),
        ]

        x0 = [4, #5,
              32, 0.00029613, 'relu']
        experiment = MyTransformationExperiments(
            cases=cases,
             input_features=input_features,
             output_features = outputs,
             param_space=search_space,
             x0=x0,
             verbosity=0,
             ts_args={"lookback": 5},
             exp_name = f"testing_{dateandtime_now()}",
            show=False, save=False)
        experiment.num_samples = 2
        experiment.fit(data = df, run_type='optimize',
                       opt_method='random',
                       num_iterations=2)

        experiment.compare_regression_plots(data=df)
        experiment.compare_residual_plots(data=df)
        experiment.compare_edf_plots(data=df)
        return

    def test_fit_with_tpot(self):
        exp = MLRegressionExperiments(
            exp_name=f"tpot_{dateandtime_now()}",
            verbosity=0,
            show=False, save=False)

        exp.fit(
            data=busan_beach(),
            include=[
            "RandomForestRegressor",
            "GradientBoostingRegressor"])

        exp.fit_with_tpot( data=busan_beach(), models=2, generations=1,
                           population_size=1)
        return

    def test_fit_with_tpot1(self):

        exp = MLRegressionExperiments(
            exp_name=f"tpot_{dateandtime_now()}",
            verbosity=0,
            show=False, save=False)

        exp.fit_with_tpot(
            data=busan_beach(),
            models = ["LGBMRegressor", "RandomForestRegressor"],
            generations=1,
            population_size=1)
        return


if __name__=="__main__":
    unittest.main()

