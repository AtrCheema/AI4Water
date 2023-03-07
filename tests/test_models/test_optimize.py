
import os
import time
import unittest

import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import site
site.addsitedir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ai4water import Model
from ai4water.preprocessing import DataSet
from ai4water.datasets import busan_beach
from ai4water._optimize import make_space
from ai4water.utils.utils import process_config_dict
from ai4water.utils.utils import find_opt_paras_from_model_config
from ai4water.hyperopt import Categorical, Real, Integer
from ai4water.postprocessing.SeqMetrics import RegressionMetrics

data = busan_beach()
input_features = data.columns.to_list()[0:-1]
output_features = data.columns.to_list()[-1:]


def get_lstm():

    return {"layers": {
        "LSTM": {"config": {"units": Integer(2, 5, num_samples=10)}},
        "relu": {},
        "Dense_0": {"units": Integer(2, 5, name="dense1_units", num_samples=10),
                   "activation": Categorical(["relu", "tanh"], name="dense1_act")},
        "Dense_1": {"config": {"units": 1, "activation": "relu"}}
    }}



class TestOptimize(unittest.TestCase):

    def test_optimize_transformations(self):

        df = busan_beach(inputs=["tide_cm", "wat_temp_c", "rel_hum", "sal_psu"])

        setattr(Model, 'from_check_point', False)
        model = Model(model="XGBRegressor")

        model.optimize_transformations(
            data=df,
            exclude="tide_cm",
            algorithm="random",
            num_iterations=3,
            process_results=False,)

        assert isinstance(model.config['x_transformation'], list)
        assert model.config['y_transformation'] is None
        return

    def test_optimize_xy_transformations(self):

        df = busan_beach(inputs=["tide_cm", "wat_temp_c", "rel_hum"])

        setattr(Model, 'from_check_point', False)
        model = Model(model="XGBRegressor")

        model.optimize_transformations(
            data=df,
            exclude="tide_cm",
            algorithm="random",
            y_transformations=['log', 'sqrt'],
            num_iterations=3,
            process_results=False,
        )

        assert isinstance(model.config['x_transformation'], list)
        assert isinstance(model.config['y_transformation'], list)
        return


    def test_make_space(self):
        space = make_space(data.columns.to_list(),
                           categories=['log', 'log2', 'minmax', 'none'])
        assert len(space) == 14
        # include
        space = make_space(data.columns.to_list(), include="tide_cm",
                           categories=['log', 'log2', 'minmax', 'none'])
        assert len(space) == 1

        include = ["tide_cm", "tetx_coppml"]
        space = make_space(data.columns.to_list(), include=include,
                           categories=['log', 'log2', 'minmax', 'none'])
        for sp, _name in zip(space, include):
            assert sp.name == _name

        exclude = "tide_cm"
        space = make_space(data.columns.to_list(), exclude=exclude,
                           categories=['log', 'log2', 'minmax', 'none'])
        for sp in space:
            assert sp.name != exclude

        exclude = ["tide_cm", "tetx_coppml"]
        space = make_space(data.columns.to_list(), exclude=exclude,
                           categories=['log', 'log2', 'minmax', 'none'])
        for sp in space:
            assert sp.name not in exclude

        new = {"tetx_coppml": ["log", "log2", "log10"]}
        space = make_space(data.columns.to_list(), include="tetx_coppml", append=new,
                           categories=['log', 'log2', 'minmax', 'none', 'log10'])
        assert len(space) == 1, space
        assert len(space[0].categories) == 3

        space = make_space(data.columns.to_list(), include=[],
                    categories=['log', 'log2', 'minmax', 'none'])
        return


class TestOptimizeHyperparas(unittest.TestCase):
    config = {"XGBRegressor": {"n_estimators": Integer(low=10, high=20, num_samples=10),
                               "max_depth": Categorical([10, 20, 30]),
                               "learning_rate": Real(0.00001, 0.1, num_samples=10)}}

    def test_no_space(self):
        # We don't provide space, it is inferred from Model name
        model = Model(model="RandomForestRegressor",
                      verbosity=0)
        optimizer = model.optimize_hyperparameters(data=data,
                                                   algorithm='bayes',
                                                   num_iterations=11,
                                                   process_results=False
                                                   )
        s = set([xy['x']['n_estimators'] for xy in optimizer.xy_of_iterations().values()])
        assert len(s) >= 5, s  # assert that all suggestions are not same

        # make sure that model's config has been updated
        for k, v in optimizer.best_paras().items():
            assert model.config['model']['RandomForestRegressor'][k] == v
        return

    def test_no_opt_paras(self):
        conf = "XGBRegressor"
        c, op, _ = find_opt_paras_from_model_config(conf)
        assert len(op) == 0
        return

    def test_no_opt_paras1(self):
        config = {"XGBRegressor": {"n_estimators": 2, "max_depth": 23}}
        c, op, _ = find_opt_paras_from_model_config(config)
        assert len(op) == 0
        return

    def test_ml(self):
        setattr(Model, 'from_check_point', False)
        model = Model(model=self.config,
                      verbosity=0)
        optimizer = model.optimize_hyperparameters(data=data, algorithm='random',
                                                   num_iterations=6
                                                   )
        s = set([xy['x']['n_estimators'] for xy in optimizer.xy_of_iterations().values()])
        assert len(s) >= 5, s  # assert that all suggestions are not same
        op = os.path.join(os.getcwd(), optimizer.opt_path)
        fname = os.path.join(op, "convergence.png")
        assert os.path.exists(fname)

        # make sure that model's config has been updated
        for k, v in optimizer.best_paras().items():
            assert model.config['model']['XGBRegressor'][k] == v
        return

    def test_with_custom_data(self):
        setattr(Model, 'from_check_point', False)
        model = Model(model=self.config,
                      verbosity=0)
        x, y = DataSet(data.values).training_data()
        optimizer = model.optimize_hyperparameters(data=(x, y.reshape(-1, 1)),
                                                   algorithm = 'random', num_iterations = 3,
                                                   process_results=False)

        # make sure that model's config has been updated
        for k, v in optimizer.best_paras().items():
            assert model.config['model']['XGBRegressor'][k] == v

        return

    def test_cv_with_custom_data(self):
        setattr(Model, 'from_check_point', False)
        model = Model(model=self.config,
                      cross_validator={"KFold": {'n_splits': 5}},
                      verbosity=0)
        x, y = DataSet(data.values).training_data()
        optimizer = model.optimize_hyperparameters(
            algorithm="random",
            data=(x, y.reshape(-1, 1)), num_iterations=4)

        # make sure that model's config has been updated
        for k, v in optimizer.best_paras().items():
            assert model.config['model']['XGBRegressor'][k] == v

        return

    def test_ml_without_procesisng_results(self):
        setattr(Model, 'from_check_point', False)

        # without process results
        model = Model(model=self.config,
                      verbosity=0)
        optimizer = model.optimize_hyperparameters(
            data=data,
            algorithm="random",
            num_iterations=3,
            process_results=False)
        op = os.path.join(os.getcwd(), optimizer.opt_path)
        fname = os.path.join(op, "convergence.png")
        assert not os.path.exists(fname)

        # make sure that model's config has been updated
        for k, v in optimizer.best_paras().items():
            assert model.config['model']['XGBRegressor'][k] == v
        return

    def test_without_model(self):
        m_conf = None
        c, op, _ = find_opt_paras_from_model_config(m_conf)
        assert c is None and len(op) == 0

        return

    def test_nn_without_space(self):
        m_conf = {"layers": {"LSTM": 64}}
        c, op = process_config_dict(m_conf)
        assert len(m_conf['layers']) == 1
        assert len(op) == 0

        # with multi layers
        m_conf = {"layers":
                      {"LSTM": 64,
                       "Dense": 1}}
        assert len(m_conf['layers']) == 2
        assert len(op) == 0

        # multi layers with empty config
        m_conf = {"layers":
                      {"LSTM": 64,
                       "ReLu": {},
                       "Dense": 1}}
        c, op = process_config_dict(m_conf)
        assert len(m_conf['layers']) == 3
        assert len(op) == 0

        return

    def test_nn_with_space(self):
        # with space
        m_conf = {"layers": {"LSTM": Integer(32, 64)}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM": {"units": Integer(32, 64)}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM": {"units": Integer(32, 64)},
                             "Dense": 1}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM": {"units": Integer(32, 64)},
                             "Dense": {"units": 1}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM": {"units": Integer(32, 64)},
                             "Dense": {"units": 1, "activation": "relu"}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM": {"units": Integer(32, 64)},
                             "relu": {},
                             "Dense": {"units": 1, "activation": "relu"}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['units'], int)
        assert len(op) == 1

        return

    def test_nn_with_config_kw(self):
        # with config keyword argument
        m_conf = {"layers": {"LSTM": {"config": {"units": Integer(32, 64)}}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['config']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM": {"config": {"units": Integer(32, 64)}},
                             "Dense": 1}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['config']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM": {"config": {"units": Integer(32, 64)}},
                             "Dense": {"config": {"units": 1}}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['config']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM":  {"config": {"units": Integer(32, 64)}},
                             "Dense": {"config": {"units": 1, "activation": "relu"}}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['config']['units'], int)
        assert len(op) == 1

        m_conf = {"layers": {"LSTM":  {"config": {"units": Integer(32, 64), "activation": "relu"}},
                             "relu": {},
                             "Dense": {"config": {"units": 1, "activation": "relu"}}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['config']['units'], int)
        assert len(op) == 1
        return

    def test_nn_multiple_opt_paras(self):
        # multiple hpo arguments in multiple layers
        m_conf = {"layers": {"LSTM":  {"config": {"units": Integer(32, 64), "activation": "relu"}},
                             "relu": {},
                             "Dense1": {"units": Integer(10, 20, name="dense1_units"),
                                        "activation": Categorical(["relu", "tanh"], name="dense1_act")},
                             "Dense": {"config": {"units": 1, "activation": "relu"}}}}
        c, op = process_config_dict(m_conf)
        assert isinstance(c['layers']['LSTM']['config']['units'], int)
        assert len(op) == 3
        return

    def test_raise_duplicate_name_error(self):
        # duplication error
        m_conf = {"layers": {"LSTM":  {"config": {"units": Integer(32, 64), "activation": "relu"}},
                             "relu": {},
                             "Dense1": {"units": Integer(10, 20),
                                        "activation": Categorical(["relu", "tanh"], name="dense1_act")}}}
        self.assertRaises(ValueError, process_config_dict, m_conf)
        return

    def test_nn_complex(self):

        m_conf ={
            "Input": {'config': {'shape': (15, 8), 'name': "MyInputs"}},
            "LSTM": {'config': {'units': Integer(32, 64, name="lstm_units"),
                                'return_sequences': True, 'return_state': True, 'name': 'MyLSTM1'},
                     'inputs': 'MyInputs',
                     'outputs': ['junk', 'h_state', 'c_state']},

            "Dense_0": {'config': {'units': 1, 'name': 'MyDense'},
                        'inputs': 'h_state'},

            "Conv1D_1": {'config': {'filters': Integer(32, 64), 'kernel_size': 3, 'name': 'myconv'},
                         'inputs': 'junk'},
            "MaxPool1D": {'config': {'name': 'MyMaxPool'},
                          'inputs': 'myconv'},
            "Flatten": {'config': {'name': 'MyFlatten'},
                        'inputs': 'MyMaxPool'},

            "LSTM_3": {"config": {'units': Integer(32, 64, name="lsmt3_units"), 'name': 'MyLSTM2'},
                       'inputs': 'MyInputs',
                       'call_args': {'initial_state': ['h_state', 'c_state']}},

            "Concatenate": {'config': {'name': 'MyConcat'},
                            'inputs': ['MyDense', 'MyFlatten', 'MyLSTM2']},

            "Dense": 1
        }
        c, op = process_config_dict(m_conf)
        assert len(op) == 3
        return

    def test_nn_optimize(self):

        time.sleep(1)
        m_conf = get_lstm()

        setattr(Model, 'from_check_point', False)

        model = Model(model=m_conf,
                      verbosity=0,
                      input_features=input_features,
                      output_features=output_features,
                      ts_args = {"lookback": 5},
                      epochs=5)

        print(model.path, 'model.path')
        optimizer = model.optimize_hyperparameters(
            data=data,
            algorithm="random",
            num_iterations=5,
            process_results=False,
            refit=False
        )
        s = set([xy['x']['units'] for xy in optimizer.xy_of_iterations().values()])
        assert len(s) >= 3  # assert that all suggestions are not same

        # make sure that model's config has been updated
        assert model.config['model']['layers']['LSTM']['config']['units'] == optimizer.best_paras()['units']
        assert model.config['model']['layers']['Dense_0']['activation'] == optimizer.best_paras()['dense1_act']
        assert model.config['model']['layers']['Dense_0']['units'] == optimizer.best_paras()['dense1_units']

        return

    def test_optimize_with_model_and_other_paras(self):
        # test that model_config and other in config such as lookback can also be optimzied
        m_conf = get_lstm()

        setattr(Model, 'from_check_point', False)

        model = Model(model=m_conf,
                      ts_args={'lookback':Integer(3, 10, num_samples=10)},
                      input_features=input_features,
                      output_features=output_features,
                      verbosity=0,
                      epochs=5)

        optimizer = model.optimize_hyperparameters(
            data=data,
            algorithm="random",
            num_iterations=5,
            process_results=False,
            refit=False
        )
        assert model.config['model']['layers']['LSTM']['config']['units'] == optimizer.best_paras()['units']
        assert model.config['ts_args']['lookback'] == optimizer.best_paras()['lookback']
        return

    def test_reproducibility(self):

        num_samples = 10

        model = Model(
            model={"HistGradientBoostingRegressor":
                {"min_samples_leaf": Integer(low=5, high=100, name="min_samples_leaf", num_samples=num_samples),
                "max_iter": Integer(low=50, high=500, name='max_iter', num_samples=num_samples),
                "learning_rate": Real(low=0.05, high=0.95, name="learning_rate", num_samples=num_samples),
                "max_leaf_nodes": Integer(low=5, high=100, name="max_leaf_nodes", num_samples=num_samples),
                "max_depth": Integer(low=5, high=500, name="max_depth", num_samples=num_samples),
                },
            },
            input_features=input_features[0:3],
            output_features=output_features,
            split_random=True,
            seed=891,
            train_fraction=1.0,
            val_fraction=0.3,
            val_metric="r2",
            verbosity=0

        )

        optimizer = model.optimize_hyperparameters(data=data, num_iterations=5, algorithm='random')

        best_val_score = optimizer.best_xy()['y']
        model.fit(data=data)
        val_metric_post_train = model.evaluate_on_validation_data(data=data, metrics='r2')
        self.assertAlmostEqual(val_metric_post_train, 1.0 - best_val_score, places=2)
        return


class TestOptimizeTransformationReproducibility(unittest.TestCase):
    """The model trained with with optimized transformations should
    give same results as were obtained during optimization."""
    # TODO, currently yeo-johnson, and quantile are not tested

    def test_ml_ytransformation_RandomSplit_Holdout(self):
        """ML model with only y transformations"""
        model = Model(
            model={"XGBRegressor": {
                "random_state": 25398
            }},
            input_features=input_features,
            output_features=output_features,
            seed=25398,
            split_random=True,
            verbosity=0
            )

        optimizer = model.optimize_transformations(
            include=[],
            y_transformations=['log', 'log2', 'sqrt', 'none', 'log10'],
            data=data,
            num_iterations=4,
            algorithm='random',
            process_results=False)
        best_val_score = optimizer.best_xy()['y']
        best_val_iter = optimizer.best_iter()

        # train model with optimized transformations
        model.fit(data=data)

        t, p = model.predict_on_validation_data(data=data, return_true=True, process_results=False)

        val_metric_post_train = getattr(RegressionMetrics(t,p), model.val_metric)()

        print(f"{best_val_score} at {best_val_iter} {val_metric_post_train}")
        self.assertAlmostEqual(val_metric_post_train, 1.0-best_val_score, places=2)
        return

    def test_ml_ytransformation_RandomSplit_Holdout_no_test_data(self):
            """ML model with only y transformations.
            No test data."""
            model = Model(
                model={"XGBRegressor": {
                    "random_state": 57420
                }},
                input_features=input_features,
                output_features=output_features,
                train_fraction=1.0,
                val_fraction=0.3,
                seed=57420,
                split_random=True,
                verbosity=0)

            optimizer = model.optimize_transformations(
                include=[],
                y_transformations=['log', 'log2', 'sqrt', 'none', 'log10'],
                data=data,
                num_iterations=4,
                algorithm='random',
                process_results=False
            )
            best_val_score = optimizer.best_xy()['y']
            best_val_iter = optimizer.best_iter()

            # train model with optimized transformations
            model.fit(data=data)

            t, p = model.predict_on_validation_data(data=data, return_true=True, process_results=False)

            val_metric_post_train = getattr(RegressionMetrics(t,p), model.val_metric)()

            print(f"{best_val_score} at {best_val_iter} {val_metric_post_train}")
            self.assertAlmostEqual(val_metric_post_train, 1.0-best_val_score, places=2)
            return

    def test_ml_ytransformation_SeqSplit_Holdout(self):
        """ML model with only y transformations"""
        model = Model(
            model="XGBRegressor",
            input_features=input_features,
            output_features=output_features,
            verbosity=0)

        optimizer = model.optimize_transformations(
            include=[],
            y_transformations=['log', 'log2', 'sqrt', 'none', 'log10'],
            data=data,
            num_iterations=4,
            algorithm='random',
            process_results=False
        )
        best_val_score = optimizer.best_xy()['y']

        # train model with optimized transformations
        model.fit(data=data)

        t, p = model.predict_on_validation_data(data=data, return_true=True, process_results=False)

        val_metric_post_train = getattr(RegressionMetrics(t,p), model.val_metric)()
        self.assertAlmostEqual(val_metric_post_train, 1.0-best_val_score, places=2)
        return

    def test_ml_xytransformation_SeqSplit_Holdout(self):
        """ML model with only y transformations
        """
        model = Model(
            model="XGBRegressor",
            input_features=input_features,
            output_features=output_features,
            verbosity=0
            )

        optimizer = model.optimize_transformations(
            transformations=['log', 'log2'],
            y_transformations=['log', 'log2'],
            data=data,
            num_iterations=4,
            algorithm='random',
            process_results=False
            )
        best_val_score = optimizer.best_xy()['y']
        best_val_iter = optimizer.best_iter()

        # train model with optimized transformations
        model.fit(data=data)

        t, p = model.predict_on_validation_data(data=data, return_true=True, process_results=False)

        val_metric_post_train = getattr(RegressionMetrics(t,p), model.val_metric)()
        print(f"{best_val_score} at {best_val_iter} {val_metric_post_train}")
        self.assertAlmostEqual(val_metric_post_train, 1.0-best_val_score, places=2)
        return

    def test_ml_xytransformation_RandomSplit_Holdout(self):
        """ML model with only y transformations
        """
        model = Model(
            model="XGBRegressor",
            input_features=input_features,
            output_features=output_features,
            seed=25398,
            split_random=True,
            verbosity=0
            )

        optimizer = model.optimize_transformations(
            transformations=['log', 'log2'],
            y_transformations=['log', 'log2'],
            data=data,
            num_iterations=4,
            algorithm='random',
            process_results=False
            )
        best_val_score = optimizer.best_xy()['y']
        best_val_iter = optimizer.best_iter()

        # train model with optimized transformations
        model.fit(data=data)

        t, p = model.predict_on_validation_data(data=data, return_true=True, process_results=False)

        val_metric_post_train = getattr(RegressionMetrics(t,p), model.val_metric)()
        print(f"{best_val_score} at {best_val_iter} {val_metric_post_train}")
        self.assertAlmostEqual(val_metric_post_train, 1.0-best_val_score, places=2)
        return

    def test_ml_xytransformation_RandomSplit_Holdout_NoTestData(self):
        """ML model with only y transformations
        """
        model = Model(
            model="XGBRegressor",
            input_features=input_features,
            output_features=output_features,
            seed=25398,
            split_random=True,
            verbosity=0
            )

        optimizer = model.optimize_transformations(
            transformations=['log', 'log2', 'sqrt', 'none', 'log10'],
            data=data,
            num_iterations=4,
            algorithm='random',
            process_results=False
            )
        best_val_score = optimizer.best_xy()['y']
        best_val_iter = optimizer.best_iter()

        # train model with optimized transformations
        model.fit(data=data)

        t, p = model.predict_on_validation_data(data=data, return_true=True, process_results=False)

        val_metric_post_train = getattr(RegressionMetrics(t,p), model.val_metric)()
        print(f"{best_val_score} at {best_val_iter} {val_metric_post_train}")
        self.assertAlmostEqual(val_metric_post_train, 1.0-best_val_score, places=2)
        return


if __name__ == "__main__":

    unittest.main()