
import os
import time
import unittest

from ai4water import Model
from ai4water.preprocessing import DataHandler
from ai4water.datasets import arg_beach
from ai4water._optimize import make_space
from ai4water.utils.utils import process_config_dict
from ai4water.utils.utils import find_opt_paras_from_model_config
from ai4water.hyperopt import Categorical, Real, Integer


data = arg_beach()
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

        df = arg_beach(inputs=["tide_cm", "wat_temp_c", "rel_hum"])

        setattr(Model, 'from_check_point', False)
        model = Model(model="XGBRegressor")

        model.optimize_transformations(
            data=df,
            exclude="tide_cm",
            algorithm="random",
            num_iterations=3)

        assert isinstance(model.config['x_transformation'], list)
        assert model.config['y_transformation'] is None
        return

    def test_optimize_xy_transformations(self):

        df = arg_beach(inputs=["tide_cm", "wat_temp_c", "rel_hum"])

        setattr(Model, 'from_check_point', False)
        model = Model(model="XGBRegressor")

        model.optimize_transformations(
            data=df,
            exclude="tide_cm",
            algorithm="random",
            y_transformations=['log', 'sqrt'],
            num_iterations=3)

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

        return


class TestOptimizeHyperparas(unittest.TestCase):
    config = {"XGBRegressor": {"n_estimators": Integer(low=10, high=20, num_samples=10),
                               "max_depth": Categorical([10, 20, 30]),
                               "learning_rate": Real(0.00001, 0.1, num_samples=10)}}

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
        optimizer = model.optimize_hyperparameters(data=arg_beach())
        s = set([v['n_estimators'] for v in optimizer.xy_of_iterations().values()])
        assert len(s) > 5  # assert that all suggestions are not same
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
        x, y = DataHandler(arg_beach().values).training_data()
        optimizer = model.optimize_hyperparameters(data=(x, y.reshape(-1, 1)), process_results=False)

        # make sure that model's config has been updated
        for k, v in optimizer.best_paras().items():
            assert model.config['model']['XGBRegressor'][k] == v

        return

    def test_cv_with_custom_data(self):
        setattr(Model, 'from_check_point', False)
        model = Model(model=self.config,
                      cross_validator={"KFold": {'n_splits': 5}},
                      verbosity=0)
        x, y = DataHandler(arg_beach().values).training_data()
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
            data=arg_beach(),
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
                      epochs=5)

        print(model.path, 'model.path')
        optimizer = model.optimize_hyperparameters(
            data=arg_beach(),
            algorithm="random",
            num_iterations=5,
            process_results=False
        )
        s = set([v['units'] for v in optimizer.xy_of_iterations().values()])
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
                      lookback=Integer(3, 10, num_samples=10),
                      input_features=input_features,
                      output_features=output_features,
                      verbosity=0,
                      epochs=5)

        optimizer = model.optimize_hyperparameters(
            data=arg_beach(),
            algorithm="random",
            num_iterations=5,
            process_results=False)
        assert model.config['model']['layers']['LSTM']['config']['units'] == optimizer.best_paras()['units']
        assert model.config['lookback'] == optimizer.best_paras()['lookback']
        return


if __name__ == "__main__":

    unittest.main()