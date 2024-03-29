
import os
import time
import unittest
import site

import matplotlib.pyplot as plt

site.addsitedir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestRegressor

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.postprocessing import PermutationImportance
from ai4water.postprocessing import PartialDependencePlot
from ai4water.datasets import busan_beach, MtropicsLaos
from ai4water._main import DataNotFound
from ai4water.functional import Model as FModel
from ai4water.preprocessing import DataSet
from ai4water.models import LSTM


data = busan_beach(inputs=['tide_cm', 'pcp_mm', 'sal_psu'])
dh = DataSet(data=data, verbosity=0)
x_reg, y_reg = dh.training_data()

laos = MtropicsLaos()
data_cls = laos.make_classification(lookback_steps=2)
dh_cls = DataSet(data=data_cls, verbosity=0)
x_cls, y_cls = dh_cls.training_data()


class MyRF(RandomForestRegressor):
    pass


def test_user_defined_data(_model, x, y):
    # using user defined x
    t, p = _model.predict(x=x, return_true=True)
    assert t is None
    assert len(p) == len(x)

    # using user defined x and y, post_processing must happen
    t, p = _model.predict(x=x, y=y, return_true=True)
    assert isinstance(t, np.ndarray)
    assert isinstance(p, np.ndarray)
    assert len(t) == len(p) == len(y)

    return


def _test_ml_inbuilt_data_reg(_model):
    model = _model(model="RandomForestRegressor",
                   verbosity=0)
    model.fit(data=data)

    test_user_defined_data(model, x_reg, y_reg)

    model.predict_on_all_data(data=data)
    model.predict_on_training_data(data=data)
    model.predict_on_validation_data(data=data)

    t, p = model.predict_on_test_data(data=data, return_true=True)
    assert isinstance(t, np.ndarray)
    assert isinstance(p, np.ndarray)

    assert len(t) == len(p)
    return


def _test_ml_inbuilt_data_cls(_model):
    model = _model(model="RandomForestClassifier",
                   verbosity=0)
    model.fit(data=data_cls)

    test_user_defined_data(model, x_cls, y_cls)

    t, p = model.predict_on_test_data(data=data_cls, return_true=True)
    assert isinstance(t, np.ndarray)
    assert isinstance(p, np.ndarray)
    assert len(t) == len(p)
    return


def _test_ml_userdefined_data(_model, model_name, x, y):
    model = _model(model=model_name, verbosity=0)
    model.fit(x=x, y=y)

    test_user_defined_data(model, x, y)

    return model


def _test_ml_userdefined_non_kw(_model, model_name, x, y):
    # using non-keyword arguments to .predict
    model = _model(model=model_name, verbosity=0)
    model.fit(x=x, y=y)

    p = model.predict(x)
    assert isinstance(p, np.ndarray)
    return


def _test_hydro_metrics(_model, model_name, x, y):
    model = _model(model=model_name, verbosity=0)
    model.fit(x=x, y=y)

    for metrics in ["minimal", "hydro_metrics", "all"]:
        p = model.predict(x=x, metrics=metrics)
        assert isinstance(p, np.ndarray)
    return


class TestPredictMethod(unittest.TestCase):
    """Tests the `predict` method of Model class"""

    def test_ml_inbuilt_data(self):
        _test_ml_inbuilt_data_reg(Model)
        #_test_ml_inbuilt_data_cls(Model)
        return

    def test_ml_inbuilt_data_fn(self):
        #_test_ml_inbuilt_data_reg(FModel)
        _test_ml_inbuilt_data_cls(FModel)
        return

    def test_ml_userdefined_data(self):
        # the model does not have dh_, so we can not call .predict()
        model = _test_ml_userdefined_data(Model, "RandomForestRegressor", x_reg, y_reg)
        self.assertRaises(DataNotFound, model.predict)

        model = _test_ml_userdefined_data(Model, "RandomForestClassifier", x_cls, y_cls)
        self.assertRaises(DataNotFound, model.predict)
        return

    def test_ml_userdefined_data_fn(self):
        model = _test_ml_userdefined_data(FModel, "RandomForestRegressor", x_reg, y_reg)
        # using data generated by DataHnadler
        self.assertRaises(DataNotFound, model.predict)

        model = _test_ml_userdefined_data(FModel, "RandomForestClassifier", x_cls, y_cls)
        # using data generated by DataHnadler
        self.assertRaises(DataNotFound, model.predict)
        return

    def test_ml_userdefined_non_kw(self):
        _test_ml_userdefined_non_kw(Model, "RandomForestRegressor", x_reg, y_reg)
        _test_ml_userdefined_non_kw(Model, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_ml_userdefined_non_kw_fn(self):
        _test_ml_userdefined_non_kw(FModel, "RandomForestRegressor", x_reg, y_reg)
        _test_ml_userdefined_non_kw(FModel, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_hydro_metrics(self):
        _test_hydro_metrics(Model, "RandomForestRegressor", x_reg, y_reg)
        _test_hydro_metrics(Model, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_hydro_metrics_functional(self):
        _test_hydro_metrics(FModel, "RandomForestRegressor", x_reg, y_reg)
        _test_hydro_metrics(FModel, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_without_fit_with_data(self):
        """call to predict method without training/fit"""
        model = Model(model={"layers": {"Dense_0": 8,
                                        "Dense_1": 1}},
                    input_features=data.columns.tolist()[0:-1],
                    output_features=data.columns.tolist()[-1:],
                    verbosity=0)
        p = model.predict_on_test_data(data=data)
        assert isinstance(p, np.ndarray)
        return

    def test_without_fit_with_xy(self):
        """call to predict method without training/fit by provided x and y keywords"""
        model = Model(model={"layers": {"Dense_0": 8,
                                        "Dense_1": 1}},
                    input_features=data.columns.tolist()[0:-1],
                    output_features=data.columns.tolist()[-1:],
                    verbosity=0)
        t, p = model.predict(x=x_reg, y=y_reg, return_true=True)
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)
        return

    def test_without_fit_with_only_x(self):
        """call to predict method without training/fit by providing only x"""
        model = Model(model={"layers": {"Dense_0": 8,
                                        "Dense_1": 1}},
                    input_features=data.columns.tolist()[0:-1],
                    output_features=data.columns.tolist()[-1:],
                    verbosity=0)
        p = model.predict(x=x_reg)
        assert isinstance(p, np.ndarray)
        return

    def test_with_no_test_data(self):
        """we have only training and validation data and not test data"""
        model = Model(model="RandomForestRegressor",
            train_fraction=1.0, verbosity=0)
        model.fit(data=data)
        statement = model.predict_on_test_data

        self.assertRaises(DataNotFound, statement, data=data)
        return

    def test_with_no_val_and_test_data(self):
        """we have only training data and no validation and test data"""
        model = Model(model="RandomForestRegressor",
            train_fraction=1.0,
            val_fraction=0.0, verbosity=0)
        model.fit(data=data)
        statement = model.predict_on_test_data

        self.assertRaises(DataNotFound, statement, data=data)

        return

    def test_tf_data(self):
        """when x is tf.data.Dataset"""

        model = Model(model={"layers": {"Dense": 1}},
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      verbosity=0
                      )
        x,y = DataSet(data=data, verbosity=0).training_data()
        tr_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=32)
        p = model.predict(x=tr_ds)
        return


class TestPermImp(unittest.TestCase):

    def test_basic0(self):
        model = Model(model="XGBRegressor",
                      verbosity=0)
        model.fit(data=data)
        imp = model.permutation_importance(data_type="validation", data=data)
        assert isinstance(imp, PermutationImportance)
        return

    def test_lookback(self):
        time.sleep(1)
        model = Model(model=LSTM(1, input_shape=(3, 3)),
                      ts_args={"lookback": 3},
                      verbosity=0)
        model.fit(data=data)
        imp = model.permutation_importance(data=data, data_type="validation")
        assert isinstance(imp, PermutationImportance)
        return


class TestPDP(unittest.TestCase):

    def test_basic0(self):
        model = Model(model="XGBRegressor",
                      verbosity=0)
        model.fit(data=data)
        pdp = model.partial_dependence_plot(x=data.iloc[:, 0:-1],
                                            feature_name='tide_cm',
                                            num_points=2)
        assert isinstance(pdp, PartialDependencePlot)
        return

    def test_lookback(self):
        model = Model(model=LSTM(1, input_shape=(3, 3)),
                      ts_args={"lookback": 3},
                      verbosity=0)
        model.fit(data=data)
        pdp = model.partial_dependence_plot(data=data,
                                            feature_name='tide_cm',
                                            num_points=2)
        assert isinstance(pdp, PartialDependencePlot)
        return


class TestShapValues(unittest.TestCase):

    def test_basic0(self):
        model = Model(model="XGBRegressor",
                      verbosity=0)
        model.fit(data=data)
        sv = model.shap_values(data=data)
        assert isinstance(sv, np.ndarray)
        return

    def test_lookback(self):
        time.sleep(1) # todo
        model = FModel(model=LSTM(1, input_shape=(3, 3)),
                      ts_args={"lookback": 3},
                      verbosity=0)
        model.fit(data=data)
        sv = model.shap_values(data=data)
        assert isinstance(sv, (np.ndarray, list)), f"{type(sv)}"
        return


class TestCustomModel(unittest.TestCase):
    """for custom models, user has to tell lookback and loss"""
    def test_uninitiated(self):

        model = Model(model=MyRF,
                      ts_args={'lookback':1}, verbosity=0, mode="regression")
        model.fit(data=data)
        test_user_defined_data(model, x_reg, y_reg)
        model.evaluate()
        return

    def test_uninitiated_with_kwargs(self):
        model = Model(model={MyRF: {"n_estimators": 10}},
                      ts_args={'lookback': 1},
                      verbosity=0,
                      mode="regression")
        model.fit(data=data)
        test_user_defined_data(model, x_reg, y_reg)
        model.evaluate()
        return

    def test_initiated(self):
        model = Model(model=MyRF(), mode="regression", verbosity=0)
        model.fit(data=data)
        test_user_defined_data(model, x_reg, y_reg)
        model.evaluate()
        return

    def test_initiated_with_kwargs(self):
        # should raise error
        self.assertRaises(ValueError, Model,
                          model={RandomForestRegressor(): {"n_estimators": 10}})
        return

    def test_without_fit_method(self):
    #     # should raise error
        rgr = RandomForestRegressor

        class MyModel:
            def predict(SELF, *args, **kwargs):
                return rgr.predict(*args, **kwargs)

        self.assertRaises(ValueError, Model, model=MyModel)
        return

    def test_without_predict_method(self):
        rgr = RandomForestRegressor
        class MyModel:
            def fit(SELF, *args, **kwargs):
                return rgr.fit(*args, **kwargs)

        self.assertRaises(ValueError, Model, model=MyModel)
        return


class TestChangeNQuantiles(unittest.TestCase):

    def test_x_transformation_str(self):
        model = Model(model="RandomForestRegressor",
                      x_transformation="quantile",
                      verbosity=0)
        model.fit(data=data)
        assert model.config['x_transformation']['n_quantiles']<1000
        return

    def test_x_transformation_dict(self):
        model = Model(model="RandomForestRegressor",
                      x_transformation={'method': 'quantile', 'n_quantiles': 1000},
                      verbosity=0)
        model.fit(data=data)
        assert model.config['x_transformation']['n_quantiles']<1000
        return

    def test_y_transformation_str(self):
        model = Model(model="RandomForestRegressor",
                      y_transformation="quantile",
                      verbosity=0)
        model.fit(data=data)
        assert model.config['y_transformation']['n_quantiles']<1000
        return

    def test_y_transformation_dict(self):
        model = Model(model="RandomForestRegressor",
                      y_transformation={'method': 'quantile', 'n_quantiles': 1000},
                      verbosity=0)
        model.fit(data=data)
        assert model.config['y_transformation']['n_quantiles']<1000
        return

    def test_x_transformation_list(self):
        model = Model(model="RandomForestRegressor",
                      x_transformation=[{'method': 'quantile', 'n_quantiles': 1000},
                                        {'method': 'quantile', 'n_quantiles': 2000}],
                      verbosity=0)
        model.fit(data=data)
        assert model.config['x_transformation'][0]['n_quantiles'] < 1000
        assert model.config['x_transformation'][1]['n_quantiles'] < 1000

        return


class TestPredictionAnalysis(unittest.TestCase):

    def test_prediction_dist(self):

        model = Model(model="XGBRegressor")
        model.fit(data=busan_beach())
        ax = model.prediction_analysis(features="tide_cm",
        data=busan_beach(), show=False, save_metadata=False)
        assert isinstance(ax, plt.Axes)
        return

    def test_interaction(self):

        model = Model(model="XGBRegressor")
        model.fit(data=busan_beach())
        ax = model.prediction_analysis(
            ['tide_cm', 'sal_psu'],
            data=busan_beach(),
            custom_grid=[[-41.4, -20.0, 0.0, 20.0, 42.0],
                         [33.45, 33.7, 33.9, 34.05, 34.4]],
            annotate_kws={"annotate_counts": True,
                          "annotate_colors": ("black", "black"),
                          "annotate_fontsize": 10},
            show=False, save_metadata=False
        )
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":

    unittest.main()
