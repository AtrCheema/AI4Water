import unittest
import os
import sys
import site

import lime.lime_tabular

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import pandas as pd

from ai4water import Model
from ai4water.datasets import arg_beach
from ai4water.datasets import MtropicsLaos
from ai4water.post_processing.explain import LimeExplainer, explain_model_with_lime

laos = MtropicsLaos()

reg_data = laos.make_regression()
class_data = laos.make_classification()


def make_mlp_model():

    model = Model(
        model={'layers': {
            "Dense_0": {'units': 8},
            "Dense_1": {'units': 4},
            "Dense_2": {'units': 2},
            "Flatten": {},
            "Dense_3": 1,
        }},
        data=arg_beach(),
        epochs=2,
        lookback=1,
        verbosity=0
    )

    model.fit()
    return model


def get_fitted_model(model_name, data):

    model = Model(
        model=model_name,
        data=data,
        verbosity=0
    )

    model.fit()

    return model


def lstm_model():

    model = Model(
        model = {"layers": {
            "LSTM": 32,
            "Dense": 1
        }},
        data = arg_beach(),
        verbosity=0
    )
    return model


def make_class_model(**kwargs):

    model = Model(
        model="DecisionTreeClassifier",
        data=class_data,
        transformation={'method': 'quantile', 'features': ['Ecoli_mpn100']},
        verbosity=0,
        **kwargs
    )
    model.fit()
    return model


def make_lstm_reg_model():

    class MyModel(Model):
        def predict(self,
                *args, **kwargs
                ):
            p = super(MyModel, self).predict(*args, **kwargs)

            return p.reshape(-1,)


    model = MyModel(
        model = {"layers": {
            "LSTM": 32,
            "Dense": 1
        }},
        data = reg_data,
        verbosity=0
    )
    return model

def make_reg_model(**kwargs):

    model = Model(
        model="GradientBoostingRegressor",
        data=reg_data,
        cross_validator={'TimeSeriesSplit': {'n_splits': 10}},
        transformation={'method': 'quantile', 'features': ['Ecoli_mpn100']},
        val_metric='r2',
        verbosity=0,
        **kwargs
    )

    model.fit()

    return model


def get_data(model, to_dataframe, examples_to_explain):
    train_x, train_y = model.training_data()
    test_x, test_y = model.test_data()

    if to_dataframe:
        train_x = pd.DataFrame(train_x, columns=model.in_cols)
        test_x = pd.DataFrame(test_x[0:examples_to_explain], columns=model.in_cols)
    else:
        test_x = test_x[0:examples_to_explain]

    return train_x, test_x


def get_lime(to_dataframe=False, examples_to_explain=5, model_type="regression"):

    if model_type=="regression":
        model = make_reg_model()
    elif model_type == "lstm_reg":
        model = make_reg_model()
    elif model_type == "classification":
        model = make_class_model()
    else:
        raise ValueError

    train_x, test_x = get_data(model, to_dataframe=to_dataframe, examples_to_explain=examples_to_explain)

    lime_exp = LimeExplainer(model=model._model,
                               train_data=train_x,
                               test_data=test_x,
                               mode="regression",
                               features=list(model.in_cols),
                               path=model.path,
                               verbosity=False,
                               )
    return lime_exp


class TestLimeExplainer(unittest.TestCase):

    def test_all_examples(self):
        lime_exp = get_lime(examples_to_explain=5)
        lime_exp.explain_all_examples()
        assert len(lime_exp.explaination_objects) == 5

        return

    def test_single_example(self):

        lime_exp = get_lime()
        lime_exp.explain_example(0)
        assert len(lime_exp.explaination_objects) == 1

        return

    def test_save_as_html(self):

        lime_exp = get_lime()
        lime_exp.explain_example(0, plot_type="html")
        return

    def test_wth_pandas(self):
        lime_exp = get_lime(True)
        lime_exp.explain_example(0)

        assert len(lime_exp.explaination_objects) == 1

        return

    def test_docstring_example(self):

        model = Model(model="GradientBoostingRegressor", data=arg_beach(), verbosity=0)
        model.fit()
        lime_exp = LimeExplainer(model=model._model,
                                   train_data=model.training_data()[0],
                                   test_data=model.test_data()[0][0:3],
                                   mode="regression",
                                   verbosity=False,
                                   path=model.path
                                   )
        lime_exp()

        assert len(lime_exp.explaination_objects) == 3

        return

    def test_classification(self):

        lime_exp = get_lime(examples_to_explain=2, model_type="classification")
        lime_exp.explain_all_examples()
        assert len(lime_exp.explaination_objects) == 2

        return

    def test_ai4water_regression(self):
        model = make_reg_model(test_fraction=0.05)
        assert model.mode == "regression"

        model.explain()
        return

    def test_ai4water(self):
        model = make_class_model(test_fraction=0.05)
        self.assertEqual(model.mode, "classification")
        model.fit()
        model.explain()
        return

    def test_custom_colors(self):
        explainer = get_lime()
        explainer.explain_example(0, colors=([0.9375    , 0.01171875, 0.33203125],
                                             [0.23828125, 0.53515625, 0.92578125]))

    def test_lstm_model(self):
        m = make_lstm_reg_model()
        train_x, _ =  m.training_data()
        test_x, _ =  m.test_data()

        exp = LimeExplainer(m, test_x, train_x, mode="regression",
                              path=m.path,
                              explainer="RecurrentTabularExplainer", features=m.dh.input_features)
        exp.explain_example(0)
        return

    def test_ai4water_ml(self):

        for m in [
            "XGBoostRegressor",
            "RandomForestRegressor",
            "GRADIENTBOOSTINGREGRESSOR"
                  ]:

            model = get_fitted_model(m, arg_beach())
            exp = explain_model_with_lime(model, examples_to_explain=2)
            assert isinstance(exp, LimeExplainer)
        return

    def test_ai4water_mlp(self):
        model = make_mlp_model()

        exp = explain_model_with_lime(model, examples_to_explain=2)
        assert isinstance(exp, LimeExplainer)
        return

    def test_ai4water_lstm(self):
        m = lstm_model()
        m.fit()
        exp = explain_model_with_lime(m, examples_to_explain=2)
        assert isinstance(exp, LimeExplainer)


if __name__ == "__main__":

    unittest.main()
