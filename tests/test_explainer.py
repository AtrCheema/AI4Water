import os.path
import unittest

import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from ai4water import Model
from ai4water.datasets import arg_beach
from sklearn.model_selection import train_test_split
from ai4water.post_processing.explain import ShapMLExplainer, ShapDLExplainer



def fit_and_plot(model_name, heatmap=False, beeswarm_plot=False):

    model = Model(
        model=model_name,
        data=arg_beach(),
        verbosity=0
    )

    model.fit()

    x_test, y_test = model.test_data()
    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]
    interpreter = ShapMLExplainer(model._model, x_test, path=model.path,
                                  explainer="Explainer")
    if heatmap: interpreter.heatmap()
    if beeswarm_plot: interpreter.beeswarm_plot()

    interpreter = ShapMLExplainer(model._model, x_test.values, path=model.path,
                                  explainer="Explainer")
    if heatmap: interpreter.heatmap()
    if beeswarm_plot: interpreter.beeswarm_plot()

    return


def fit_and_interpret(model_name:str, draw_heatmap=True):

    model = Model(
        model=model_name,
        data=arg_beach(),
        verbosity=0
    )

    model.fit()

    x_test, y_test = model.test_data()
    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]

    interpreter = ShapMLExplainer(model._model, x_test, path=model.path)
    interpreter()

    if draw_heatmap:
        interpreter.heatmap()

    return


def fit_and_draw_plots(model_name, draw_heatmap=False):

    model = Model(
        model=model_name,
        data=arg_beach(),
        verbosity=0
    )

    model.fit()
    x_test, y_test = model.test_data()

    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]

    explainer = ShapMLExplainer(model._model, x_test, explainer="Explainer",
                                  path=os.path.join(os.getcwd(), "results"))
    explainer.waterfall_plot_all_examples()
    explainer.scatter_plot_all_features()
    if draw_heatmap:
        explainer.heatmap()

    explainer = ShapMLExplainer(model._model, x_test.values, explainer="Explainer",
                           path=os.path.join(os.getcwd(), "results"))
    explainer.waterfall_plot_all_examples()
    explainer.scatter_plot_all_features()
    #explainer.heatmap()

    return


def get_mlp():

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
    #train_x, train_y = model.training_data()
    testx, testy = model.test_data()
    testx = pd.DataFrame(testx, columns=model.dh.input_features).iloc[0:5]
    #train_x = pd.DataFrame(train_x, columns=model.dh.input_features).iloc[0:5]
    plt.rcParams.update(plt.rcParamsDefault)

    return model, testx


class TestShapExplainers(unittest.TestCase):

    def test_doc_example(self):

        X, y = shap.datasets.diabetes()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        lin_regr = linear_model.LinearRegression()
        lin_regr.fit(X_train, y_train)

        explainer = ShapMLExplainer(lin_regr, test_data=X_test.iloc[0:10],
                                        train_data=X_train,
                                        num_means=10,
                                        path=os.path.join(os.getcwd(), "results"))
        explainer(plot_force_all=True)

        #explainer.heatmap()

        return

    def test_ai4water_model(self):

        model = Model(
            model="LinearRegression",
            data=arg_beach(),
            verbosity=0
        )

        model.fit()

        x_train, y_train = model.training_data()
        x_test, y_test = model.test_data()

        x_train = pd.DataFrame(x_train, columns=model.dh.input_features)

        x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]

        explainer = ShapMLExplainer(model._model,
                                        test_data=x_test, train_data=x_train,
                                        num_means=10, path=model.path)

        explainer(plot_force_all=False)
        #explainer.heatmap()

        explainer = ShapMLExplainer(model._model,
                                             train_data=x_train.values, test_data=x_test.values,
                                             num_means=10, path=model.path)
        explainer(plot_force_all=False)
        #explainer.heatmap()
        return

    def test_raise_error(self):

        model = Model(
            model={"layers": {"LSTM":{"units": 4}}},
            data=arg_beach(),
            verbosity=0
        )

        x_test, y_test = model.test_data()

        def initiate_class():
            return ShapMLExplainer(model._model, x_test)

        self.assertRaises(ValueError,
                          initiate_class)
        return

    def test_xgboost(self):

        fit_and_interpret("XGBoostRegressor", draw_heatmap=True)

        return

    def test_lgbm(self):

        fit_and_interpret("LGBMRegressor", draw_heatmap=False)

        return

    def test_catboost(self):

        fit_and_interpret("CatBoostRegressor", draw_heatmap=False)

        return

    def test_waterfall_with_xgboost(self):

        fit_and_draw_plots("XGBoostRegressor", draw_heatmap=True)

        return

    def test_waterfall_with_catboost(self):

        fit_and_draw_plots("CatBoostRegressor")

        return

    def test_heatmap(self):

        for mod in [
            "XGBoostRegressor",
            "RandomForestRegressor",
            ##"LGBMRegressor",  # process stopping problem
            "DECISIONTREEREGRESSOR",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "GRADIENTBOOSTINGREGRESSOR",
            ##"HISTGRADIENTBOOSTINGREGRESSOR", # taking very long time
            "XGBOOSTRFREGRESSOR"
                    ]:

            fit_and_plot(mod, heatmap=True)

        return

    def test_beeswarm_plot(self):

        for mod in [
            "XGBoostRegressor",
            "RandomForestRegressor",
            "LGBMRegressor",
            "DECISIONTREEREGRESSOR",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "GRADIENTBOOSTINGREGRESSOR",
            "HISTGRADIENTBOOSTINGREGRESSOR",
            "XGBOOSTRFREGRESSOR"
                    ]:

            fit_and_plot(mod, beeswarm_plot=True)

        return

    def test_deepexplainer_mlp(self):

        model, testx = get_mlp()
        ex = ShapDLExplainer(model, testx, explainer="DeepExplainer", path=model.path)

        ex.plot_shap_values()

        return

    def test_gradientexplainer_mlp(self):

        model, testx = get_mlp()

        ex = ShapDLExplainer(model, testx, layer=1, explainer="GradientExplainer", path=model.path)
        plt.rcParams.update(plt.rcParamsDefault)
        ex.plot_shap_values()

        return


if __name__ == "__main__":

    unittest.main()
