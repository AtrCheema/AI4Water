import time
import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from ai4water import Model
from ai4water.datasets import arg_beach, MtropicsLaos
from sklearn.model_selection import train_test_split
from ai4water.post_processing.explain import ShapMLExplainer, ShapDLExplainer

laos = MtropicsLaos()

class_data = laos.make_classification()


def get_fitted_model(model_name, data):

    model = Model(
        model=model_name,
        data=data,
        verbosity=0
    )

    model.fit()

    return model


def fit_and_plot(model_name, data, heatmap=False, beeswarm_plot=False):

    model = get_fitted_model(model_name, data)

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


def fit_and_interpret(model_name:str,
                      data,
                      draw_heatmap=True,
                      ):

    model = get_fitted_model(model_name, data)

    x_train, y_train = model.training_data()
    x_train = pd.DataFrame(x_train, columns=model.dh.input_features).iloc[0:11]

    x_test, y_test = model.test_data()
    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:2]

    interpreter = ShapMLExplainer(model._model, x_test,
                                  train_data=x_train,
                                  path=model.path)
    interpreter()

    if draw_heatmap:
        interpreter.heatmap()

    interpreter = ShapMLExplainer(model._model,
                                  x_test.values,
                                  train_data=x_train.values,
                                  path=model.path)
    interpreter()

    return


def fit_and_draw_plots(model_name, data, draw_heatmap=False):

    model = get_fitted_model(model_name, data)

    x_test, y_test = model.test_data()

    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]

    explainer = ShapMLExplainer(model._model, x_test, explainer="Explainer",
                                  path=model.path)
    explainer.waterfall_plot_all_examples()
    explainer.scatter_plot_all_features()
    if draw_heatmap:
        explainer.heatmap()

    explainer = ShapMLExplainer(model._model, x_test.values, explainer="Explainer",
                           path=model.path)
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

        fit_and_interpret("XGBoostRegressor", data=arg_beach(), draw_heatmap=True)

        return

    def test_lgbm(self):

        fit_and_interpret("LGBMRegressor", data=arg_beach(), draw_heatmap=False)

        return

    def test_catboost(self):

        fit_and_interpret("CatBoostRegressor", data=arg_beach(), draw_heatmap=False)

        return

    def test_waterfall_with_xgboost(self):

        fit_and_draw_plots("XGBoostRegressor", arg_beach(), draw_heatmap=True)

        return

    def test_waterfall_with_catboost(self):

        fit_and_draw_plots("CatBoostRegressor", arg_beach())

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

            fit_and_plot(mod, arg_beach(), heatmap=True)
        time.sleep(1)
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

            fit_and_plot(mod, arg_beach(), beeswarm_plot=True)
            time.sleep(1)
        return

    def test_deepexplainer_mlp(self):

        model, testx = get_mlp()
        ex = ShapDLExplainer(model, testx, explainer="DeepExplainer",
                             path=model.path)

        ex.plot_shap_values()

        return

    def test_gradientexplainer_mlp(self):

        model, testx = get_mlp()

        ex = ShapDLExplainer(model, testx, layer=1, explainer="GradientExplainer",
                             path=model.path)
        plt.rcParams.update(plt.rcParamsDefault)
        ex.plot_shap_values()

        return

    def test_class_model(self):

        fit_and_interpret("DecisionTreeClassifier", data=class_data, draw_heatmap=False)

        return

if __name__ == "__main__":

    unittest.main()
