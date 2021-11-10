import time
import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from ai4water import Model
from ai4water.datasets import arg_beach, MtropicsLaos
from ai4water.postprocessing.explain import ShapExplainer, explain_model_with_shap

from test_lime_explainer import make_lstm_reg_model, lstm_model, get_fitted_model, make_mlp_model

laos = MtropicsLaos()

class_data = laos.make_classification()


def get_lstm_model():
    from tensorflow.keras.models import Model as KModel
    from tensorflow.keras.layers import Flatten

    m = make_lstm_reg_model()
    train_x, _ = m.training_data()
    o = Flatten()(m.outputs[0])
    # #
    m2 = KModel(inputs=m.inputs, outputs=o)

    return m2, train_x, m.dh.input_features, m.path


def fit_and_plot(model_name, data, heatmap=False, beeswarm_plot=False):

    model = get_fitted_model(model_name, data)

    x_test, y_test = model.test_data()
    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]
    interpreter = ShapExplainer(model._model, x_test, path=model.path,
                                  explainer="Explainer")
    if heatmap: interpreter.heatmap(show=False)
    if beeswarm_plot: interpreter.beeswarm_plot(show=False)

    interpreter = ShapExplainer(model._model, x_test.values, path=model.path,
                                  explainer="Explainer")
    if heatmap: interpreter.heatmap(show=False)
    if beeswarm_plot: interpreter.beeswarm_plot(show=False)

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

    interpreter = ShapExplainer(model._model, x_test,
                                  train_data=x_train,
                                  path=model.path)
    interpreter(save=False)

    if draw_heatmap:
        interpreter.heatmap(show=False)

    explainer = ShapExplainer(model._model,
                              x_test.values,
                              train_data=x_train.values,
                              features=model.dh.input_features,
                              path=model.path)
    explainer(save=False)

    return

def get_explainer(model_name, data):
    model = get_fitted_model(model_name, data)

    x_test, y_test = model.test_data()

    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]

    explainer = ShapExplainer(model._model, x_test, explainer="Explainer",
                                  path=model.path)
    return explainer

def fit_and_draw_plots(model_name, data, draw_heatmap=False):

    model = get_fitted_model(model_name, data)

    x_test, y_test = model.test_data()

    x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]

    explainer = ShapExplainer(model._model, x_test, explainer="Explainer",
                                  path=model.path)

    explainer.waterfall_plot_all_examples(show=False)
    explainer.scatter_plot_all_features(show=False)
    if draw_heatmap:
        explainer.heatmap(show=False)

    explainer = ShapExplainer(model._model, x_test.values, explainer="Explainer",
                           path=model.path)
    explainer.waterfall_plot_all_examples(show=False)
    explainer.scatter_plot_all_features(show=False)
    #explainer.heatmap()

    return


def get_mlp():

    model = make_mlp_model()
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

        explainer = ShapExplainer(lin_regr,
                                  data=X_test.iloc[0:14],
                                  train_data=X_train,
                                  num_means=12,
                                  path=os.path.join(os.getcwd(), "results"))
        explainer(plot_force_all=True)

        explainer.heatmap(show=False)
        explainer.plot_shap_values(show=False)

        return

    def test_pd_plot(self):

        for mod in [
            "XGBRegressor", # todo error
            "RandomForestRegressor",
            "LGBMRegressor",
            "DecisionTreeRegressor",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "GradientBoostingRegressor",
            "HistGradientBoostingRegressor",
            "XGBRFRegressor" # todo error
                    ]:

            exp = get_explainer(mod, arg_beach(inputs=["pcp_mm", "air_p_hpa", "air_temp_c"]))
            exp.pdp_single_feature(feature_name=exp.features[0], show=False, save=False)

            time.sleep(1)
        return

    def test_ai4water_model(self):

        model = Model(
            model="LinearRegression",
            data=arg_beach(inputs=['wat_temp_c', 'tide_cm']),
            verbosity=0
        )

        model.fit()

        x_train, y_train = model.training_data()
        x_test, y_test = model.test_data()

        x_train = pd.DataFrame(x_train, columns=model.dh.input_features)

        x_test = pd.DataFrame(x_test, columns=model.dh.input_features).iloc[0:5]

        explainer = ShapExplainer(model._model,
                                  data=x_test, train_data=x_train,
                                        num_means=10, path=model.path)

        explainer(plot_force_all=False)
        #explainer.heatmap()

        explainer = ShapExplainer(model._model,
                                  train_data=x_train.values, data=x_test.values,
                                  num_means=10, path=model.path)
        explainer(plot_force_all=False)
        #explainer.heatmap()
        return

    def test_raise_error(self):

        model = Model(
            model={"layers": {"LSTM":{"units": 4}}},
            data=arg_beach(inputs=['wat_temp_c', 'tide_cm']),
            verbosity=0
        )

        x_test, y_test = model.test_data()

        def initiate_class():
            return ShapExplainer(model._model, x_test)

        self.assertRaises(ValueError,
                          initiate_class)
        return

    def test_xgb(self):

        fit_and_interpret("XGBRegressor", data=arg_beach(inputs=['wat_temp_c', 'tide_cm']), draw_heatmap=True)

        return

    def test_lgbm(self):

        fit_and_interpret("LGBMRegressor", data=arg_beach(inputs=['wat_temp_c', 'tide_cm']), draw_heatmap=False)

        return

    def test_catboost(self):

        fit_and_interpret("CatBoostRegressor", data=arg_beach(inputs=['wat_temp_c', 'tide_cm']), draw_heatmap=False)

        return

    def test_waterfall_with_xgb(self):

        fit_and_draw_plots("XGBRegressor", arg_beach(inputs=['wat_temp_c', 'tide_cm']), draw_heatmap=True)

        return

    def test_waterfall_with_catboost(self):

        fit_and_draw_plots("CatBoostRegressor", arg_beach(inputs=['wat_temp_c', 'tide_cm']))

        return

    def test_heatmap(self):

        for mod in [
            "XGBRegressor",
            "RandomForestRegressor",
            ##"LGBMRegressor",  # process stopping problem
            "DecisionTreeRegressor",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "GradientBoostingRegressor",
            ##"HISTGRADIENTBOOSTINGREGRESSOR", # taking very long time
            "XGBRFRegressor"
                    ]:

            fit_and_plot(mod, arg_beach(), heatmap=True)
        time.sleep(1)
        return

    def test_beeswarm_plot(self):

        for mod in [
            "XGBRegressor",
            "RandomForestRegressor",
            "LGBMRegressor",
            "DecisionTreeRegressor",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "GradientBoostingRegressor",
            "HistGradientBoostingRegressor",
            "XGBRFRegressor"
                    ]:

            fit_and_plot(mod, arg_beach(), beeswarm_plot=True)
            time.sleep(1)
        return

    def test_deepexplainer_mlp(self):

        model, testx = get_mlp()
        ex = ShapExplainer(model, testx, explainer="DeepExplainer", layer=2,
                             path=model.path)

        ex.plot_shap_values(show=False)

        return

    def test_gradientexplainer_mlp(self):

        model, testx = get_mlp()

        ex = ShapExplainer(model, testx, layer=1, explainer="GradientExplainer",
                             path=model.path)
        plt.rcParams.update(plt.rcParamsDefault)
        ex.plot_shap_values(show=False)

        return

    def test_class_model(self):

        fit_and_interpret("DecisionTreeClassifier", data=class_data, draw_heatmap=False)

        return

    def test_lstm_model_deep_exp(self):

        m, train_x, features, _path = get_lstm_model()
        #
        exp = ShapExplainer(model=m, data=train_x, layer=2, features=features, path=_path)
        exp.summary_plot(show=False)
        exp.force_plot_single_example(0, show=False)
        exp.plot_shap_values(show=False)
        return

    def test_lstm_model_gradient_exp(self):

        m, train_x, features, _path = get_lstm_model()

        exp = ShapExplainer(model=m, data=train_x, layer="LSTM", explainer="GradientExplainer",
                            features=features, path=_path)
        exp.plot_shap_values(show=False)
        exp.force_plot_single_example(0, show=False)
        return

    def test_lstm_model_ai4water(self):
        m = make_lstm_reg_model()
        train_x, _ = m.training_data()
        exp = ShapExplainer(model=m, data=train_x, layer="LSTM", explainer="GradientExplainer",
                            features=m.dh.input_features, path=m.path)
        exp.force_plot_single_example(0, show=False)
        return

    def test_ai4water_ml(self):

        for m in [
            "XGBRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
                  ]:

            model = get_fitted_model(m, arg_beach(inputs=['wat_temp_c', 'tide_cm']))
            exp = explain_model_with_shap(model, examples_to_explain=2)
            assert isinstance(exp, ShapExplainer)

        return

    def test_ai4water_mlp(self):
        model = make_mlp_model()

        exp = explain_model_with_shap(model, examples_to_explain=2)
        assert isinstance(exp, ShapExplainer)
        return

    def test_ai4water_lstm(self):
        m = lstm_model()
        m.fit()
        exp = explain_model_with_shap(m, examples_to_explain=2)
        assert isinstance(exp, ShapExplainer)
        return

    def test_plots_for_3d_input(self):
        model = lstm_model()
        test_x, _ = model.test_data()
        p = model.predict(test_x)

        exp = ShapExplainer(model, test_x, layer=2, path=model.path,
                            features=model.dh.input_features
                            )
        exp.force_plot_single_example(np.argmin(p).item(), show=False)
        exp.force_plot_single_example(np.argmax(p).item(), show=False)
        exp.waterfall_plot_single_example(np.argmin(p).item(), show=False)
        exp.waterfall_plot_single_example(np.argmax(p).item(), show=False)
        exp.pdp_all_features(lookback=0, show=False)

        return

    def test_multiple_inputs(self):
        model = Model(model={"layers": {"Input_0": {"shape": (10, 2)},
                                        "LSTM_0": {"config": {"units": 62},
                                                   "inputs": "Input_0",
                                                   "outputs": "lstm0_output"},
                                        "Input_1": {"shape": (5, 3)},
                                        "LSTM_1": {"config": {"units": 32},
                                                   "inputs": "Input_1",
                                                   "outputs": "lstm1_output"},
                                        "Concatenate": {"config": {}, "inputs": ["lstm0_output", "lstm1_output"]},
                                        "Dense": {"config": 1}
                                        }}, verbosity=0)
        test_x = [np.random.random((100, 10, 2)), np.random.random((100, 5, 3))]
        exp = ShapExplainer(model, test_x, layer="LSTM_1", path=model.path)
        exp.summary_plot(show=False)
        exp.plot_shap_values(show=False)
        return


#
# from ai4water.datasets import CAMELS_AUS
# np.set_printoptions(linewidth=200)
#
# dataset = CAMELS_AUS()
#
# inputs = ['et_morton_point_SILO',
#            'precipitation_AWAP',
#            'tmax_AWAP',
#            'tmin_AWAP',
#            'vprp_AWAP',
#            'rh_tmax_SILO',
#            'rh_tmin_SILO'
#           ]
#
# outputs = ['streamflow_MLd']
#
# data = dataset.fetch('401203', dynamic_features=inputs+outputs, as_dataframe=True,
#                      st="2000", en="2005")
# data = data.unstack()
# data.columns = [a[1] for a in data.columns.to_flat_index()]
# m_config = {"layers":
#                 {"LSTM_0": {"units": 128, "return_sequences": True},
#                  "LSTM_1": 32,
#                  "Dense": 1}}
# model = Model(model=m_config,
#               data=data,
#               lookback=5)
# test_x, _ = model.test_data()
#
# exp = ShapExplainer(model, test_x, layer=2, path=model.path,
#                     #features=model.dh.input_features
#                     )
#
# exp.pdp_single_feature('Feature 0')


if __name__ == "__main__":

    unittest.main()
