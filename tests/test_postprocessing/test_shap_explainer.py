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
from ai4water.functional import Model as FModel
from ai4water.datasets import busan_beach, MtropicsLaos
from ai4water.postprocessing.explain import ShapExplainer, explain_model_with_shap

from test_lime_explainer import make_lstm_reg_model, lstm_model, get_fitted_model, make_mlp_model

laos = MtropicsLaos()

class_data = laos.make_classification()
beach_data = busan_beach()

# todo, do not use any transformation on y for classification problem
# todo, allowed y_transformation are only log and sqrt
# todo unable to use functional api with transformation for model explainability



def fit_and_plot(model_name, data, heatmap=True, beeswarm_plot=True):

    model = get_fitted_model(model_name, data)

    x_test, y_test = model.test_data()
    x_test = pd.DataFrame(x_test, columns=model.input_features).iloc[0:5]
    interpreter = ShapExplainer(model, x_test,
                                path=model.path,
                                show=False,
                                save=False,
                                explainer="Explainer",
                                framework="ML")
    if heatmap: interpreter.heatmap()
    if beeswarm_plot: interpreter.beeswarm_plot()

    interpreter = ShapExplainer(model, x_test.values,
                                path=model.path,
                                explainer="Explainer",
                                show=False,
                                save=False,
                                framework="ML")
    if heatmap: interpreter.heatmap()
    if beeswarm_plot: interpreter.beeswarm_plot()

    return


def fit_and_interpret(model_name:str,
                      data,
                      draw_heatmap=True,
                      draw_beeswarm=True,
                      summary_plot=True,
                      waterfall=True,
                      scatter=True,
                      **kwargs
                      ):

    model = get_fitted_model(model_name, data)

    x_train, y_train = model.training_data()
    x_train = pd.DataFrame(x_train, columns=model.input_features).iloc[0:11]

    x_test, y_test = model.test_data()
    x_test = pd.DataFrame(x_test, columns=model.input_features).iloc[0:2]

    explainer = ShapExplainer(model, x_test,
                                train_data=x_train,
                                path=model.path,
                                framework="ML",
                                save=False,
                                show=False,
                                **kwargs
                                )
    explainer()

    if draw_heatmap:
        explainer.heatmap()
    if draw_beeswarm:
        explainer.beeswarm_plot()
    if summary_plot:
        explainer.summary_plot(plot_type="dot")
    if waterfall:
        explainer.waterfall_plot_single_example(0)
    if scatter:
        explainer.scatter_plot_single_feature(0)

    explainer = ShapExplainer(model,
                              x_test.values,
                              train_data=x_train.values,
                              feature_names=model.input_features,
                              path=model.path,
                              framework="ML",
                                save=False,
                                show=False,
                              **kwargs
                              )
    assert isinstance(explainer.data, pd.DataFrame)

    return


def get_explainer(model_name, data):
    model = get_fitted_model(model_name, data)

    x_test, y_test = model.test_data()

    x_test = pd.DataFrame(x_test, columns=model.input_features).iloc[0:5]

    explainer = ShapExplainer(model, x_test,
                              explainer="Explainer",
                                save=False,
                                show=False,
                                  path=model.path)
    return explainer


def fit_and_draw_plots(model_name, data, draw_heatmap=True):

    model = get_fitted_model(model_name, data)

    x_test, y_test = model.test_data()

    x_test = pd.DataFrame(x_test, columns=model.input_features).iloc[0:5]

    explainer = ShapExplainer(model, x_test,
                              explainer="Explainer",
                                save=False,
                                show=False,
                              path=model.path, framework="ML")

    explainer.waterfall_plot_all_examples()
    explainer.scatter_plot_all_features()
    if draw_heatmap:
        explainer.heatmap()

    explainer = ShapExplainer(model, x_test.values,
                              explainer="Explainer",
                              save=False,
                              show=False,
                           path=model.path, framework="ML")
    explainer.waterfall_plot_all_examples()
    explainer.scatter_plot_all_features()
    #explainer.heatmap()

    return


def get_mlp():

    model = make_mlp_model()
    #train_x, train_y = model.training_data()
    testx, testy = model.test_data()
    testx = pd.DataFrame(testx, columns=model.input_features).iloc[0:5]
    #train_x = pd.DataFrame(train_x, columns=model.input_features).iloc[0:5]
    plt.rcParams.update(plt.rcParamsDefault)

    return model, testx


def get_dl_fmodel_for_multi_inputs():
    model = FModel(
        model={"layers": {
            "Input": {"shape": (3, 3, 8)},
            "Dense": 1,
            "Flatten": {},
            "Dense_1": 1,

            "Input_1": {"shape": (2162, 2)},
            "Dense_2": 1,
            "Flatten_1": {},
            "Dense_3": 1,

            "Concatenate": {"config": {},
                            "inputs": ["Dense_1", "Dense_3"]},
            "Dense_4": 1
        }},
        verbosity=0
    )
    inp1 = np.random.random((100, 3, 3, 8))
    inp2 = np.random.random((100, 2162, 2))
    return model, [inp1, inp2]


class TestShapExplainers(unittest.TestCase):

    def test_doc_example(self):

        X, y = shap.datasets.diabetes()
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=0)
        lin_regr = linear_model.LinearRegression()
        lin_regr.fit(X_train, y_train)

        explainer = ShapExplainer(lin_regr,
                                  data=X_test.iloc[0:14],
                                  train_data=X_train,
                                  num_means=12,
                                save=False,
                                show=False,
                                  path=os.path.join(os.getcwd(), "results"))
        explainer(plot_force_all=True)

        explainer.heatmap()
        explainer.plot_shap_values()

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

            exp = get_explainer(mod,
                                busan_beach(inputs=["pcp_mm", "air_p_hpa", "air_temp_c"]))
            exp.pdp_single_feature(feature_name=exp.features[0])

            time.sleep(1)
        return

    def test_ai4water_model(self):

        model = Model(
            model="LinearRegression",
            verbosity=0
        )

        model.fit(data=busan_beach(inputs=['wat_temp_c', 'tide_cm']))

        x_train, y_train = model.training_data()
        x_test, y_test = model.test_data()

        x_train = pd.DataFrame(x_train, columns=model.input_features)

        x_test = pd.DataFrame(x_test, columns=model.input_features).iloc[0:5]

        explainer = ShapExplainer(model,
                                  data=x_test, train_data=x_train,
                                  num_means=10, path=model.path,
                                save=False,
                                show=False,
                                  explainer="KernelExplainer")

        explainer(plot_force_all=False)
        explainer.heatmap()

        explainer = ShapExplainer(model,
                                  train_data=x_train.values, data=x_test.values,
                                  num_means=10, path=model.path,
                                save=False,
                                show=False,
                                  explainer="KernelExplainer")
        explainer(plot_force_all=False)
        explainer.heatmap()
        return

    def test_xgb(self):

        fit_and_interpret("XGBRegressor",
                          data=busan_beach(inputs=['wat_temp_c','tide_cm']),
                          draw_heatmap=True,
                          explainer="TreeExplainer")

        return

    def test_lgbm(self):

        fit_and_interpret("LGBMRegressor",
                          data=busan_beach(inputs=['wat_temp_c', 'tide_cm']),
                          draw_heatmap=True,
                          explainer="TreeExplainer")

        return

    def test_catboost_tree(self):

        fit_and_interpret("CatBoostRegressor",
                          data=busan_beach(inputs=['wat_temp_c', 'tide_cm']),
                          explainer="TreeExplainer")

        return

    def test_catboost_kernel(self):

        fit_and_interpret("CatBoostRegressor",
                          data=busan_beach(inputs=['wat_temp_c', 'tide_cm']),
                          explainer="KernelExplainer")

        return

    def test_waterfall_with_xgb(self):

        fit_and_draw_plots("XGBRegressor",
                           busan_beach(inputs=['wat_temp_c','tide_cm']),
                           draw_heatmap=True)

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

            fit_and_plot(mod, beach_data, heatmap=True)
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

            fit_and_plot(mod, beach_data, beeswarm_plot=True)
            time.sleep(1)
        return

    def test_ai4water_ml(self):

        data = busan_beach(inputs=['wat_temp_c', 'tide_cm'])
        for m in [
            "XGBRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
                  ]:

            model = get_fitted_model(m, data)
            exp = explain_model_with_shap(model, total_data=data,
                                          explainer="TreeExplainer")
            assert isinstance(exp, ShapExplainer)

        return

    def test_class_model(self):

        fit_and_interpret("DecisionTreeClassifier", data=class_data,
                          draw_heatmap=False,
                          draw_beeswarm=False,
                          explainer="KernelExplainer")
        return


class TestDL(unittest.TestCase):

    def test_raise_error(self):

        model = Model(
            model={"layers": {"LSTM":{"units": 4}}},
            input_features=['wat_temp_c', 'tide_cm'],
            output_features=['tetx_coppml', "ecoli", "16s", "inti1"],
            verbosity=0,
            ts_args={'lookback':15},
        )
        model.fit(data=busan_beach(inputs=['wat_temp_c', 'tide_cm'],
                                 target=['tetx_coppml', "ecoli", "16s",
                                         "inti1"]))

        x_test, y_test = model.test_data()

        def initiate_class():
            return ShapExplainer(model, x_test)

        # self.assertRaises(AssertionError,
        #                   initiate_class)
        return

    def test_deepexplainer_mlp(self):

        model, testx = get_mlp()
        ex = ShapExplainer(model, testx, explainer="DeepExplainer",
                           layer=2,
                                save=False,
                                show=False,
                             path=model.path)

        ex.plot_shap_values()

        return

    def test_gradientexplainer_mlp(self):

        model, testx = get_mlp()

        ex = ShapExplainer(model, testx, layer=1,
                           explainer="GradientExplainer",
                                save=False,
                                show=False,
                             path=model.path)
        plt.rcParams.update(plt.rcParamsDefault)
        ex.plot_shap_values()

        return

    def test_lstm_model_deep_exp(self):

        m = make_lstm_reg_model()
        train_x, _ = m.training_data()
        exp = ShapExplainer(model=m, data=train_x, layer=2,
                            feature_names=m.input_features,
                                save=False,
                                show=False,
                            path=m.path)
        exp.summary_plot()
        exp.force_plot_single_example(0)
        exp.plot_shap_values()
        return

    def test_lstm_model_gradient_exp(self):

        m = make_lstm_reg_model()
        train_x, _ = m.training_data()

        exp = ShapExplainer(model=m, data=train_x, layer="LSTM",
                            explainer="GradientExplainer",
                                save=False,
                                show=False,
                            feature_names=m.input_features, path=m.path)
        exp.plot_shap_values()
        exp.force_plot_single_example(0)
        return
#
    def test_lstm_model_ai4water(self):
        time.sleep(1)
        m = make_lstm_reg_model()
        train_x, _ = m.training_data()
        exp = ShapExplainer(model=m, data=train_x, layer="LSTM",
                            explainer="GradientExplainer",
                                save=False,
                                show=False,
                            feature_names=m.input_features, path=m.path)
        exp.force_plot_single_example(0)
        return


    def test_ai4water_mlp(self):
        time.sleep(1)
        model = make_mlp_model()

        exp = explain_model_with_shap(model, total_data=busan_beach(inputs=['wat_temp_c', 'tide_cm']))
        assert isinstance(exp, ShapExplainer)
        return

    def test_ai4water_lstm(self):
        m = lstm_model()

        exp = explain_model_with_shap(m, total_data=busan_beach(inputs=['wat_temp_c', 'tide_cm']))
        assert isinstance(exp, ShapExplainer)
        return

    def test_plots_for_3d_input(self):
        model = lstm_model()
        test_x, _ = model.test_data()
        p = model.predict(test_x)

        exp = ShapExplainer(model, test_x, layer=2, path=model.path,
                                save=False,
                                show=False,
                            feature_names=model.input_features)
        exp.force_plot_single_example(np.argmin(p).item())
        exp.force_plot_single_example(np.argmax(p).item())
        exp.waterfall_plot_single_example(np.argmin(p).item())
        exp.waterfall_plot_single_example(np.argmax(p).item())
        exp.pdp_all_features(lookback=0)

        return

    def test_multiple_inputs(self):
        model = Model(model={"layers": {
            "Input_0": {"shape": (10, 2)},
            "LSTM_0": {"config": {"units": 62},
                       "inputs": "Input_0",
                       "outputs": "lstm0_output"},
            "Input_1": {"shape": (5, 3)},
            "LSTM_1": {"config": {"units": 32},
                       "inputs": "Input_1",
                       "outputs": "lstm1_output"},
            "Concatenate": {"config": {},
                            "inputs": ["lstm0_output", "lstm1_output"]},
            "Dense": {"config": 1}
        }}, verbosity=0)
        test_x = [np.random.random((100, 10, 2)), np.random.random((100, 5, 3))]
        exp = ShapExplainer(model, test_x, layer="LSTM_1",
                                save=False,
                                show=False,
                            path=model.path)
        exp.summary_plot()
        exp.plot_shap_values()
        return

    def test_functional_model_gradient(self):
        """tests functional Model with multi inputs"""
        model, inputs = get_dl_fmodel_for_multi_inputs()
        expl = ShapExplainer(model, data=inputs,
                                save=False,
                                show=False,
                             train_data=inputs,
                             explainer="GradientExplainer")
        sv = expl.shap_values
        assert isinstance(sv, list)
        return

    def test_functional_model_deep(self):
        """tests functional Model with multi inputs"""
        model, inputs = get_dl_fmodel_for_multi_inputs()
        expl = ShapExplainer(model, data=inputs,
                                save=False,
                                show=False,
                             train_data=inputs,
                             explainer="DeepExplainer")
        sv = expl.shap_values
        assert isinstance(sv, list)
        return


if __name__ == "__main__":

    unittest.main()
