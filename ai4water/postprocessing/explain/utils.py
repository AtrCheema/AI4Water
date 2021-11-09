import os
from typing import Union

import numpy as np

from ._shap import ShapExplainer, shap
from ._lime import LimeExplainer, lime
from ..utils import choose_examples


def explain_model(
        model,
        features_to_explain: Union[str, list] = None,
        examples_to_explain: Union[int, float, list] = 20,
        explainer=None,
        layer: Union[str, int] = None,
        method:str = "both"
):
    """
    Explains the ai4water's Model class.

    Arguments:
        model : the AI4Water's model to explain
        features_to_explain : the input features to explain. It must be a string
            or a list of strings where a string is a  feature name.
        examples_to_explain : the examples to explain. If integer, it will be
            the number of examples to explain. If float, it will be fraction
            of values to explain. If list/array, it will be index of examples
            to explain. The examples are choosen which have highest variance
            in prediction.
        explainer : the explainer to use. If None, it will be inferred based
            upon the model type.
        layer : layer to explain. Only relevant if the model consits of layers
            of neural networks. If integer, it will be the number of layer
            to explain. If string, it will be name of layer of to explain.
        method : either 'both', 'shap' or 'lime'. If both, then the model will
            be explained using both lime and shap methods.
    Returns:
        if `method`==both, it will return a tuple of LimeExplainer and ShapExplainer
        otherwise it will return the instance of either LimeExplainer or ShapExplainer.

    Example
    -------
    ```python
    >>>from ai4water import Model
    >>>from ai4water.datasets import arg_beach
    >>>from ai4water.postprocessing.explain import explain_model
    >>>model = Model(model="RandForestRegressor", data=arg_beach())
    >>>model.fit()
    >>>explain_model(model)
    ```
    """
    if method == 'both':

        exp1 = _explain_with_lime(model, examples_to_explain)

        exp2 = _explain_with_shap(model, features_to_explain,
                                  examples_to_explain,  explainer, layer)
        explainer =  (exp1, exp2)

    elif method == 'shap' and shap:
        explainer =  _explain_with_shap(model, features_to_explain, examples_to_explain,
                                       explainer, layer)

    elif method == 'lime' and lime:
        explainer =  _explain_with_lime(model, examples_to_explain)

    else:
        ValueError(f"unrecognized method {method}")

    return explainer


def explain_model_with_lime(
        model,
        examples_to_explain:Union[int, float, list] = 20,
)->"LimeExplainer":
    """Explains the model with LimeExplainer

    Arguments:
        model : the AI4Water's model to explain
        examples_to_explain : the examples to explain
    Returns:
        an instance of LimeExplainer

    Example
    -------
    ```python
    >>>from ai4water import Model
    >>>from ai4water.datasets import arg_beach
    >>>from ai4water.postprocessing.explain import explain_model_with_lime
    >>>model = Model(model="RandForestRegressor", data=arg_beach())
    >>>model.fit()
    >>>explain_model_with_lime(model)
    ```
    """

    features = model.dh.input_features
    train_x, _ = model.training_data()
    test_x, test_y = model.test_data()

    lime_exp_path = maybe_make_path(os.path.join(model.path, "explainability", "lime"))

    test_x, index = choose_examples(test_x, examples_to_explain, test_y)

    mode = model.mode
    verbosity = model.verbosity
    if model.lookback > 1:
        explainer = "RecurrentTabularExplainer"
    else:
        explainer = "LimeTabularExplainer"

    if model.category == "ML":  # or model.api == "functional":
        model = model._model
    else:
        # even for functional api, we may want to change the output array to 2d
        model = make_model(model)

    if mode == "classification":
        return

    explainer = LimeExplainer(model,
                              data=test_x,
                              train_data=train_x,
                              path=lime_exp_path,
                              features=features,
                              explainer=explainer,
                              mode=mode,
                              verbosity=verbosity
                              )

    for i in range(explainer.data.shape[0]):
        explainer.explain_example(i, name=f"lime_exp_for_{index[i]}")

    return explainer


def explain_model_with_shap(
        model,
        features_to_explain:Union[str, list] = None,
        examples_to_explain:Union[int, float, list] = 20,
        explainer=None,
        layer:Union[str, int] = None,
)->"ShapExplainer":
    """Expalins the model which is built by AI4Water's Model class using SHAP.

    Arguments:
        model : the model to explain
        features_to_explain : the features to explain.
        examples_to_explain : the examples to explain. If integer, it will be
            the number of examples to explain. If float, it will be fraction
            of values to explain. If list/array, it will be index of examples
            to explain. The examples are choosen which have highest variance
            in prediction.
        explainer :
        layer : layer to explain.

    Returns:
        an instance of ShapExplainer

    Example
    -------
    ```python
    >>>from ai4water import Model
    >>>from ai4water.datasets import arg_beach
    >>>from ai4water.postprocessing.explain import explain_model_with_shap
    >>>model = Model(model="RandForestRegressor", data=arg_beach())
    >>>model.fit()
    >>>explain_model_with_shap(model)
    ```
    """
    assert hasattr(model, 'path')

    train_x, _ = model.training_data()
    test_x, test_y = model.test_data()
    features = model.dh.input_features

    shap_exp_path = maybe_make_path(os.path.join(model.path, "explainability", "shap"))

    if not model.dh.source_is_df:
        raise NotImplementedError

    features_to_explain = get_features(features, features_to_explain)

    framework = model.category

    lookback = model.lookback

    if framework == "ML":
        model = model._model
    else:
        layer = layer or 2
        if model.api == "functional":
            model = make_model(model._model)
        else:
            model = make_model(model)


    test_x, index = choose_examples(test_x, examples_to_explain, test_y)

    explainer = ShapExplainer(model=model,
                              data=test_x,
                              train_data=train_x,
                              explainer=explainer,
                              path=shap_exp_path,
                              framework=framework,
                              features=features_to_explain,
                              layer=layer
                              )

    if lookback >1:
        for i in range(explainer.data.shape[0]):
            for lb in range(lookback):
                explainer.force_plot_single_example(i, lookback=lb, name=f"force_plot_{index[i]}_{lb}")
    else:
        for i in range(explainer.data.shape[0]):
            explainer.force_plot_single_example(i, f"force_plot_{index[i]}")

    explainer.summary_plot()
    explainer.plot_shap_values()

    return explainer


def maybe_make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def make_model(model):

    class MyModel:
        def __getattr__(self, item):
            return getattr(model, item)

        def predict(self, *args, **kwargs):
            p = model.predict(*args, **kwargs)
            if isinstance(p, np.ndarray) and p.ndim == 3:
                return p.reshape(-1,)
            return p

    return MyModel()


def get_features(features, features_to_explain):

    if features_to_explain is not None:
        if isinstance(features_to_explain, str):
            features_to_explain = [features_to_explain]
    else:
        features_to_explain = features

    assert isinstance(features_to_explain, list)

    for f in features_to_explain:
        assert f in features

    return features_to_explain


def _explain_with_lime(*args, **kwargs):
    explainer = None
    if lime:
        explainer = explain_model_with_lime(*args, **kwargs)
    return explainer


def _explain_with_shap(*args, **kwargs):
    explainer = None
    if shap:
        explainer = explain_model_with_shap(*args, **kwargs)
    return explainer
