
from typing import Union

from ...preprocessing import DataSet
from ._shap import ShapExplainer, shap
from ._lime import LimeExplainer, lime
from ..utils import choose_examples
from .utils import convert_ai4water_model, get_features
from ai4water.backend import os

def explain_model(
        model,
        data_to_expalin=None,
        train_data=None,
        total_data=None,
        features_to_explain: Union[str, list] = None,
        examples_to_explain: Union[int, float, list] = 0,
        explainer=None,
        layer: Union[str, int] = None,
        method: str = "both"
):
    """
    Explains the ai4water's Model class.

    Arguments:
        model : the AI4Water's model to explain
        features_to_explain : the input features to explain. It must be a string
            or a list of strings where a string is a  feature name.
        examples_to_explain : the examples to explain. If integer, it will be
            the number/index of example to explain. If float, it will be fraction
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

    Example:
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> from ai4water.postprocessing.explain import explain_model
        >>> model = Model(model="RandomForestRegressor")
        >>> model.fit(data=busan_beach())
        >>> explain_model(model, total_data=busan_beach())
    """
    data = {'data_to_explain': data_to_expalin,
            'train_data': train_data,
            'total_data': total_data}

    if method == 'both':

        exp1 = _explain_with_lime(model=model, examples_to_explain=examples_to_explain, **data)

        exp2 = _explain_with_shap(model,
                                  features_to_explain=features_to_explain,
                                  examples_to_explain=examples_to_explain,
                                  explainer=explainer, layer=layer, **data)
        explainer = (exp1, exp2)

    elif method == 'shap' and shap:
        explainer = _explain_with_shap(model,
                                       features_to_explain=features_to_explain,
                                       examples_to_explain=examples_to_explain,
                                       explainer=explainer, layer=layer, **data)

    elif method == 'lime' and lime:
        explainer = _explain_with_lime(model=model, examples_to_explain=examples_to_explain, **data)

    else:
        ValueError(f"unrecognized method {method}")

    return explainer


def explain_model_with_lime(
        model,
        data_to_explain=None,
        train_data=None,
        total_data=None,
        examples_to_explain: Union[int, float, list] = 0,
) -> "LimeExplainer":
    """Explains the model with LimeExplainer

    Parameters
    ----------
        data_to_explain :
            the data to explain
        train_data :
            the data used for training.
        total_data :
            total data from which training and test data will be extracted.
            This is only required if data_to_explain/train data is not given.
        model :
            the AI4Water's model to explain
        examples_to_explain :
            the examples to explain

    Returns
    -------
        an instance of [LimeExplainer][ai4water.postprocessing.explain.LimeExplainer]

    Example
    -------
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> from ai4water.postprocessing.explain import explain_model_with_lime
        >>> model = Model(model="RandomForestRegressor")
        >>> model.fit(data=busan_beach())
        >>> explain_model_with_lime(model, total_data=busan_beach())

    """

    if total_data is None:
        train_x = train_data
        test_x = data_to_explain
        test_y = None
    else:
        assert total_data is not None
        train_x, _ = model.training_data(data=total_data)
        test_x, test_y = model.test_data(data=total_data)

    features = model.input_features

    lime_exp_path = maybe_make_path(os.path.join(model.path, "explainability", "lime"))

    test_x, index = choose_examples(test_x, examples_to_explain, test_y)

    mode = model.mode
    verbosity = model.verbosity
    if model.lookback > 1:
        explainer = "RecurrentTabularExplainer"
    else:
        explainer = "LimeTabularExplainer"

    model, _, _, _ = convert_ai4water_model(model)

    if mode == "classification":
        return

    explainer = LimeExplainer(model,
                              data=test_x,
                              train_data=train_x,
                              path=lime_exp_path,
                              feature_names=features,
                              explainer=explainer,
                              mode=mode,
                              verbosity=verbosity,
                              show=False
                              )

    for i in range(explainer.data.shape[0]):
        explainer.explain_example(i, name=f"lime_exp_for_{index[i]}")

    return explainer


def explain_model_with_shap(
        model,
        data_to_explain=None,
        train_data=None,
        total_data=None,
        features_to_explain: Union[str, list] = None,
        examples_to_explain: Union[int, float, list] = 0,
        explainer=None,
        layer: Union[str, int] = None,
        plot_name="summary",
) -> "ShapExplainer":
    """Expalins the model which is built by AI4Water's Model class using SHAP.

    Parameters
    ----------
        model :
            the model to explain.
        data_to_explain :
            the data to explain.  If given, then ``train_data`` must be given as well.
            If not given then ``total_data`` must be given.
        train_data :
            the data on which model was trained. If not given, then ``total_data`` must
            be given.
        total_data :
            raw unpreprocessed data from which train and test data will be extracted.
            The explanation will be done on test data. This is only required if
            data_to_explain and train_data are not given.
        features_to_explain :
            the features to explain.
        examples_to_explain :
            the examples to explain. If integer, it will be
            the number of examples to explain. If float, it will be fraction
            of values to explain. If list/array, it will be index of examples
            to explain. The examples are choosen which have highest variance
            in prediction.
        explainer :
            the explainer to use
        layer :
            layer to explain.
        plot_name :
            name of plot to draw

    Returns
    -------
        an instance of ShapExplainer

    Examples
    --------
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> from ai4water.postprocessing.explain import explain_model_with_shap
        >>> model = Model(model="RandomForestRegressor")
        >>> model.fit(data=busan_beach())
        >>> explain_model_with_shap(model, total_data=busan_beach())
    """
    assert hasattr(model, 'path')

    if data_to_explain is None:
        assert total_data is not None
        train_x, _ = model.training_data(data=total_data)
        data_to_explain, test_y = model.test_data(data=total_data)
    else:
        assert train_data is not None
        assert data_to_explain is not None

        train_x = train_data
        data_to_explain = data_to_explain
        test_y = None

    features = model.input_features

    shap_exp_path = None
    save = True
    if model.verbosity>=0:
        shap_exp_path = maybe_make_path(os.path.join(model.path, "explainability", "shap"))
        save = False

    if not isinstance(model.dh_, DataSet):
        raise NotImplementedError

    features_to_explain = get_features(features, features_to_explain)

    model, framework, _explainer, _ = convert_ai4water_model(model)

    if framework == "DL":
        layer = layer or 2
    explainer = explainer or _explainer

    if examples_to_explain is None:
        examples_to_explain = 0

    data_to_explain, index = choose_examples(data_to_explain, examples_to_explain, test_y)

    explainer = ShapExplainer(model=model,
                              data=data_to_explain,
                              train_data=train_x,
                              explainer=explainer,
                              path=shap_exp_path,
                              framework=framework,
                              feature_names=features_to_explain,
                              layer=layer,
                              show=False,
                              save=save
                              )

    if plot_name == "all":
        for i in range(explainer.data.shape[0]):
            explainer.force_plot_single_example(i, f"force_plot_{index[i]}")

        explainer.summary_plot()
        explainer.plot_shap_values()
    else:
        explainer.plot_shap_values()

    return explainer


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


def maybe_make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
