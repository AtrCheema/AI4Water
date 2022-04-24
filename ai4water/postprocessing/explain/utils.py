
from ai4water.backend import sklearn_models


def convert_ai4water_model(old_model, framework=None, explainer=None):
    """convert ai4water's Model class to sklearn/xgboost..etc type model classes
    """
    new_model = old_model

    model_name = old_model.__class__.__name__

    if old_model.__class__.__name__ == "Model" and "ai4water" in str(type(old_model)):
        # this is ai4water model class
        if old_model.category == "ML":
            model_name = list(old_model.config['model'].keys())[0]
            new_model, _explainer = to_native(old_model, model_name)
            explainer = explainer or _explainer
            framework = "ML"
        else:
            framework = "DL"
            explainer = explainer or "DeepExplainer"
            if 'functional' in str(type(old_model)):
                new_model = functional_to_keras(old_model)

    return new_model, framework, explainer, model_name


def to_native(model, model_name:str):
    # because transformations are part of Model in ai4water, and TreeExplainer
    # is based upon on tree structure, it will not consider ransformation as part of Model
    if model.config['x_transformation']or model.config['y_transformation']:
        explainer = "KernelExplainer"
    else:
        explainer = "TreeExplainer"

    if model_name.startswith("XGB"):
        import xgboost
        BaseModel = xgboost.__dict__[model_name]
    elif model_name.startswith("LGB"):
        import lightgbm
        BaseModel = lightgbm.__dict__[model_name]
    elif model_name.startswith("Cat"):
        import catboost
        BaseModel = catboost.__dict__[model_name]
    elif model_name in sklearn_models:
        BaseModel = sklearn_models[model_name]
        explainer = "KernelExplainer"

    else:
        raise ValueError

    class DummyModel(BaseModel):
        """First priority is to get attribute from ai4water's Model and then from
        the underlying library's model class."""
        def __getattribute__(self, item):
            return getattr(model, item)

        def __getattr__(self, item):
            return getattr(model._model, item)

    return DummyModel(), explainer


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


def functional_to_keras(old_model):
    """converts the model of functional api to keras model"""
    assert old_model.config['x_transformation'] is None
    assert old_model.config['y_transformation'] is None

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Flatten

    # keras model from functional api
    old_model = old_model._model

    old_m_outputs = old_model.outputs
    if isinstance(old_m_outputs, list):
        assert len(old_m_outputs) == 1
        old_m_outputs = old_m_outputs[0]

    if len(old_m_outputs.shape) > 2:  # (None, ?, ?)
        new_outputs = Flatten()(old_m_outputs)  # (None, ?)
        assert new_outputs.shape.as_list()[-1] == 1  # (None, 1)
        new_model = Model(old_model.inputs, new_outputs)

    else:  # (None, ?)
        assert old_m_outputs.shape.as_list()[-1] == 1  # (None, 1)
        new_model = old_model

    return new_model