
from ai4water.backend import sklearn_models


def convert_ai4water_model(old_model, framework=None, explainer=None):
    """convert ai4water's Model class to sklearn/xgboost..etc type model classes
    """
    new_model = old_model

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

    return new_model, framework, explainer


def to_native(model, model_name:str):
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
