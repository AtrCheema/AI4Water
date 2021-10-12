import os
from typing import Union, Callable

try:
    import shap
except ModuleNotFoundError:
    shap = None

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    import tensorflow.keras.backend as K
except ModuleNotFoundError:
    Flatten, Model, K = None, None, None

from ai4water.utils.utils import axis_imshow
from ai4water.backend import get_sklearn_models
from ._explain import ExplainerMixin


class ShapExplainer(ExplainerMixin):
    """
    Wrapper around SHAP `explainers` and `plots` to draw and save all the plots
    for a given model.

    Attributes
    -------
    - features
    - train_summary : only for KernelExplainer
    - explainer
    - shap_values

    Methods
    --------
    - summary_plot
    - force_plot_single_example
    - dependence_plot_single_feature
    - force_plot_all

    Example
    --------
    ```python
    >>>from ai4water.post_processing.explain import ShapExplainer
    >>>from sklearn.model_selection import train_test_split
    >>>from sklearn import linear_model
    >>>import shap

    >>>X,y = shap.datasets.diabetes()
    >>>X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    >>>lin_regr = linear_model.LinearRegression()
    >>>lin_regr.fit(X_train, y_train)
    >>>explainer = ShapExplainer(lin_regr, X_test, X_train, num_means=10)
    >>>explainer()
    ```
    """

    allowed_explainers = [
        "Explainer",
        "DeepExplainer",
        "TreeExplainer",
        "KernelExplainer",
        "LinearExplainer",
        "AdditiveExplainer",
        "GPUTreeExplainer",
        "GradientExplainer"
    ]

    def __init__(
            self,
            model,
            test_data: Union[np.ndarray, pd.DataFrame],
            train_data: Union[np.ndarray, pd.DataFrame] = None,
            explainer: Union[str, Callable] = None,
            num_means: int = 10,
            path: str = os.getcwd(),
            features: list = None,
            framework: str = None,
            layer: Union[int, str] = None,
    ):
        """

        Args:
            model: an sklearn/xgboost/catboost/LightGBM Model/regressor
            test_data: Data on which to make interpretation. Its dimension should be
                same as that of training data.
            train_data: training data. It is used to get train_summary. It can a numpy array
                or a pandas DataFrame. Only required for scikit-learn based models.
            explainer : the explainer to use. If not given, the explainer will be inferred.
            num_means: Numher of means, used in `shap.kmeans` to calculate train_summary
            path: path to save the plots. By default, plots will be saved in current
                working directory
            features: Names of features. Should only be given if train/test data is numpy
                array.
            framework : either "DL" or "ML", where "DL" represents deep learning or neural
                network based models and "ML" represents other models. For "DL" the explainer
                will be eihter "DeepExplainer" or "GradientExplainer". If not given, it will
                be inferred but "DeepExplainer" will be prioritized over "GradientExplainer".
            layer : only relevant when framework is "DL" i.e when the model consits of layers
                of neural networks.

        """
        test_data = maybe_to_dataframe(test_data, features)
        train_data = maybe_to_dataframe(train_data, features)

        super(ShapExplainer, self).__init__(path=path, data=test_data, features=features)

        if train_data is None:
            self._check_data(test_data)
        else:
            self._check_data(train_data, test_data)

        self.is_sklearn = True
        if model.__class__.__name__.upper() not in get_sklearn_models():
            if model.__class__.__name__ in ["XGBRegressor",
                                            "LGBMRegressor",
                                            "CatBoostRegressor",
                                            "XGBRFRegressor"
                                            ]:
                self.is_sklearn = False
            elif not self._is_dl(model):
                raise ValueError(f"{model.__class__.__name__} is not a valid model model")

        self._framework = self.infer_framework(model, framework, layer, explainer)

        if self._framework == "DL" and infer_framework(model) == "tensorflow":
            self.model = make_keras_model(model)
        else:
            self.model = model
        self.data = test_data
        self.layer = layer
        self.features = features

        self.explainer = self._get_explainer(explainer, train_data=train_data, num_means=num_means)

        self.shap_values = self.get_shap_values(test_data)

    @staticmethod
    def _is_dl(model):
        if hasattr(model, "count_params") or hasattr(model, "named_parameters"):
            return True
        return False

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, x):
        if x is not None:

            if not isinstance(x, str):
                assert isinstance(x, int), f"layer must either b string or integer"
                assert x <= len(self.model.layers)  # todo, what about pytorch

        self._layer = x

    def map2layer(self, x, layer):
        feed_dict = dict(zip([self.model.layers[0].input], [x.copy()]))
        import tensorflow as tf
        if int(tf.__version__[0])<2:
            sess = K.get_session()
        else:
            sess = tf.compat.v1.keras.backend.get_session()

        if isinstance(layer, int):
            return sess.run(self.model.layers[layer].input, feed_dict)
        else:
            return sess.run(self.model.get_layer(layer).input, feed_dict)

    def infer_framework(self, model, framework, layer, explainer):
        if framework is not None:
            inf_framework = framework
        elif self._is_dl(model):
            inf_framework = "DL"
        elif isinstance(explainer, str) and explainer in ("DeepExplainer", "GradientExplainer"):
            inf_framework = "DL"
        elif explainer.__class__.__name__ in ("DeepExplainer", "GradientExplainer"):
            inf_framework = "DL"
        else:
            inf_framework = "ML"

        assert inf_framework in ("ML", "DL")

        if inf_framework == "DL":
            assert layer is not None, f"Inferred framework is {inf_framework}." \
                                      f"Therefore, you must define layer to explain"
        else:
            assert layer is None

        if inf_framework == "DL" and isinstance(explainer, str):
            assert explainer in ("DeepExplainer", "GradientExplainer"), f"invalid explainer {inf_framework}"

        return inf_framework

    def _get_explainer(self,
                       explainer: Union[str, Callable],
                       num_means,
                       train_data
                       ):

        if explainer is not None:
            if callable(explainer):
                return explainer

            assert isinstance(explainer, str), f"explainer should be callable or string but" \
                                               f" it is {explainer.__class__.__name__}"
            assert explainer in self.allowed_explainers, f"{explainer} is not a valid explainer"

            if explainer == "KernelExplainer":
                explainer = self._get_kernel_explainer(train_data, num_means)

            elif explainer == "DeepExplainer":
                explainer = self._get_deep_explainer()

            elif explainer == "GradientExplainer":
                explainer = self._get_gradient_explainer()

            else:
                explainer = getattr(shap, explainer)(self.model)

        else:  # explainer is not given explicitly, we need to infer it
            explainer = self._infer_explainer_to_use(train_data, num_means)

        return explainer

    def _get_kernel_explainer(self, data, num_means):

        assert isinstance(num_means,
                          int), f'num_means should be integer but given value is of type {num_means.__class__.__name__}'

        if data is None:
            raise ValueError("Provide train_data in order to use KernelExplainer.")

        self.train_summary = shap.kmeans(data, num_means)
        explainer = shap.KernelExplainer(self.model.predict, self.train_summary)

        return explainer

    def _infer_explainer_to_use(self, train_data, num_means):
        """Tries to infer explainer to use from the type of model."""
        # todo, Fig 3 of Lundberberg's Nature MI paper shows that TreeExplainer
        # performs better than KernelExplainer, so try to use supports_model_with_masker

        if self.model.__class__.__name__ in ["XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
                                             "XGBRFRegressor"]:
            explainer = shap.TreeExplainer(self.model)

        elif self.model.__class__.__name__.upper() in get_sklearn_models():
            explainer = self._get_kernel_explainer(train_data, num_means)

        elif self._framework == "DL":
            explainer = self._get_deep_explainer()
        else:
            raise ValueError(f"Can not infer explainer for model {self.model.__class__.__name__}."
                             f" Plesae specify explainer by using `explainer` keyword argument")
        return explainer

    def _get_deep_explainer(self):
        data = self.data.values if isinstance(self.data, pd.DataFrame) else self.data
        return getattr(shap, "DeepExplainer")(self.model, data)

    def _get_gradient_explainer(self):

        if isinstance(self.layer, int):
            return shap.GradientExplainer((self.model.layers[self.layer].input, self.model.layers[-1].output),
                                          self.map2layer(self.data, self.layer))
        else:
            return shap.GradientExplainer((self.model.get_layer(self.layer).input, self.model.layers[-1].output),
                                          self.map2layer(self.data, self.layer))

    def _check_data(self, *data):
        for d in data:
            assert type(d) == np.ndarray or type(d) == pd.DataFrame, f"" \
                                                                     f"data must be numpy array or pandas dataframe " \
                                                                     f"but it is of type {d.__class__.__name__}"

        assert len(set([d.ndim for d in data])) == 1, "train and test data should have same ndim"
        assert len(set([d.shape[-1] for d in data])) == 1, "train and test data should have same input features"
        assert len(set([type(d) for d in data])) == 1, "train and test data should be of same type"

        return

    def get_shap_values(self, data, **kwargs):
        if self._framework == "DL":
            return self._shap_values_dl(data, **kwargs)
        return self.explainer.shap_values(data)

    def _shap_values_dl(self, data, ranked_outputs=None, **kwargs):
        """Gets the SHAP values"""
        data = data.values if isinstance(data, pd.DataFrame) else data
        if self.explainer.__class__.__name__ == "Deep":
            return self.explainer.shap_values(data)
        else:

            shap_values = self.explainer.shap_values(self.map2layer(data, self.layer),
                                                     ranked_outputs=ranked_outputs, **kwargs)
            if ranked_outputs:
                shap_values, indexes = shap_values

            return shap_values

    def __call__(self,
                 force_plots=True,
                 plot_force_all=False,
                 dependence_plots=False,
                 beeswarm_plots=False,
                 heatmap=False,
                 ):
        """Draws and saves all the plots for a given sklearn model in the path.

        plot_force_all is set to False by default because it is causing
        Process finished error due. Avoiding this error is a complex function
        of scipy and numba versions.
        """

        if dependence_plots:
            for feature in self.features:
                self.dependence_plot_single_feature(feature, f"dependence_plot_{feature}")

        if force_plots:
            for i in range(self.data.shape[0]):

                self.force_plot_single_example(i, f"force_plot_{i}")

        if beeswarm_plots:
            self.beeswarm_plot()

        if plot_force_all:
            self.force_plot_all("force_plot_all")

        if heatmap:
            self.heatmap()

        self.summary_plot("summary_plot")

        return

    def summary_plot(self, name="summary_plot", show=False, **kwargs):
        """Plots the summary plot of SHAP package."""

        def _summary_plot(_shap_val, _data, _name):
            plt.close('all')
            shap.summary_plot(_shap_val, _data, show=show, feature_names=self.features, **kwargs)
            plt.savefig(os.path.join(self.path, _name), dpi=300, bbox_inches="tight")

            shap.summary_plot(_shap_val, _data, show=show, plot_type="bar", feature_names=self.features,
                              **kwargs)
            plt.savefig(os.path.join(self.path, _name + " _bar"), dpi=300, bbox_inches="tight")

            return

        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        data = self.data

        if data.ndim == 3:
            assert shap_vals.ndim == 3

            for lookback in range(data.shape[1]):

                _summary_plot(shap_vals[:, lookback], data[:, lookback],  _name=f"{name}_{lookback}")
        else:
            _summary_plot(shap_vals, data, name)

        return

    def force_plot_single_example(self, idx:int, name=None, lookback=None, show=False):
        """Draws force_plot for a single example/row/sample/instance/data point.

        Arguments:
            idx : index of exmaple
            name : name of saved file
            show : whether to show the plot or not
            lookback : only relevent if input data is 3d
        """

        shap_vals = self.shap_values

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = shap_vals[idx]

        if type(self.data) == np.ndarray:
            data = self.data[idx]
        else:
            data = self.data.iloc[idx, :]

        if self.explainer.__class__.__name__ == "Gradient":
            expected_value = [0]
        else:
            expected_value = self.explainer.expected_value

        if data.ndim == 2:  # input is 3d
            assert lookback is not None, f"for 3d input data, lookback must not be None"
            assert shap_vals.ndim == 2
            expected_value =  expected_value[0] #todo
            shap_vals = shap_vals[lookback]
            data = None

        plt.close('all')

        plotter = shap.force_plot(
            expected_value,
            shap_vals,
            data,
            feature_names=self.features,
            show=False,
            matplotlib=True)

        name = name or f"force_plot_{idx}_{lookback}"
        plotter.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")

        if show:
            plotter.show()

        return

    def dependence_plot_single_feature(self, feature, name="dependence_plot"):
        """dependence plot for a single feature."""
        plt.close('all')

        if len(name) > 150:  # matplotlib raises error if the length of filename is too large
            name = name[0:150]

        shap.dependence_plot(feature, self.shap_values, self.data, show=False)
        plt.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")
        return

    def force_plot_all(self, name="force_plot.html"):
        """draws force plot for all examples in the given data and saves it in an html"""

        # following scipy versions cause kernel stoppage when calculating
        if sp.__version__ in ["1.4.1", "1.5.2", "1.7.1"]:
            print(f"force plot can not be plotted for scipy version {sp.__version__}. Please change your scipy")
            return
        plt.close('all')
        plot = shap.force_plot(self.explainer.expected_value, self.shap_values, self.data)
        shap.save_html(os.path.join(self.path, name), plot)

        return

    def waterfall_plot_all_examples(self, name: str = "waterfall"):
        """Plots the waterfall plot of SHAP package

        It plots for all the examples/instances from test_data.
        """
        for i in range(len(self.data)):
            self.waterfall_plot_single_example(i, name=name)

        return

    def waterfall_plot_single_example(self, example_index: int, name: str = "waterfall", show: bool = False):
        """draws and saves waterfall plot for one prediction.

        Currently only possible with xgboost, catboost, lgbm models
        show : if False, the plot will be saved otherwise drawn.
        """
        shap_vals = self.explainer(self.data)
        plt.close('all')
        shap.plots.waterfall(shap_vals[example_index], show=show)
        if not show:
            plt.savefig(os.path.join(self.path, f"{name}_{example_index}"), dpi=300, bbox_inches="tight")

        return

    def scatter_plot_single_feature(self, feature: str, name: str = "scatter", show=False):

        shap_values = self.explainer(self.data)
        shap.plots.scatter(shap_values[:, feature], show=show)
        if not show:
            plt.savefig(os.path.join(self.path, f"{name}_{feature}"), dpi=300, bbox_inches="tight")

        return

    def scatter_plot_all_features(self, name="scatter_plot"):
        """draws scatter plot for all features"""
        if isinstance(self.data, pd.DataFrame):
            features = self.features
        else:
            features = [i for i in range(self.data.shape[-1])]

        for feature in features:
            self.scatter_plot_single_feature(feature, name=name)

        return

    def heatmap(self, name: str = 'heatmap', show: bool = False, max_display=10):
        """Plots the heat map and saves it
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html
        This can be drawn for xgboost/lgbm as well as for randomforest type models
        but not for CatBoostRegressor which is todo.
        """

        # if heat map is drawn with np.ndarray, it throws errors therefore convert
        # it into pandas DataFrame. It is more interpretable and does not hurt.
        shap_values = self._get_shap_values_locally()

        self._heatmap(shap_values, f"{name}_basic", show=show, max_display=max_display)

        # sort by the maximum absolute value of a feature over all the samples
        self._heatmap(shap_values, f"{name}_sortby_maxabs", show=show, max_display=max_display,
                      feature_values=shap_values.abs.max(0))

        # sorting by the sum of the SHAP values over all features gives a complementary perspective on the data
        self._heatmap(shap_values, f"{name}_sortby_SumOfShap", show=show, max_display=max_display,
                      instance_order=shap_values.sum(1))
        return

    def _heatmap(self, shap_values, name, show=False, max_display=10,  **kwargs):
        plt.close('all')

        shap.plots.heatmap(shap_values, show=show,  max_display=max_display, **kwargs)

        if not show:
            plt.savefig(os.path.join(self.path, f"{name}_sortby_SumOfShap"), dpi=300, bbox_inches="tight")
        return

    def _get_shap_values_locally(self):
        data = self.data
        if not isinstance(self.data, pd.DataFrame):
            data = pd.DataFrame(self.data, columns=self.features)

        # not using global explainer because, this explainer should data as well
        explainer = shap.Explainer(self.model, data)
        shap_values = explainer(data)

        return shap_values

    def beeswarm_plot(self, name: str = "beeswarm", show=False, max_display: int = 10):
        """
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html
        """

        shap_values = self._get_shap_values_locally()

        self._beeswarm_plot(shap_values, name=f"{name}_basic", show=show, max_display=max_display)

        # find features with high impacts
        self._beeswarm_plot(shap_values, name=f"{name}_sortby_maxabs", show=show, max_display=max_display,
                            order=shap_values.abs.max(0))

        # plot the absolute value
        self._beeswarm_plot(shap_values.abs, name=f"{name}_abs_shapvalues", show=show, max_display=max_display)

        return

    def _beeswarm_plot(self, shap_values, name, show=False, max_display=10, **kwargs):

        plt.close('all')
        shap.plots.beeswarm(shap_values,  show=show, max_display=max_display, **kwargs)
        if not show:
            plt.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")

        return

    def decision_plot(self, name: str = "decision_"):
        # https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba
        raise NotImplementedError

    def plot_shap_values(self, name: str = "shap_values", show: bool = False,
                         interpolation=None, cmap="coolwarm"):
        """Plots the SHAP values."""
        shap_values = self.shap_values

        if isinstance(shap_values, list):
            shap_values: np.ndarray = shap_values[0]

        if self.data.ndim == 3: # input is 3d
            assert shap_values.ndim == 3
            return imshow_3d(shap_values, self.data, self.features, self.path, show=show,
                             cmap=cmap)

        plt.close('all')
        fig, axis = plt.subplots()
        axis.imshow(shap_values, aspect='auto', interpolation=interpolation, cmap=cmap)
        axis.set_xticklabels(self.features, rotation=90)
        axis.set_xlabel("Features")
        axis.set_ylabel("Examples")

        plt.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return


def imshow_3d(values,
              data,
              feature_names:list,
              path, vmin=None, vmax=None, name="shap_values",
              show=False,
              cmap=None):

    num_examples, lookback, input_features = values.shape
    assert data.shape == values.shape

    for idx, feat in enumerate(feature_names):
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2, sharex='all', figsize=(10, 12))

        axis, im = axis_imshow(ax1, data[:, :, idx].transpose(), lookback, vmin, vmax,
                               title=feat, cmap=cmap)
        fig.colorbar(im, ax=axis, orientation='vertical', pad=0.2)

        axis, im = axis_imshow(ax2, values[:, :, idx].transpose(), lookback, vmin, vmax,
                               xlabel="Examples", title=f"SHAP Values", cmap=cmap)

        fig.colorbar(im, ax=axis, orientation='vertical', pad=0.2)

        _name = f'{name}_{feat}'
        plt.savefig(os.path.join(path, _name), dpi=400, bbox_inches='tight')

        if show:
            plt.show()

    return


def infer_framework(model):
    if hasattr(model, 'config') and 'backend' in model.config:
        framework = model.config['backend']

    elif type(model) is tuple:
        a, _ = model
        try:
            a.named_parameters()
            framework = 'pytorch'
        except:
            framework = 'tensorflow'
    else:
        try:
            model.named_parameters()
            framework = 'pytorch'
        except:
            framework = 'tensorflow'

    return framework


def make_keras_model(old_model):

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


def maybe_to_dataframe(data, features=None) -> pd.DataFrame:
    if isinstance(data, np.ndarray) and isinstance(features, list) and data.ndim == 2:
        data = pd.DataFrame(data, columns=features)
    return data
