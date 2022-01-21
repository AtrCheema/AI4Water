import os
from typing import Union, Callable, List

try:
    import shap
except ModuleNotFoundError:
    shap = None

import scipy as sp
import numpy as np
import pandas as pd
from easy_mpl import imshow
import matplotlib.pyplot as plt

try:
    import tensorflow.keras.backend as K
except ModuleNotFoundError:
    K = None

from ai4water.backend import sklearn_models
from ._explain import ExplainerMixin
from .utils import convert_ai4water_model


class ShapExplainer(ExplainerMixin):
    """
    Wrapper around SHAP `explainers` and `plots` to draw and save all the plots
    for a given model.

    Attributes:
        features :
        train_summary : only for KernelExplainer
        explainer :
        shap_values :


    Methods
    --------
    - summary_plot
    - force_plot_single_example
    - dependence_plot_single_feature
    - force_plot_all

    Examples:
        >>>from ai4water.postprocessing.explain import ShapExplainer
        >>>from sklearn.model_selection import train_test_split
        >>>from sklearn import linear_model
        >>>import shap
        ...
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
        "GradientExplainer",
        "PermutationExplainer",
        "SamplingExplainer",
        "PartitionExplainer"
    ]

    def __init__(
            self,
            model,
            data: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
            train_data: Union[np.ndarray, pd.DataFrame, List[np.ndarray]] = None,
            explainer: Union[str, Callable] = None,
            num_means: int = 10,
            path: str = None,
            feature_names: list = None,
            framework: str = None,
            layer: Union[int, str] = None,
    ):
        """

        Args:
            model :
                a Model/regressor/classifier from sklearn/xgboost/catboost/LightGBM/tensorflow/pytorch/ai4water
                The model must have a `predict` method.
            data :
                Data on which to make interpretation. Its dimension should be
                same as that of training data. It can be either training or test
                data
            train_data :
                The data on which the `model` was trained. It is used to
                get train_summary. It can a numpy array or a pandas DataFrame.
                Only required for scikit-learn based models.
            explainer : str
                the explainer to use. If not given, the explainer will be inferred.
            num_means : int
                Numher of means, used in `shap.kmeans` to calculate train_summary
            path : str
                path to save the plots. By default, plots will be saved in current
                working directory
            feature_names : list
                Names of features. Should only be given if train/test data is numpy
                array.
            framework : str
                either "DL" or "ML", where "DL" represents deep learning or neural
                network based models and "ML" represents other models. For "DL" the explainer
                will be either "DeepExplainer" or "GradientExplainer". If not given, it will
                be inferred. In such a case "DeepExplainer" will be prioritized over
                "GradientExplainer" for DL frameworks and "TreeExplainer" will be prioritized
                for "ML" frameworks.
            layer : Union[int, str]
                only relevant when framework is "DL" i.e when the model consits of layers
                of neural networks.

        """
        test_data = maybe_to_dataframe(data, feature_names)
        train_data = maybe_to_dataframe(train_data, feature_names)

        super(ShapExplainer, self).__init__(path=path or os.getcwd(), data=test_data, features=feature_names)

        if train_data is None:
            self._check_data(test_data)
        else:
            self._check_data(train_data, test_data)

        model, framework, explainer, model_name = convert_ai4water_model(model,
                                                                         framework,
                                                                         explainer)
        self.is_sklearn = True
        if model_name not in sklearn_models:
            if model_name in ["XGBRegressor",
                              "XGBClassifier",
                              "LGBMRegressor",
                              "LGBMClassifier",
                              "CatBoostRegressor",
                              "CatBoostClassifier"  
                              "XGBRFRegressor"
                              "XGBRFClassifier"
                                            ]:
                self.is_sklearn = False

            elif not self._is_dl(model):
                raise ValueError(f"{model.__class__.__name__} is not a valid model model")

        self._framework = self.infer_framework(model, framework, layer, explainer)

        self.model = model
        self.data = test_data
        self.layer = layer
        self.features = feature_names

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
        if int(tf.__version__[0]) < 2:
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

        if inf_framework != "DL":
            assert layer is None

        if inf_framework == "DL" and isinstance(explainer, str):
            assert explainer in ("DeepExplainer",
                                 "GradientExplainer",
                                 "PermutationExplainer"), f"invalid explainer {inf_framework}"

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

            elif explainer == "PermutationExplainer":
                explainer = shap.PermutationExplainer(self.model, self.data)

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

        elif self.model.__class__.__name__ in sklearn_models:
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

        if self.layer is None:
            # GradientExplainer is also possible without specifying a layer
            return shap.GradientExplainer(self.model, self.data)
        if isinstance(self.layer, int):
            return shap.GradientExplainer((self.model.layers[self.layer].input, self.model.layers[-1].output),
                                          self.map2layer(self.data, self.layer))
        else:
            return shap.GradientExplainer((self.model.get_layer(self.layer).input, self.model.layers[-1].output),
                                          self.map2layer(self.data, self.layer))

    def _check_data(self, *data):
        if self.single_source:
            for d in data:
                assert type(d) == np.ndarray or type(d) == pd.DataFrame, f"""
                data must be numpy array or pandas dataframe but it is of type {d.__class__.__name__}"""

            assert len(set([d.ndim for d in data])) == 1, "train and test data should have same ndim"
            assert len(set([d.shape[-1] for d in data])) == 1, "train and test data should have same input features"
            assert len(set([type(d) for d in data])) == 1, "train and test data should be of same type"

        return

    def get_shap_values(self, data, **kwargs):

        if self.explainer.__class__.__name__ in ["Permutation"]:
            return self.explainer(data)

        elif self._framework == "DL":
            return self._shap_values_dl(data, **kwargs)

        return self.explainer.shap_values(data)

    def _shap_values_dl(self, data, ranked_outputs=None, **kwargs):
        """Gets the SHAP values"""
        data = data.values if isinstance(data, pd.DataFrame) else data

        if self.explainer.__class__.__name__ == "Deep":
            shap_values = self.explainer.shap_values(data, ranked_outputs=ranked_outputs, **kwargs)

        elif isinstance(self.explainer, shap.GradientExplainer) and self.layer is None:
            shap_values = self.explainer.shap_values(data, ranked_outputs=ranked_outputs, **kwargs)

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
                 show=False,
                 save=True
                 ):
        """Draws and saves all the plots for a given sklearn model in the path.

        plot_force_all is set to False by default because it is causing
        Process finished error due. Avoiding this error is a complex function
        of scipy and numba versions.
        """

        if dependence_plots:
            for feature in self.features:
                self.dependence_plot_single_feature(feature, f"dependence_plot_{feature}", show=show, save=save)

        if force_plots:
            for i in range(self.data.shape[0]):

                self.force_plot_single_example(i, f"force_plot_{i}", show=show, save=save)

        if beeswarm_plots:
            self.beeswarm_plot(show=show, save=save)

        if plot_force_all:
            self.force_plot_all("force_plot_all", show=show, save=save)

        if heatmap:
            self.heatmap(show=show, save=save)

        self.summary_plot("summary_plot", show=show, save=save)

        return

    def summary_plot(
            self,
            name: str = "summary_plot",
            show: bool = True,
            save: bool = False,
            **kwargs
    ):
        """Plots the summary plot of SHAP package.

        Arguments:
            name:
                name of saved file
            show:
                whether to show the plot or not
            save:
                whether to save the plot or not
            kwargs:
                any keyword arguments to shap.summary_plot
        """

        def _summary_plot(_shap_val, _data, _features, _name):
            plt.close('all')
            shap.summary_plot(_shap_val, _data, show=False, feature_names=_features, **kwargs)
            if save:
                plt.savefig(os.path.join(self.path, _name), dpi=300, bbox_inches="tight")
            if show:
                plt.show()

            shap.summary_plot(_shap_val, _data, show=False, plot_type="bar", feature_names=_features,
                              **kwargs)
            if save:
                plt.savefig(os.path.join(self.path, _name + " _bar"), dpi=300, bbox_inches="tight")
            if show:
                plt.show()

            return

        shap_vals = self.shap_values
        if isinstance(shap_vals, list) and len(shap_vals) == 1:
            shap_vals = shap_vals[0]

        data = self.data

        if self.single_source:
            if data.ndim == 3:
                assert shap_vals.ndim == 3

                for lookback in range(data.shape[1]):

                    _summary_plot(shap_vals[:, lookback], data[:, lookback], self.features,  _name=f"{name}_{lookback}")
            else:
                _summary_plot(shap_vals, data, self.features, name)
        else:
            # data is a list of data sources
            for idx, _data in enumerate(data):
                if _data.ndim == 3:
                    for lb in range(_data.shape[1]):
                        _summary_plot(shap_vals[idx][:, lb], _data[:, lb], self.features[idx],
                                      _name=f"{name}_{idx}_{lb}")
                else:
                    _summary_plot(shap_vals[idx], _data, self.features[idx], _name=f"{name}_{idx}")

        return

    def force_plot_single_example(
            self,
            idx:int,
            name=None,
            show=True,
            save=False,
            **force_kws
    ):
        """Draws [force_plot](https://shap.readthedocs.io/en/latest/generated/shap.plots.force.html)
        for a single example/row/sample/instance/data point.

        If the data is 3d and shap values are 3d then they are unrolled/flattened
        before plotting

        Arguments:
            idx:
                index of exmaple to use. It can be any value >=0
            name:
                name of saved file
            show:
                whether to show the plot or not
            save:
                whether to save the plot or not
            force_kws : any keyword argument for force plot

        Returns:
            plotter object
        """

        shap_vals = self.shap_values

        if isinstance(shap_vals, list) and len(shap_vals) == 1:
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

        features = self.features
        if data.ndim == 2 and shap_vals.ndim == 2:  # input was 3d i.e. ml model uses 3d input
            features = self.unrolled_features
            expected_value = expected_value[0]  # todo
            shap_vals = shap_vals.reshape(-1,)
            data = data.reshape(-1, )

        plt.close('all')

        plotter = shap.force_plot(
            expected_value,
            shap_vals,
            data,
            feature_names=features,
            show=False,
            matplotlib=True,
            **force_kws
        )

        if save:
            name = name or f"force_plot_{idx}"
            plotter.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")

        if show:
            plotter.show()

        return plotter

    def dependence_plot_all_features(self, show=True, save=False, **dependence_kws):
        """dependence plot for all features"""
        for feature in self.features:
            self.dependence_plot_single_feature(feature, f"dependence_plot_{feature}", show=show, save=save,
                                                **dependence_kws)
        return

    def dependence_plot_single_feature(self, feature, name="dependence_plot", show=False, save=True, **kwargs):
        """dependence plot for a single feature.
        https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html
        https://shap-lrjball.readthedocs.io/en/docs_update/generated/shap.dependence_plot.html
        """
        plt.close('all')

        if len(name) > 150:  # matplotlib raises error if the length of filename is too large
            name = name[0:150]

        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]

        shap.dependence_plot(feature,
                             shap_values,
                             self.data,
                             show=False,
                             **kwargs)
        if save:
            plt.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return

    def force_plot_all(self, name="force_plot.html", save=True, show=True, **force_kws):
        """draws force plot for all examples in the given data and saves it in an html"""

        # following scipy versions cause kernel stoppage when calculating
        if sp.__version__ in ["1.4.1", "1.5.2", "1.7.1"]:
            print(f"force plot can not be plotted for scipy version {sp.__version__}. Please change your scipy")
            return

        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        plt.close('all')
        plot = shap.force_plot(self.explainer.expected_value, shap_values, self.data, **force_kws)
        if save:
            shap.save_html(os.path.join(self.path, name), plot)

        return

    def waterfall_plot_all_examples(
            self,
            name: str = "waterfall",
            save=True,
            show=False,
            **waterfall_kws
    ):
        """Plots the [waterfall plot](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html)
         of SHAP package

        It plots for all the examples/instances from test_data.
        """
        for i in range(len(self.data)):
            self.waterfall_plot_single_example(i, name=name, save=save, show=show, **waterfall_kws)

        return

    def waterfall_plot_single_example(
            self,
            example_index: int,
            name: str = "waterfall",
            show: bool = False,
            save=True,
            max_display: int = 10,
    ):
        """draws and saves [waterfall plot](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html)
         for one example.

        The waterfall plots are based upon SHAP values and show the
        contribution by each feature in model's prediction. It shows which
        feature pushed the prediction in which direction. They answer the
        question, why the ML model simply did not predict mean of training y
        instead of what it predicted. The mean of training observations that
        the ML model saw during training is called base value or expected value.

        Arguments:
            example_index : int
                index of example to use
            max_display : int
                maximu features to display
            name : str
                name of plot
            save : bool
                whether to save the plot or not
            show : bool
                whether to show the plot or now

        """
        if self.explainer.__class__.__name__ in ["Deep"]:
            shap_vals_as_exp = None
        else:
            shap_vals_as_exp = self.explainer(self.data)

        shap_values = self.shap_values
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]

        plt.close('all')

        if shap_vals_as_exp is None:

            features = self.features
            if not self.data_is_2d:
                features = self.unrolled_features

            class Explanation:
                # waterfall plot expects first argument as Explaination class
                # which must have at least these attributes (values, data, feature_names, base_values)
                # https://github.com/slundberg/shap/issues/1420#issuecomment-715190610
                if not self.data_is_2d:  # if original data is 3d then we flat it into 1d array
                    values = shap_values[example_index].reshape(-1, )
                    data = self.data[example_index].reshape(-1, )
                else:
                    values = shap_values[example_index]
                    data = self.data[example_index]

                feature_names = features
                base_values = self.explainer.expected_value[0]

            shap.plots.waterfall(Explanation(), show=False, max_display=max_display)
        else:
            shap.plots.waterfall(shap_vals_as_exp[example_index], show=False, max_display=max_display)

        if save:
            plt.savefig(os.path.join(self.path, f"{name}_{example_index}"), dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return

    def scatter_plot_single_feature(
            self, feature: str,
            name: str = "scatter",
            show=False,
            save=True,
            **scatter_kws
    ):

        shap_values = self.explainer(self.data)
        shap.plots.scatter(shap_values[:, feature], show=False, **scatter_kws)
        if save:
            plt.savefig(os.path.join(self.path, f"{name}_{feature}"), dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return

    def scatter_plot_all_features(self, name="scatter_plot", save=True, show=False, **scatter_kws):
        """draws scatter plot for all features"""
        if isinstance(self.data, pd.DataFrame):
            features = self.features
        else:
            features = [i for i in range(self.data.shape[-1])]

        for feature in features:
            self.scatter_plot_single_feature(feature, name=name, save=save, show=show, **scatter_kws)

        return

    def heatmap(self, name: str = 'heatmap', show: bool = False, save=False, max_display=10):
        """Plots the heat map and saves it
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html
        This can be drawn for xgboost/lgbm as well as for randomforest type models
        but not for CatBoostRegressor which is todo.

        Explanation
        ----------
        The upper line plot on the heat map shows $-fx/max(abs(fx))$ where $fx$ is
        the mean SHAP value of all features. The length of $fx$ is equal to length
        of data/examples. Thus one point on this line is the mean of SHAP values
        of all input features for the given/one example normalized by the maximum
        absolute value of $fx$.
        """

        # if heat map is drawn with np.ndarray, it throws errors therefore convert
        # it into pandas DataFrame. It is more interpretable and does not hurt.
        shap_values = self._get_shap_values_locally()

        # # by default examples are ordered in such a way that examples with similar
        # # explanations are grouped together.
        # self._heatmap(shap_values, f"{name}_basic",
        #               show=show,
        #               save=save,
        #               max_display=max_display)
        #
        # # sort by the maximum absolute value of a feature over all the examples
        # self._heatmap(shap_values, f"{name}_sortby_maxabs", show=show,
        #               max_display=max_display,
        #               save=save,
        #               feature_values=shap_values.abs.max(0))

        # sorting by the sum of the SHAP values over all features gives a complementary perspective on the data
        self._heatmap(shap_values, f"{name}_sortby_SumOfShap", show=show,
                      save=save,
                      max_display=max_display,
                      instance_order=shap_values.sum(1))
        return

    def _heatmap(self, shap_values, name, show=False, max_display=10, save=True,  **kwargs):
        plt.close('all')

        # set show to False because we want to reset xlabel
        shap.plots.heatmap(shap_values, show=False,  max_display=max_display, **kwargs)
        plt.xlabel("Examples")

        if save:
            plt.savefig(os.path.join(self.path, f"{name}_sortby_SumOfShap"), dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        return

    def _get_shap_values_locally(self):
        data = self.data
        if not isinstance(self.data, pd.DataFrame) and data.ndim == 2:
            data = pd.DataFrame(self.data, columns=self.features)

        # not using global explainer because, this explainer should data as well
        explainer = shap.Explainer(self.model, data)
        shap_values = explainer(data)

        return shap_values

    def beeswarm_plot(
            self,
            name: str = "beeswarm",
            show: bool = False,
            max_display: int = 10,
            save: bool = True,
            **kwargs
    ):
        """
        Draws the [beeswarm plot](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html)
        of shap.

        Arguments:
            name : str
                name of saved file
            show : bool
                whether to show the plot or not
            save :
                whether to save the plot or not
            max_display :
                maximum
            kwargs :
                any keyword arguments for shap.beeswarm plot
        """

        shap_values = self._get_shap_values_locally()

        self._beeswarm_plot(shap_values,
                            name=f"{name}_basic", show=show, max_display=max_display,
                            save=save,
                            **kwargs)

        # find features with high impacts
        self._beeswarm_plot(shap_values, name=f"{name}_sortby_maxabs", show=show, max_display=max_display,
                            order=shap_values.abs.max(0), save=save, **kwargs)

        # plot the absolute value
        self._beeswarm_plot(shap_values.abs,
                            name=f"{name}_abs_shapvalues", show=show, max_display=max_display,
                            save=save, **kwargs)
        return

    def _beeswarm_plot(self, shap_values, name, show=False, max_display=10, save=True, **kwargs):

        plt.close('all')
        shap.plots.beeswarm(shap_values,  show=False, max_display=max_display, **kwargs)
        if save:
            plt.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return

    def decision_plot(
            self,
            indices=None,
            name: str = "decision_",
            show=False,
            save=True,
            **decision_kwargs):
        """decision plot
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html
        https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba
        """
        shap_values = self.shap_values
        legend_location = "best"
        legend_labels = None
        if indices is not None:
            shap_values = shap_values[(indices), :]
            if len(shap_values) <= 10:
                legend_labels = indices
                legend_location = "lower right"

        if self.explainer.__class__.__name__ in ["Tree"]:
            shap.decision_plot(self.explainer.expected_value,
                               shap_values,
                               self.features,
                               legend_labels=legend_labels,
                               show=False,
                               legend_location=legend_location,
                               **decision_kwargs)
            if save:
                plt.savefig(os.path.join(self.path, name), dpi=300, bbox_inches="tight")
            if show:
                plt.show()
        else:
            raise NotImplementedError

    def plot_shap_values(
            self,
            interpolation=None,
            cmap="coolwarm",
            name: str = "shap_values",
            show: bool = True,
            save: bool = False
    ):
        """Plots the SHAP values.

        Arguments:
            name:
                name of saved file
            show:
                whether to show the plot or not
            interpolation:
                interpolation argument to axis.imshow
            cmap:
                color map
            save:
                whether to save the plot or not

        """
        shap_values = self.shap_values

        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values: np.ndarray = shap_values[0]

        def plot_shap_values_single_source(_data, _shap_vals, _features, _name):
            if _data.ndim == 3 and _shap_vals.ndim == 3:  # input is 3d
                # assert _shap_vals.ndim == 3
                return imshow_3d(_shap_vals,
                                 _data,
                                 _features,
                                 name=_name,
                                 path=self.path,
                                 show=show,
                                 cmap=cmap)

            plt.close('all')
            fig, axis = plt.subplots()
            im = axis.imshow(_shap_vals.T,
                             aspect='auto',
                             interpolation=interpolation,
                             cmap=cmap
                             )
            if _features is not None:  # if imshow is successful then don't worry if features are None
                axis.set_yticks(np.arange(len(_features)))
                axis.set_yticklabels(_features)
            axis.set_ylabel("Features")
            axis.set_xlabel("Examples")

            fig.colorbar(im)
            if save:
                plt.savefig(os.path.join(self.path, _name), dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            return

        if self.single_source:
            plot_shap_values_single_source(self.data, shap_values, self.features, name)
        else:
            for idx, d in enumerate(self.data):
                plot_shap_values_single_source(d,
                                               shap_values[idx],
                                               self.features[idx],
                                               f"{idx}_{name}")
        return

    def pdp_all_features(
            self,
            show: bool = False,
            save: bool = True,
            **pdp_kws
    ):
        """partial dependence plot of all features.

        Arguments:
            show:
            save:
            pdp_kws:
                any keyword arguments
            """
        for feat in self.features:
            self.pdp_single_feature(feat, show=show, save=save, **pdp_kws)

        return

    def pdp_single_feature(
            self,
            feature_name: str,
            show=True,
            save=False,
            **pdp_kws
    ):
        """partial depence plot using SHAP package for a single feature."""

        shap_values = None
        if hasattr(self.shap_values, 'base_values'):
            shap_values = self.shap_values

        if self.model.__class__.__name__.startswith("XGB"):
            self.model.get_booster().feature_names = self.features

        fig = shap.partial_dependence_plot(
            feature_name,
            model=self.model.predict,
            data=self.data,
            model_expected_value=True,
            feature_expected_value=True,
            shap_values=shap_values,
            feature_names=self.features,
            show=False,
            **pdp_kws
        )

        if save:
            fname = f"pdp_{feature_name}"
            plt.savefig(os.path.join(self.path, fname), dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig


def imshow_3d(values,
              data,
              feature_names: list,
              path, vmin=None, vmax=None,
              name="",
              show=False,
              cmap=None,
              ):

    num_examples, lookback, input_features = values.shape
    assert data.shape == values.shape

    for idx, feat in enumerate(feature_names):
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2, sharex='all', figsize=(10, 12))

        yticklabels=[f"t-{int(i)}" for i in np.linspace(lookback - 1, 0, lookback)]
        axis, im = imshow(data[:, :, idx].transpose(),
                          yticklabels=yticklabels,
                          ax=ax1,
                          vmin=vmin,
                          vmax=vmax,
                          title=feat,
                          cmap=cmap,
                          show=False
                          )
        fig.colorbar(im, ax=axis, orientation='vertical', pad=0.2)

        axis, im = imshow(values[:, :, idx].transpose(),
                          yticklabels=yticklabels,
                          vmin=vmin, vmax=vmax,
                          xlabel="Examples",
                          title=f"SHAP Values",
                          cmap=cmap,
                          show=False,
                          ax=ax2)

        fig.colorbar(im, ax=axis, orientation='vertical', pad=0.2)

        _name = f'{name}_{feat}_shap_values'
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


def maybe_to_dataframe(data, features=None) -> pd.DataFrame:
    if isinstance(data, np.ndarray) and isinstance(features, list) and data.ndim == 2:
        data = pd.DataFrame(data, columns=features)
    return data
