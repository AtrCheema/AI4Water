
import warnings
from typing import Union


from ai4water.backend import easy_mpl as ep
from ai4water.backend import tf, keras, np, plt, os, random, lightgbm, xgboost, sklearn

if tf is not None:
    import ai4water.keract_mod as keract
else:
    keract = None

from ai4water.utils.plotting_tools import Plots
from ai4water.utils.utils import maybe_three_outputs, get_nrows_ncols

from ..utils import choose_examples

try:
    from ai4water.utils.utils_from_see_rnn import rnn_histogram
except ModuleNotFoundError:
    rnn_histogram = None

try:
    from dtreeviz import trees
except ModuleNotFoundError:
    trees = None


plot_tree = sklearn.tree.plot_tree

RNN_INFO = {"LSTM": {'rnn_type': 'LSTM',
                     'gate_names': ['INPUT', 'FORGET', 'CELL', 'OUTPUT'],
                     'n_gates': 4,
                     'is_bidir': False,
                     'rnn_dim': 64,
                     'uses_bias': True,
                     'direction_names': [[]]
                     }
            }

CMAPS = [
    'jet_r', 'ocean_r', 'viridis_r', 'BrBG',
    'GnBu',
    #'crest_r',
    'Blues_r', 'bwr_r',
    #'flare',
    'YlGnBu'
]


TREE_BASED_MODELS = [
    "DecisionTreeRegressor",
    "ExtraTreeRegressor",
    "XGBRFRegressor",
    "XGBRegressor",
    "CatBoostRegressor",
    "LGBMRegressor",

    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "XGBClassifier",
    "XGBRFClassifier",
    "CatBoostClassifier",
    "LGBMClassifier"

]


class Visualize(Plots):
    """Hepler class to peek inside the machine learning mdoel.

    If the machine learning model consists of layers of neural networks,
    then this class can be used to plot following 4 items

        - outputs of individual layers
        - gradients of outputs of individual layers
        - weights and biases of individual layers
        - gradients of weights of individual layers

    If the machine learning model consists of tree, then this
    class can be used to plot the learned tree of the model.

    methods
    -------
        - get_activations
        - activations
        - get_activation_gradients
        - activation_gradients
        - get_weights
        - weights
        - get_weight_gradients
        - weight_gradients
        - decision_tree
    """

    def __init__(
            self,
            model,
            save=True,
            show=True,
    ):
        """
        Arguments:
            model : the learned machine learning model.
        """
        plt.rcParams.update(plt.rcParamsDefault)

        self.model = model
        self.verbosity = model.verbosity
        self.save=save
        self.show=show

        self.vis_path = os.path.join(model.path, "visualize")
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)

        Plots.__init__(self,
                       path=self.vis_path,
                       config=model.config)

    def __call__(
            self,
            layer_name,
            data='training',
            x=None,
            y=None,
            examples_to_use=None,
            show: bool = False,
    ):

        if self.model.category == "DL":
            self.activations(layer_name, data, x, examples_to_use, show=show)
            self.activation_gradients(layer_name, x=x, y=y,
                                      examples_to_use=examples_to_use, show=show)
            self.weights(layer_name, show=show)
            self.weight_gradients(layer_name, data=data, x=x, y=y, show=show)
        else:
            self.decision_tree(show=show)
            self.decision_tree_leaves(data=data)

        return

    def get_activations(self,
                        layer_names: Union[list, str] = None,
                        x=None,
                        data: str = 'training',
                        batch_size=None,
                        ) -> dict:
        """gets the activations/outputs of any layer of the Keras Model.

        Arguments:
            layer_names : name of list of names of layers whose activations are
                to be returned.
            x : If provided, it will override `data`.
            data : data to use to get activations. Only relevent if `x` is not
                provided. By default training data is used. Possible values are
                `training`, `test` or `validation`.
        Returns:
            a dictionary whose keys are names of layers and values are weights
            of those layers as numpy arrays
        """
        if x is None:
            # if layer names are not specified, this will get get activations
            # of allparameters
            x, _ = getattr(self.model, f'{data}_data')()
            #x, _ = maybe_three_outputs(data, self.model.teacher_forcing)

        if self.model.api == "subclassing":
            dl_model = self.model
        else:
            dl_model = self.model._model

        if isinstance(x, list):
            num_examples = len(x[0])
        elif isinstance(x, np.ndarray):
            num_examples = len(x)
        else:
            raise ValueError

        if batch_size:
            # feed each batch and get activations per batch
            assert isinstance(layer_names, str)
            _activations = []
            for batch in range(num_examples // batch_size):
                batch_x = _get_batch_input(x, batch, batch_size)
                batch_activations = keract.get_activations(dl_model, batch_x, layer_names=layer_names,
                                                 auto_compile=True)
                assert len(batch_activations) == 1  # todo
                _activations.append(list(batch_activations.values())[0])
            activations = {layer_names: np.concatenate(_activations)}
        else:
            activations = keract.get_activations(dl_model, x, layer_names=layer_names,
                                             auto_compile=True)

        return activations

    def activations(
            self,
            layer_names=None,
            data: str = 'training',
            x=None,
            examples_to_use: Union[int, list, np.ndarray, range] = None,
            show: bool = False
    ):
        """Plots outputs of any layer of neural network.

        Arguments:
            data : The data to be used for calculating outputs of layers.
            x : if given, will override, 'data'.
            layer_names : name of layer whose output is to be plotted. If None,
                it will plot outputs of all layers
            examples_to_use : If integer, it will be the number of examples to use.
                If array like, it will be the indices of examples to use.
            show :
        """
        activations = self.get_activations(x=x, data=data)

        if layer_names is not None:
            if isinstance(layer_names, str):
                layer_names = [layer_names]
            else:
                layer_names = layer_names
        else:
            layer_names = list(activations.keys())

        assert isinstance(layer_names, list)

        if self.verbosity > 0:
            print("Plotting activations of layers")

        for lyr_name, activation in activations.items():

            if lyr_name in layer_names:
                # activation may be tuple e.g if input layer receives more than
                # 1 input
                if isinstance(activation, np.ndarray):

                    if activation.ndim == 2 and examples_to_use is None:
                        examples_to_use = len(activation)

                    self._plot_activations(activation, lyr_name, examples_to_use,
                                           show)

                elif isinstance(activation, tuple):
                    for act in activation:
                        self._plot_activations(act, lyr_name, show)
        return

    def _plot_activations(self, activation, lyr_name, examples_to_use=24, show=False):

        if examples_to_use is None:
            indices = range(len(activation))
        else:
            activation, indices = choose_examples(activation, examples_to_use)

        kwargs = {}

        if "LSTM" in lyr_name.upper() and np.ndim(activation) in (2, 3):

            if activation.ndim == 3:

                self.features_2d(activation,
                                 show=show,
                                 name=lyr_name + "_outputs",
                                 sup_title="Activations",
                                 n_rows=6,
                                 sup_xlabel="LSTM units",
                                 sup_ylabel="Lookback steps",
                                 title=indices,
                                 )
            else:
                self._imshow(activation, f"{lyr_name} Activations",
                             fname=lyr_name,
                             show=show,
                             ylabel="Examples", xlabel="LSTM units",
                             cmap=random.choice(CMAPS))

        elif np.ndim(activation) == 2 and activation.shape[1] > 1:
            if "lstm" in lyr_name.lower():
                kwargs['xlabel'] = "LSTM units"
            self._imshow(activation, lyr_name + " Activations", show=show,
                         fname=lyr_name, **kwargs)

        elif np.ndim(activation) == 3:
            if "input" in lyr_name.lower():
                kwargs['xticklabels'] = self.model.input_features
            self._imshow_3d(activation, lyr_name, save=show, **kwargs, where='')
        elif np.ndim(activation) == 2:  # this is now 1d
            # shape= (?, 1)
            self.plot1d(activation, label=lyr_name + ' Outputs', show=show,
                        fname=lyr_name + '_outputs')
        else:
            print("ignoring activations for {} because it has shape {}, {}".format(lyr_name, activation.shape,
                                                                                   np.ndim(activation)))
        return

    def get_weights(self):
        """ returns all trainable weights as arrays in a dictionary"""
        weights = {}
        for weight in self.model.trainable_weights:
            if tf.executing_eagerly():
                weights[weight.name] = weight.numpy()
            else:
                weights[weight.name] = keras.backend.eval(weight)
        return weights

    def weights(
            self,
            layer_names: Union[str, list] = None,
            show: bool = False
    ):
        """Plots the weights of a specific layer or all layers.

        Arguments:
            layer_names : The layer whose weights are to be viewed.
            show :
        """

        weights = self.get_weights()

        if self.verbosity > 0:
            print("Plotting trainable weights of layers of the model.")

        if layer_names is None:
            layer_names = list(weights.keys())
        elif isinstance(layer_names, str):
            layer_names = [layer_names]
        else:
            layer_names = layer_names

        for lyr in layer_names:

            for _name, weight in weights.items():

                if lyr in _name:
                    title = _name
                    fname = _name + '_weights'

                    rnn_args = None
                    if "LSTM" in title.upper():
                        rnn_args = {'n_gates': 4,
                                    'gate_names_str': "(input, forget, cell, output)"}

                    if np.ndim(weight) == 2 and weight.shape[1] > 1:

                        self._imshow(weight, title, show=show, fname=fname,
                                     rnn_args=rnn_args)

                    elif len(weight) > 1 and np.ndim(weight) < 3:
                        self.plot1d(weight, title, show, fname, rnn_args=rnn_args)

                    elif "conv" in _name.lower() and np.ndim(weight) == 3:
                        _name = _name.replace("/", "_")
                        _name = _name.replace(":", "_")

                        self.features_2d(data=weight,
                                         save=show,
                                         name=_name,
                                         slices=64,
                                         slice_dim=2,
                                         tight=True,
                                         borderwidth=1)
                    else:
                        print("ignoring weight for {} because it has shape {}".format(_name, weight.shape))
        return

    def get_activation_gradients(
            self,
            layer_names: Union[str, list] = None,
            x=None,
            y=None,
            data: str = 'training'
    ) -> dict:
        """
        Finds gradients of outputs of a layer.

        either x,y or data is required
        Arguments:
            layer_names : The layer for which, the gradients of its outputs are to be
                calculated.
            x : input data. Will overwrite `data`
            y : corresponding label of x. Will overwrite `data`.
            data : one of `training`, `test` or `validation`
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        if x is None:
            data = getattr(self.model, f'{data}_data')()
            x, y = maybe_three_outputs(data)

        return keract.get_gradients_of_activations(self.model, x, y,
                                                   layer_names=layer_names)

    def activation_gradients(
            self,
            layer_names: Union[str, list],
            data='training',
            x=None,
            y=None,
            examples_to_use=None,
            plot_type="2D",
            show: bool = False
    ):
        """Plots the gradients o activations/outputs of layers

        Arguments:
            layer_names : the layer name for which the gradients of its outputs
                are to be plotted.
            data : the data to be used for calculating gradients
            x : alternative to data
            y : alternative to data
            examples_to_use : the examples from the data to use. If None, then all
                examples will be used, which is equal to the length of data.
            plot_type :
            show :
        """
        if plot_type == "2D":
            return self.activation_gradients_2D(layer_names, data, x, y,
                                                examples_to_use, show)

        return self.activation_gradients_1D(layer_names, data, x, y, examples_to_use,
                                            show)

    def activation_gradients_2D(self,
                                layer_names=None,
                                data='training',
                                x=None,
                                y=None,
                                examples_to_use=24,
                                show=True
                                ):
        """Plots activations of intermediate layers except input and output

        Arguments:
            layer_names :
            data :
            x :
            y :
            examples_to_use : if integer, it will be the number of examples to use.
                If array like, it will be index of examples to use.
            show :
        """

        gradients = self.get_activation_gradients(layer_names=layer_names,
                                                  data=data, x=x, y=y)

        return self._plot_act_grads(gradients, examples_to_use, show=show)

    def activation_gradients_1D(self,
                                layer_names,
                                data='training',
                                x=None,
                                y=None,
                                examples_to_use=None,
                                show=False):
        """Plots gradients of layer outputs as 1D

        Arguments:
            layer_names :
            examples_to_use :
            data :
            x :
            y :
            show :
        """
        gradients = self.get_activation_gradients(layer_names=layer_names,
                                                  data=data, x=x, y=y)

        for lyr_name, gradient in gradients.items():
            fname = lyr_name + "_output_grads"
            title = lyr_name + " Output Gradients"

            if np.ndim(gradient) == 3:
                for idx, example in enumerate(gradient):

                    _title = f"{title}_{idx}"
                    _fname = f"{fname}_{idx}"

                    if "LSTM" in lyr_name:
                        example = example.T

                    self.features_1d(example, name=_fname, title=_title, show=show,
                                     xlabel="Lookback steps", ylabel="Gradients")
        return

    def _plot_act_grads(self, gradients, examples_to_use=24, show=True):

        if self.verbosity > 0:
            print("Plotting gradients of activations of layersr")

        for lyr_name, gradient in gradients.items():

            if examples_to_use is None:
                indices = range(len(gradient))
            else:
                gradient, indices = choose_examples(gradient, examples_to_use)

            fname = lyr_name + "_output_grads"
            title = lyr_name + " Output Gradients"
            if "LSTM" in lyr_name.upper() and np.ndim(gradient) in (2, 3):

                if gradient.ndim == 2:
                    self._imshow(gradient, fname=fname, label=title, show=show,
                                 xlabel="LSTM units")
                else:

                    self.features_2d(gradient,
                                     name=fname,
                                     title=indices,
                                     show=show,
                                     n_rows=6,
                                     sup_title=title,
                                     sup_xlabel="LSTM units",
                                     sup_ylabel="Lookback steps")

            elif np.ndim(gradient) == 2:
                if gradient.shape[1] > 1:
                    # (?, ?)
                    self._imshow(gradient, title, show, fname)
                elif gradient.shape[1] == 1:
                    # (? , 1)
                    self.plot1d(np.squeeze(gradient), title, show, fname)

            elif np.ndim(gradient) == 3 and gradient.shape[1] == 1:
                if gradient.shape[2] == 1:
                    # (?, 1, 1)
                    self.plot1d(np.squeeze(gradient), title, show, fname)
                else:
                    # (?, 1, ?)
                    self._imshow(np.squeeze(gradient), title, show, fname)
            elif np.ndim(gradient) == 3:
                if gradient.shape[2] == 1:
                    # (?, ?, 1)
                    self._imshow(np.squeeze(gradient), title, show, fname)
                elif gradient.shape[2] > 1:
                    # (?, ?, ?)
                    self._imshow_3d(gradient, lyr_name, show)
            else:
                print("ignoring activation gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                               np.ndim(gradient)))

    def get_weight_gradients(
            self,
            data: str = 'training',
            x=None,
            y=None
    ) -> dict:
        """Returns the gradients of weights.

        Arguments:
            data : the data to use to calculate gradients of weights.
            x :
            y :

        Returns:
            dictionary whose keys are names of layers and values are gradients of
            weights as numpy arrays.
        """
        if x is None:
            data = getattr(self.model, f'{data}_data')()
            x, y = maybe_three_outputs(data)

        return keract.get_gradients_of_trainable_weights(self.model, x, y)

    def weight_gradients(
            self,
            layer_names: Union[str, list] = None,
            data='training',
            x=None,
            y=None,
            show: bool = False,
    ):
        """Plots gradient of all trainable weights

        Arguments:
            layer_names : the layer whose weeights are to be considered.
            data :  the data to use to calculate gradients of weights
            x : alternative to data
            y : alternative to data
            show : whether to show the plot or not.
        """
        gradients = self.get_weight_gradients(data=data, x=x, y=y)

        if layer_names is None:
            layers_to_plot = list(gradients.keys())
        elif isinstance(layer_names, str):
            layers_to_plot = [layer_names]
        else:
            layers_to_plot = layer_names

        if self.verbosity > 0:
            print("Plotting gradients of trainable weights")

        for lyr_to_plot in layers_to_plot:

            for lyr_name, gradient in gradients.items():

                # because lyr_name is most likely larger
                if lyr_to_plot in lyr_name:

                    title = lyr_name + "Weight Gradients"
                    fname = lyr_name + '_weight_grads'
                    rnn_args = None

                    if "LSTM" in title.upper():
                        rnn_args = {'n_gates': 4,
                                    'gate_names_str': "(input, forget, cell, output)"}

                        if np.ndim(gradient) == 3:
                            self.rnn_histogram(gradient, name=fname, title=title,
                                               show=show)

                    if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                        self._imshow(gradient, title, show=show, fname=fname,
                                     rnn_args=rnn_args)

                    elif len(gradient) and np.ndim(gradient) < 3:
                        self.plot1d(gradient, title, show=show, fname=fname,
                                    rnn_args=rnn_args)
                    else:
                        print(f"""ignoring weight gradients for {lyr_name} because it has
                         shape {gradient.shape} {np.ndim(gradient)}""")
        return

    def find_num_lstms(self, layer_names=None) -> list:
        """Finds names of lstm layers in model"""
        if layer_names is not None:
            if isinstance(layer_names, str):
                layer_names = [layer_names]
            assert isinstance(layer_names, list)

        lstm_names = []
        for lyr, config in self.config['model']['layers'].items():
            if "LSTM" in lyr.upper():
                config = config.get('config', config)
                prosp_name = config.get('name', lyr)

                if layer_names is not None:
                    if prosp_name in layer_names:
                        lstm_names.append(prosp_name)
                else:
                    lstm_names.append(prosp_name)

        return lstm_names

    def get_rnn_weights(self, weights: dict, layer_names=None) -> dict:
        """Finds RNN related weights.

         It combines kernel recurrent curnel and bias of each layer into a list."""
        lstm_weights = {}
        if self.config['model'] is not None and 'layers' in self.config['model']:
            if "LSTM" in self.config['model']['layers']:
                lstms = self.find_num_lstms(layer_names)
                for lstm in lstms:
                    lstm_w = []
                    for w in ["kernel", "recurrent_kernel", "bias"]:
                        w_name = lstm + "/lstm_cell/" + w
                        w_name1 = f"{lstm}/{w}"
                        for k, v in weights.items():
                            if any(_w in k for _w in [w_name, w_name1]):
                                lstm_w.append(v)

                    lstm_weights[lstm] = lstm_w

        return lstm_weights

    def rnn_weights_histograms(self, layer_name, show=False):

        weights = self.get_weights()
        rnn_weights = self.get_rnn_weights(weights, layer_name)

        for k, w in rnn_weights.items():
            self.rnn_histogram(w, name=k + "_weight_histogram", show=show)

        return

    def rnn_weight_grads_as_hist(self,
                                 layer_name=None,
                                 data='training',
                                 x=None,
                                 y=None,
                                 show=False
                                 ):

        gradients = self.get_weight_gradients(data=data, x=x, y=y)

        rnn_weights = self.get_rnn_weights(gradients)
        for k, w in rnn_weights.items():
            self.rnn_histogram(w, name=k + "_weight_grads_histogram", show=show)

        return

    def rnn_histogram(self, data, save=True, name='', show=False, **kwargs):

        if save:
            save = os.path.join(self.vis_path, name + "0D.png")
        else:
            save = None

        if rnn_histogram is None:
            warnings.warn("install see-rnn to plot rnn_histogram plot", UserWarning)
        else:
            rnn_histogram(data, RNN_INFO["LSTM"], bins=400, savepath=save,
                          show=show, **kwargs)

        return

    def decision_tree(self, show=False, **kwargs):
        """Plots the decision tree"""

        fname = os.path.join(self.path, "decision_tree")
        if self.model.category == "ML":
            model_name = list(self.model.config['model'].keys())[0]

            if model_name in TREE_BASED_MODELS:

                _fig, axis = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))

                if model_name.startswith("XGB"):
                    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_tree
                    xgboost.plot_tree(self.model._model, ax=axis, **kwargs)

                elif model_name.startswith("Cat"):

                    gv_object = self.model._model.plot_tree(0, **kwargs)
                    if show:
                        gv_object.view()
                    gv_object.save(filename="decision_tree", directory=self.path)

                elif model_name.startswith("LGBM"):
                    lightgbm.plot_tree(self.model._model, ax=axis, **kwargs)

                else:  # sklearn types
                    plot_tree(self.model._model,
                              feature_names=self.model.input_features,
                              ax=axis, **kwargs)

                plt.savefig(fname, dpi=500)
                if show:
                    plt.show()
            else:
                print(f"decision tree can not be plotted for {model_name}")
        else:
            print(f"decision tree can not be plotted for {self.model.category} models")
        return

    def decision_tree_leaves(self, save=True, data='training'):
        """Plots dtreeviz related plots if dtreeviz is installed"""

        model = list(self.config['model'].keys())[0]
        if model in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:

            if trees is None:
                print("dtreeviz related plots can not be plotted")
            else:
                x, y = getattr(self.model, f'{data}_data')()

                if np.ndim(y) > 2:
                    y = np.squeeze(y, axis=2)

                trees.viz_leaf_samples(self.model._model, x, self.in_cols)
                self.save_or_show(save, fname="viz_leaf_samples", where="plots")

                trees.ctreeviz_leaf_samples(self.model._model, x, y,
                                            self.in_cols)
                self.save_or_show(save, fname="ctreeviz_leaf_samples", where="plots")
        return

    def features_2d(self, data, name, save=True, slices=24, slice_dim=0, **kwargs):
        """Calls the features_2d from see-rnn"""
        st=0
        if 'title' in kwargs:
            title = kwargs.pop('title')
        else:
            title = None

        for en in np.arange(slices, data.shape[slice_dim] + slices, slices):

            if save:
                fname = name + f"_{st}_{en}"
                save = os.path.join(self.path, fname+".png")
            else:
                save = None

            if isinstance(title, np.ndarray):
                _title = title[st:en]
            else:
                _title = title

            if slice_dim == 0:
                features_2D(data[st:en, :], savepath=save, title=_title, **kwargs)
            else:
                # assuming it will always be the last dim if not first
                features_2D(data[..., st:en], savepath=save, title=_title, **kwargs)
            st = en
        return

    def features_1d(self, data, save=True, name='', **kwargs):

        if save:
            save = os.path.join(self.path, name + ".png")
        else:
            save=None

        if features_1D is None:
            warnings.warn("install see-rnn to plot features-1D plot", UserWarning)
        else:
            features_1D(data, savepath=save, **kwargs)

        return


def features_2D(data,
                n_rows=None,
                cmap=None,
                sup_xlabel=None,
                sup_ylabel=None,
                sup_title=None,
                title=None,
                show=False,
                savepath=None):
    """
    title: title for individual axis
    sup_title: title for whole plot
    """

    n_subplots = len(data) if data.ndim == 3 else 1
    nrows, ncols = get_nrows_ncols(n_rows, n_subplots)

    cmap = cmap or random.choice(CMAPS)

    fig, axis = plt.subplots(nrows=nrows, ncols=ncols,
                             dpi=100, figsize=(10, 10),
                             sharex='all', sharey='all')

    num_subplots = len(axis.ravel()) if isinstance(axis, np.ndarray) else 1

    if isinstance(title, str):
        title = [title for _ in range(num_subplots)]
    elif isinstance(title, list):
        assert len(title) == num_subplots
    elif isinstance(title, np.ndarray):
        assert len(title) == num_subplots
    elif title:
        title = np.arange(num_subplots)

    if isinstance(axis, plt.Axes):
        axis = np.array([axis])

    vmin = data.min()
    vmax = data.max()

    for idx, ax in enumerate(axis.flat):
        ax, im = ep.imshow(data[idx],
                          ax=ax,
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          title=title[idx],
                          show=False)

    if sup_xlabel:
        fig.text(0.5, 0.04, sup_xlabel, ha='center', fontsize=20)

    if sup_ylabel:
        fig.text(0.04, 0.5, sup_ylabel, va='center', rotation='vertical',
                 fontsize=20)

    fig.subplots_adjust(hspace=0.2)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if sup_title:
        plt.suptitle(sup_title)

    # fig.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()
    return


def features_1D(data, xlabel=None, ylabel=None, savepath=None, show=None, 
        title=None):

    assert data.ndim == 2

    _, axis = plt.subplots()

    for i in data:
        axis.plot(i)

    if xlabel:
        axis.set_xlabel(xlabel)

    if ylabel:
        axis.set_ylabel(ylabel)

    if title:
        axis.set_title(title)

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()
    return


def _get_batch_input(inp, batch_idx, batch_size):
    if isinstance(inp, list):
        batch_inp = []
        for x in inp:
            st = batch_idx*batch_size
            batch_inp.append(x[st: st+batch_size])
    else:
        st = batch_idx * batch_size
        batch_inp = inp[st: st+batch_size]
    return batch_inp
