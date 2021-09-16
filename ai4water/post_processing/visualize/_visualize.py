
import numpy as np
from ai4water.backend import tf, keras


import ai4water.keract_mod as keract
from ai4water.utils.plotting_tools import Plots
from ai4water.utils.utils import maybe_three_outputs


class Visualize(Plots):

    def __init__(
            self,
            model
    ):
        self.model = model
        self.verbosity = model.verbosity

        Plots.__init__(self, model.path, model.problem, model.category,
                       config=model.config)

    def plot_act_grads(self, save: bool = True, **kwargs):
        """ plots activations of intermediate layers except input and output"""
        gradients = self.gradients_of_activations(**kwargs)
        return self._plot_act_grads(gradients, save=save)

    def _plot_act_grads(self, gradients, save=True):
        if self.verbosity > 0:
            print("Plotting gradients of activations of layersr")

        for lyr_name, gradient in gradients.items():
            fname = lyr_name + "_output_grads"
            title = lyr_name + " Output Gradients"
            if "LSTM" in lyr_name.upper() and np.ndim(gradient) in (2, 3):

                self.features_2d(gradient, name=fname, title=title, save=save, n_rows=8, norm=(-1e-4, 1e-4))
                self.features_1d(gradient[0], show_borders=False, name=fname, title=title, save=save, n_rows=8)

                if gradient.ndim == 2:
                    self.features_0d(gradient, name=fname, title=title, save=save)

            elif np.ndim(gradient) == 2:
                if gradient.shape[1] > 1:
                    # (?, ?)
                    self._imshow(gradient, title, save, fname)
                elif gradient.shape[1] == 1:
                    # (? , 1)
                    self.plot1d(np.squeeze(gradient), title, save, fname)

            elif np.ndim(gradient) == 3 and gradient.shape[1] == 1:
                if gradient.shape[2] == 1:
                    # (?, 1, 1)
                    self.plot1d(np.squeeze(gradient), title, save, fname)
                else:
                    # (?, 1, ?)
                    self._imshow(np.squeeze(gradient), title, save, fname)
            elif np.ndim(gradient) == 3:
                if gradient.shape[2] == 1:
                    # (?, ?, 1)
                    self._imshow(np.squeeze(gradient), title, save, fname)
                elif gradient.shape[2] > 1:
                    # (?, ?, ?)
                    self._imshow_3d(gradient, lyr_name, save)
            else:
                print("ignoring activation gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                               np.ndim(gradient)))

    def gradients_of_weights(self, x=None, y=None, data:str='training') -> dict:

        if x is None:
            data = getattr(self, f'{data}_data')()
            x, y = maybe_three_outputs(data)

        return keract.get_gradients_of_trainable_weights(self.model, x, y)

    def gradients_of_activations(self, x=None, y=None, data:str='training', layer_name=None) -> dict:
        """
        either x,y or data is required
        x = input data. Will overwrite `data`
        y = corresponding label of x. Will overwrite `data`.
        data : one of `training`, `test` or `validation`
        """
        if x is None:
            data = getattr(self, f'{data}_data')()
            x, y = maybe_three_outputs(data)

        return keract.get_gradients_of_activations(self.model, x, y, layer_names=layer_name)

    def trainable_weights_(self):
        """ returns all trainable weights as arrays in a dictionary"""
        weights = {}
        for weight in self.model.trainable_weights:
            if tf.executing_eagerly():
                weights[weight.name] = weight.numpy()
            else:
                weights[weight.name] = keras.backend.eval(weight)
        return weights

    def find_num_lstms(self) -> list:
        """Finds names of lstm layers in model"""

        lstm_names = []
        for lyr, config in self.config['model']['layers'].items():
            if "LSTM" in lyr.upper():
                config = config.get('config', config)
                prosp_name = config.get('name', lyr)

                lstm_names.append(prosp_name)

        return lstm_names

    def get_rnn_weights(self, weights: dict) -> dict:
        """Finds RNN related weights.

         It combines kernel recurrent curnel and bias of each layer into a list."""
        lstm_weights = {}
        if self.config['model'] is not None and 'layers' in self.config['model']:
            if "LSTM" in self.config['model']['layers']:
                lstms = self.find_num_lstms()
                for lstm in lstms:
                    lstm_w = []
                    for w in ["kernel", "recurrent_kernel", "bias"]:
                        w_name = lstm + "/lstm_cell/" + w
                        for k, v in weights.items():
                            if w_name in k:
                                lstm_w.append(v)

                    lstm_weights[lstm] = lstm_w

        return lstm_weights

    def plot_weights(self, save=True):
        weights = self.trainable_weights_()

        if self.verbosity > 0:
            print("Plotting trainable weights of layers of the model.")

        rnn_weights = self.get_rnn_weights(weights)
        for k, w in rnn_weights.items():
            self.rnn_histogram(w, name=k + "_weight_histogram", save=save)

        for _name, weight in weights.items():
            title = _name + " Weights"
            fname = _name + '_weights'

            rnn_args = None
            if "LSTM" in title.upper():
                rnn_args = {'n_gates': 4,
                            'gate_names_str': "(input, forget, cell, output)"}

            if np.ndim(weight) == 2 and weight.shape[1] > 1:

                self._imshow(weight, title, save, fname, rnn_args=rnn_args, where='weights')

            elif len(weight) > 1 and np.ndim(weight) < 3:
                self.plot1d(weight, title, save, fname, rnn_args=rnn_args, where='weights')

            elif "conv" in _name.lower() and np.ndim(weight) == 3:
                _name = _name.replace("/", "_")
                _name = _name.replace(":", "_")
                self.features_2d(data=weight, save=save, name=_name, where='weights',
                                 slices=64, slice_dim=2, tight=True, borderwidth=1,
                                 norm=(-.1, .1))
            else:
                print("ignoring weight for {} because it has shape {}".format(_name, weight.shape))

    def plot_layer_outputs(self, save: bool = True, lstm_activations=False, x=None, data:str='training'):
        """Plots outputs of intermediate layers except input and output.
        If called without any arguments then it will plot outputs of all layers.
        By default do not plot LSTM activations."""
        activations = self.model.activations(x=x, data=data)

        if self.verbosity > 0:
            print("Plotting activations of layers")

        for lyr_name, activation in activations.items():
            # activation may be tuple e.g if input layer receives more than 1 input
            if isinstance(activation, np.ndarray):
                self._plot_layer_outputs(activation, lyr_name, save, lstm_activations=lstm_activations)

            elif isinstance(activation, tuple):
                for act in activation:
                    self._plot_layer_outputs(act, lyr_name, save)
        return

    def _plot_layer_outputs(self, activation, lyr_name, save, lstm_activations=False):

        if "LSTM" in lyr_name.upper() and np.ndim(activation) in (2, 3) and lstm_activations:

            self.features_2d(activation, save=save, name=lyr_name + "_outputs", title="Outputs", norm=(-1, 1))

        elif np.ndim(activation) == 2 and activation.shape[1] > 1:
            self._imshow(activation, lyr_name + " Outputs", save, lyr_name)
        elif np.ndim(activation) == 3:
            self._imshow_3d(activation, lyr_name, save=save)
        elif np.ndim(activation) == 2:  # this is now 1d
            # shape= (?, 1)
            self.plot1d(activation, label=lyr_name + ' Outputs', save=save,
                        fname=lyr_name + '_outputs')
        else:
            print("ignoring activations for {} because it has shape {}, {}".format(lyr_name, activation.shape,
                                                                                   np.ndim(activation)))
        return

    def plot_weight_grads(self, save: bool = True, **kwargs):
        """ plots gradient of all trainable weights"""

        gradients = self.gradients_of_weights(**kwargs)

        if self.verbosity > 0:
            print("Plotting gradients of trainable weights")

        rnn_weights = self.get_rnn_weights(gradients)
        for k, w in rnn_weights.items():
            self.rnn_histogram(w, name=k + "_weight_grads_histogram", save=save)

        for lyr_name, gradient in gradients.items():

            title = lyr_name + "Weight Gradients"
            fname = lyr_name + '_weight_grads'
            rnn_args = None

            if "LSTM" in title.upper():
                rnn_args = {'n_gates': 4,
                            'gate_names_str': "(input, forget, cell, output)"}

                if np.ndim(gradient) == 3:
                    self.rnn_histogram(gradient, name=fname, title=title, save=save)

            if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                self._imshow(gradient, title, save, fname, rnn_args=rnn_args)

            elif len(gradient) and np.ndim(gradient) < 3:
                self.plot1d(gradient, title, save, fname, rnn_args=rnn_args)
            else:
                print("ignoring weight gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                           np.ndim(gradient)))
            return

