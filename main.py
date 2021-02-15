__all__ = ["Model"]

import numpy as np
from weakref import WeakKeyDictionary
import pandas as pd
from TSErrors import FindErrors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import json

import keract_mod as keract
from models.global_variables import keras, tf
from utils import plot_results, plot_loss, maybe_create_path, save_config_file
from models.global_variables import LOSSES, ACTIVATIONS
#

KModel = keras.models.Model
layers = keras.layers

tf.compat.v1.disable_eager_execution()
# tf.enable_eager_execution()

np.random.seed(313)
if int(tf.__version__[0]) == 1:
    tf.compat.v1.set_random_seed(313)
elif int(tf.__version__[0]) > 1:
    tf.random.set_seed(313)


class AttributeNotSetYet:
    def __init__(self, func_name):
        self.data = WeakKeyDictionary()
        self.func_name = func_name

    def __get__(self, instance, owner):
        raise AttributeError("run the function {} first to get {}".format(self.func_name, self.name))

    def __set_name__(self, owner, name):
        self.name = name

class AttributeStore(object):
    """ a class which will just make sure that attributes are set at its childs class level and not here.
    It's purpose is just to avoid cluttering of __init__ method of its child classes. """
    k_model = AttributeNotSetYet("`build_nn` to build neural network")
    method = None
    ins = None
    outs = None
    en_densor_We = None
    en_LSTM_cell = None
    auto_enc_composite = None
    de_LSTM_cell = None
    de_densor_We = None
    test_indices = None
    train_indices = None
    run_paras = AttributeNotSetYet("You must define the `run_paras` method first")
    scalers = {}
    cnn_counter = 0
    lstm_counter = 0
    act_counter = 0
    time_dist_counter = 0
    conv2d_lstm_counter = 0
    dense_counter = 0


class Model(AttributeStore):

    def __init__(self, data_config: dict,
                 nn_config: dict,
                 data: pd.DataFrame,
                 intervals=None,
                 path: str=None):
        self.data_config = data_config
        self.nn_config = nn_config
        self.intervals = intervals
        self.data = data[data_config['inputs'] + data_config['outputs']]
        self.ins = len(self.data_config['inputs'])
        self.outs = len(self.data_config['outputs'])
        self.in_cols = self.data_config['inputs']
        self.out_cols = self.data_config['outputs']
        self.loss = LOSSES[self.nn_config['loss']]
        self.KModel = KModel
        self.path = maybe_create_path(path=path)

    @property
    def in_cols(self):
        return self._in_cols

    @in_cols.setter
    def in_cols(self, x: list):
        self._in_cols = x

    @property
    def out_cols(self):
        return self._out_cols

    @out_cols.setter
    def out_cols(self, x: list):
        self._out_cols = x

    @property
    def lookback(self):
        return self.data_config['lookback']

    @property
    def ins(self):
        return self._ins

    @ins.setter
    def ins(self, x):
        self._ins = x

    @property
    def outs(self):
        return self._outs

    @outs.setter
    def outs(self, x):
        self._outs = x

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        self._loss = x

    @property
    def KModel(self):
        """In case when we want to customize the model such as for
        implementing custom `train_step`, we can provide the customized model as input the this Model class"""
        return self._k_model

    @KModel.setter
    def KModel(self, x):
        self._k_model = x

    @property
    def layer_names(self):
        _all_layers = []
        for layer in self.k_model.layers:
            _all_layers.append(layer.name)
        return _all_layers

    def fetch_data(self, data: pd.DataFrame,  st: int, en=None,
                   shuffle: bool = True,
                   cache_data=True,
                   noise: int = 0,
                   indices: list = None,
                   scaler_key: str = '0',
                   return_dt_index=False):
        """
        :param data:
        :param st:
        :param en:
        :param shuffle:
        :param cache_data:
        :param noise:
        :param indices:
        :param scaler_key: in case we are calling fetch_data multiple times, each data will be scaled with a unique
                   MinMaxScaler object and can be saved with a unique key in memory.
        :param return_dt_index: if True, first value in returned `x` will be datetime index. This can be used when
                                fetching data during predict but must be separated before feeding in NN for prediction.
        :return:
        """

        if indices is not None:
            assert isinstance(indices, list), "indices must be list"
            if en is not None or st != 0:
                raise ValueError

        df = data
        dt_index = None
        if return_dt_index:
            assert isinstance(data.index, pd.DatetimeIndex), """\nInput dataframe must have index of type
             pd.DateTimeIndex. A dummy datetime index can be inserted by using following command:
            `data.index = pd.date_range("20110101", periods=len(data), freq='H')`
            If the data file already contains `datetime` column, then use following command
            `data.index = pd.to_datetime(data['datetime'])`
            or use set `use_datetime_index` in `predict` method to False.
            """
            dt_index = list(map(int, np.array(data.index.strftime('%Y%m%d%H%M'))))  # datetime index

        # # add random noise in the data
        if noise > 0:
            x = pd.DataFrame(np.random.randint(0, 1, (len(df), noise)))
            df = pd.concat([df, x], axis=1)
            prev_inputs = self.in_cols
            self.in_cols = prev_inputs + list(x.columns)
            ys = []
            for y in self.out_cols:
                ys.append(df.pop(y))
            df[self.out_cols] = ys

        cols = self.in_cols + self.out_cols
        df = df[cols]

        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data)

        if return_dt_index:
            # pandas will add the 'datetime' column as first column. This columns will only be used to keep
            # track of indices of train and test data.
            df.insert(0, 'dt_index', dt_index)
            self.in_cols = ['dt_index'] + self.in_cols

        self.scalers['scaler_' + scaler_key] = scaler

        if en is None:
            en = df.shape[0]

        if self.intervals is None:

            df = df[st:en]
            df.columns = self.in_cols + self.out_cols

            x, y, label = self.get_data(df,
                                        len(self.in_cols),
                                        len(self.out_cols)
                                        )
            if indices is not None:
                # because the x,y,z have some initial values removed
                indices = np.subtract(np.array(indices), self.lookback - 1).tolist()

                # if indices are given then this should be done after `get_data` method
                x = x[indices]
                y = y[indices]
                label = label[indices]
        else:
            xs, ys, labels = [], [], []
            for _st, _en in self.intervals:
                df1 = df[_st:_en]

                df.columns = self.in_cols + self.out_cols
                df1.columns = self.in_cols + self.out_cols

                if df1.shape[0] > 0:
                    x, y, label = self.get_data(df1,
                                                len(self.in_cols),
                                                len(self.out_cols))
                    xs.append(x)
                    ys.append(y)
                    labels.append(label)

            if indices is None:
                x = np.vstack(xs)[st:en]
                y = np.vstack(ys)[st:en]
                label = np.vstack(labels)[st:en]
            else:
                x = np.vstack(xs)[indices]
                y = np.vstack(ys)[indices]
                label = np.vstack(labels)[indices]

        # data_path = self.data_config['data_path']
        # cache_path = os.path.join(os.path.dirname(data_path), 'cache')
        #
        # if cache_data and os.path.isdir(cache_path) is False:
        #     os.mkdir(cache_path)
        # fname = os.path.join(cache_path, 'nasdaq_T{}.h5'.format(self.lookback))
        # if os.path.exists(fname) and cache_data:
        #     input_x, input_y, label_y = read_cache(fname)
        #     print('load %s successfully' % fname)
        # else:
        #     input_x, input_y, label_y = get_data(data_path, self.lookback, st, en)
        #     if cache_data:
        #         write_cache(fname, input_x, input_y, label_y)

        if shuffle:
            x, y, label = unison_shuffled_copies(x, y, label)

        print('input_X shape:', x.shape)
        print('input_Y shape:', y.shape)
        print('label_Y shape:', label.shape)

        if return_dt_index:
            self.in_cols.remove("dt_index")
        else:
            x = x.astype(np.float32)

        return x, y, label

    def fit(self, inputs, outputs, **callbacks):

        _monitor = 'val_loss' if self.data_config['val_fraction'] > 0.0 else 'loss'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.path + "\\weights_{epoch:03d}_{val_loss:.4f}.hdf5",
            save_weights_only=True,
            monitor=_monitor,
            mode='min',
            save_best_only=True)
        _callbacks = [model_checkpoint_callback]

        _callbacks.append(keras.callbacks.EarlyStopping(
            monitor=_monitor, min_delta=self.nn_config['min_val_loss'],
            patience=self.nn_config['patience'], verbose=0, mode='auto'
        ))

        if 'tensorboard' in callbacks:
            _callbacks.append(keras.callbacks.TensorBoard(log_dir=self.path, histogram_freq=1))
        self.k_model.fit(inputs,
                         outputs,
                         epochs=self.nn_config['epochs'],
                         batch_size=self.data_config['batch_size'], validation_split=self.data_config['val_fraction'],
                         callbacks=_callbacks
                         )
        history = self.k_model.history

        self.save_config(history.history['val_loss'])

        # save all the losses or performance metrics
        df = pd.DataFrame.from_dict(history.history)
        df.to_csv(os.path.join(self.path, "losses.csv"))

        return history

    def get_indices(self, indices=None):
        # returns only train indices if `indices` is None
        if isinstance(indices, str) and indices.upper() == 'RANDOM':
            if self.data_config['ignore_nans']:
                tot_obs = self.data.shape[0]
            else:
                if self.outs == 1:
                    tot_obs = self.data.shape[0] - int(self.data[self.out_cols].isna().sum())  # self.data_config['bs']
                else:
                    # data contains nans and target series are > 1, we want to make sure that they have same nan counts
                    tot_obs = self.data.shape[0] - int(self.data[self.out_cols[0]].isna().sum())
                    nans = self.data[self.out_cols].isna().sum()
                    assert np.all(nans.values == int(nans.sum() / self.outs))

            idx = np.arange(tot_obs - self.lookback)
            train_indices, test_idx = train_test_split(idx, test_size=self.data_config['val_fraction'], random_state=313)
            setattr(self, 'test_indices', list(test_idx))
            indices = list(train_indices)

        setattr(self, 'train_indices', indices)

        return indices

    # def run_paras(self, **kwargs):
    #
    #     train_x, train_y, train_label = self.fetch_data(self.data, **kwargs)
    #     return train_x, train_label
    def run_paras(self, **kwargs):

        outputs = []
        train_x, train_y, train_label = self.fetch_data(self.data, **kwargs)
        inputs = [train_x]
        for out in range(self.outs):
            config0 = self.nn_config['enc_config']['enc_config_' + str(out)]
            s0_train = np.zeros((train_x.shape[0], config0['n_s']))
            h0_train = np.zeros((train_x.shape[0], config0['n_h']))
            inputs = inputs + [s0_train, h0_train]
            outputs.append(train_label[:, out])

        return inputs, outputs

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        indices = self.get_indices(indices)

        inputs, outputs = self.run_paras(st=st, en=en, indices=indices)

        history = self.fit(inputs, outputs, **callbacks)

        plot_loss(history, name=os.path.join(self.path, "loss_curve"))

        return history

    def predict(self, st=0, en=None, indices=None, scaler_key: str = '5', pref: str = 'test',
                use_datetime_index=True, **plot_args):
        """
        scaler_key: if None, the data will not be indexed along date_time index.
        """

        if indices is not None:
            setattr(self, 'predict_indices', indices)

        inputs, true_outputs = self.run_paras(st=st, en=en, indices=indices, scaler_key=scaler_key,
                                              return_dt_index=use_datetime_index)

        first_input = inputs[0]
        dt_index = np.arange(len(first_input))  # default case when datetime_index is not present in input data
        if use_datetime_index:
            # remove the first of first inputs which is datetime index
            dt_index = get_index(np.array(first_input[:, -1, 0], dtype=np.int64))
            first_input1 = first_input[:, :, 1:].astype(np.float32)
            inputs[0] = first_input1

        predicted = self.k_model.predict(x=inputs,
                                         batch_size=self.data_config['batch_size'],
                                         verbose=1)

        if self.outs == 1:
            # denormalize the data
            if np.ndim(first_input) > 3:
                first_input = first_input[:, -1, 0, :]
            elif np.ndim(first_input) == 3:
                if self.method == 'input_attention':
                    first_input = first_input[:, -1, 0:-1]
                else:
                    first_input = first_input[:, -1, :]

            in_obs = np.hstack([first_input, true_outputs])
            in_pred = np.hstack([first_input, predicted])
            scaler = self.scalers['scaler_'+scaler_key]
            in_obs_den = scaler.inverse_transform(in_obs)
            in_pred_den = scaler.inverse_transform(in_pred)
            true_outputs = in_obs_den[:, -self.outs]
            predicted = in_pred_den[:, -self.outs]
        else:
            if np.ndim(first_input) == 3 and self.method == 'input_attention':
                first_input = first_input[:, -1, 0:-1]

            y = np.stack(true_outputs, axis=1)
            in_obs = np.hstack([first_input, y])
            y = np.stack(predicted, axis=1).reshape(-1, self.outs)
            in_pred = np.hstack([first_input, y])
            scaler = self.scalers['scaler_'+scaler_key]
            in_obs_den = scaler.inverse_transform(in_obs)
            in_pred_den = scaler.inverse_transform(in_pred)
            true_outputs = in_obs_den[:, -self.outs:]
            predicted = in_pred_den[:, -self.outs:]
            true_outputs = [true_outputs[:, i] for i in range(self.outs)]
            predicted = [predicted[:, i] for i in range(self.outs)]


        if not isinstance(true_outputs, list):
            true_outputs = [true_outputs]
        if not isinstance(predicted, list):
            predicted = [predicted]

        # convert each output in ture_outputs and predicted lists as pd.Series with datetime indices sorted
        true_outputs = [pd.Series(t_out.reshape(-1,), index=dt_index).sort_index() for t_out in true_outputs]
        predicted = [pd.Series(p_out.reshape(-1,), index=dt_index).sort_index() for p_out in predicted]

        # save the results
        for idx, out in enumerate(self.out_cols):
            p = predicted[idx]
            t = true_outputs[idx]
            # df = pd.DataFrame(np.stack([p, t], axis=1), columns=['true_' + str(out), 'pred_' + str(out)],
            #                   #index=dt_index
            #                   )
            df = pd.concat([t, p], axis=1)
            df.columns = ['true_' + str(out), 'pred_' + str(out)]
            df.to_csv(os.path.join(self.path, pref + '_' + str(out) + ".csv"), index_label='time')

        self.process_results(true_outputs, predicted, pref+'_', **plot_args)

        return predicted, true_outputs

    def process_results(self, true: list, predicted: list, name=None, **plot_args):

        errs = dict()
        for out in range(self.outs):

            t = true[out]
            p = predicted[out]

            if np.isnan(t).sum() > 0:
                mask = np.invert(np.isnan(t))
                t = t[mask]
                p = p[mask]

            errors = FindErrors(t, p)
            errs['out_errors'] = errors.calculate_all()
            # errs['out_stats'] = errors.stats()

            plot_results(t, p, name=os.path.join(self.path, name + self.data_config['outputs'][out]),
                         **plot_args)

        save_config_file(self.path, errors=errs, pref=name)

        return

    def build_nn(self):
        print('building simple lstm model')

        # lstm = self.nn_config['lstm_config']

        inputs = keras.layers.Input(shape=(self.lookback, self.ins))

        lstm_activations = self.add_LSTM(inputs, self.nn_config['lstm_config'])

        predictions = layers.Dense(self.outs)(lstm_activations)

        self.k_model = self.compile(inputs, predictions)

        return

    def add_LSTM(self, inputs, config, seq=False, **kwargs):

        if 'name' in config:
            name = config['name']
        else:
            name = 'lstm_lyr_' + str(np.random.randint(100))

        lstm_activations = layers.LSTM(config['lstm_units'],
                               # input_shape=(self.lookback, self.ins),
                               dropout=config['dropout'],
                               recurrent_dropout=config['rec_dropout'],
                               return_sequences=seq,
                               name=name)(inputs)

        if config['act_fn'] is not None:
            name = 'lstm_act_' + str(np.random.randint(100))
            lstm_activations = ACTIVATIONS[config['act_fn']](lstm_activations)

        return lstm_activations

    def add_1dCNN(self, inputs, config: dict):

        if 'name' in config:
            name = config['name']
        else:
            name = 'cnn_lyr_' + str(np.random.randint(100))
        cnn_activations = layers.Conv1D(filters=config['filters'],
                                  kernel_size=config['kernel_size'],
                                  # activation=cnn['activation'],
                                  name=name)(inputs)

        if  config['act_fn'] is not None:
            name = 'cnn_act_' + str(np.random.randint(100))
            cnn_activations = ACTIVATIONS[config['act_fn']](name=name)(cnn_activations)

        max_pool_lyr = layers.MaxPooling1D(pool_size=config['max_pool_size'],
                                           name='max_pool_lyr')(cnn_activations)
        flat_lyr = layers.Flatten(name='flat_lyr')(max_pool_lyr)

        return flat_lyr

    def compile(self, model_inputs, outputs):

        k_model = self.KModel(inputs=model_inputs, outputs=outputs)
        adam = keras.optimizers.Adam(lr=self.nn_config['lr'])
        k_model.compile(loss=self.loss, optimizer=adam, metrics=['mse'])
        k_model.summary()

        return k_model

    def get_data(self, df, ins, outs):
        input_x = []
        input_y = []
        label_y = []

        row_length = len(df)
        column_length = df.columns.size
        for i in range(row_length - self.lookback+1):
            x_data = df.iloc[i:i+self.lookback, 0:column_length-outs]
            y_data = df.iloc[i:i+self.lookback-1, column_length-outs:]
            label_data = df.iloc[i+self.lookback-1, column_length-outs:]
            input_x.append(np.array(x_data))
            input_y.append(np.array(y_data))
            label_y.append(np.array(label_data))
        input_x = np.array(input_x).reshape(-1, self.lookback, ins)
        input_y = np.array(input_y).reshape(-1, self.lookback-1, outs)
        label_y = np.array(label_y).reshape(-1, outs)

        nans = df[self.data_config['outputs']].isna().sum()
        if int(nans.sum()) > 0:
            if self.data_config['ignore_nans']:
                print("\n{} Ignoring NANs in predictions {}\n".format(10*'*', 10*'*'))
            else:
                if self.method == 'dual_attention':
                    raise ValueError
                if outs> 1:
                    assert np.all(nans.values == int(nans.sum()/outs)), "output columns contains nan values at different indices"
                print('\n{} Removing Samples with nan labels  {}\n'.format(10*'*', 10*'*'))
                y = df[df.columns[-1]]
                nan_idx = y.isna()
                nan_idx_t = nan_idx[self.lookback - 1:]
                non_nan_idx = ~nan_idx_t.values
                label_y = label_y[non_nan_idx]
                input_x = input_x[non_nan_idx]
                input_y = input_y[non_nan_idx]

                assert np.isnan(label_y).sum() < 1, "label still contains nans"

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], "shapes are not same"

        return input_x, input_y, label_y

    def activations(self,layer_names=None, **kwargs):
        # if layer names are not specified, this will get get activations of allparameters
        inputs, outputs = self.run_paras(**kwargs)

        activations = keract.get_activations(self.k_model, inputs, layer_names=layer_names, auto_compile=True)
        return activations, inputs

    def display_activations(self, layer_name: str=None, st=0, en=None, indices=None, **kwargs):
        # not working currently because it requres the shape of activations to be (1, output_h, output_w, num_filters)
        activations, _ = self.activations(st=st, en=en, indices=indices, layer_names=layer_name)
        if layer_name is None:
            activations = activations
        else:
            activations = activations[layer_name]

        keract.display_activations(activations=activations,**kwargs)

    def gradients_of_weights(self, st=0, en=None, indices=None) -> dict:
        test_x, test_y, test_label = self.fetch_data(self.data, st=st, en=en, shuffle=False,
                                                     cache_data=False,
                                                     indices=indices)
        return keract.get_gradients_of_trainable_weights(self.k_model, test_x, test_label)

    def gradients_of_activations(self, st=0, en=None, indices=None, layer_name=None) -> dict:
        test_x, test_y, test_label = self.fetch_data(self.data, st=st, en=en, shuffle=False,
                                                     cache_data=False,
                                                     indices=indices)

        return keract.get_gradients_of_activations(self.k_model, test_x, test_label, layer_names=layer_name)

    def trainable_weights(self):
        """ returns all trainable weights as arrays in a dictionary"""
        weights = {}
        for weight in self.k_model.trainable_weights:
           weights[weight.name] = weight.numpy()
        return weights

    def plot_weights(self, save=None):
        weights = self.trainable_weights()
        for _name, weight in weights.items():
            if np.ndim(weight) == 2 and weight.shape[1] > 1:
                self._imshow(weight, _name + " Weights", save)
            elif len(weight) > 1 and np.ndim(weight) < 3:
                plt.plot(weight)
                plt.title( _name + " Weights")
                plt.show()
            else:
                print("ignoring weight for {} because it has shape {}".format(_name, weight.shape))

    def plot_activations(self, save: bool=False, **kwargs):
        """ plots activations of intermediate layers except input and output.
        If called without any arguments then it will plot activations of all layers."""
        activations, _ = self.activations(**kwargs)

        for lyr_name, activation in activations.items():
            if np.ndim(activation) == 2 and activation.shape[1] > 1:
                self._imshow(activation, lyr_name + " Activations", save, os.path.join(self.path, lyr_name))
            elif np.ndim(activation) == 3:
                self._imshow_3d(activation, lyr_name, save=save)
            else:
                print("ignoring activations for {} because it has shape {}".format(lyr_name, activation.shape))

    def plot_weight_grads(self, save: bool=False, **kwargs):
        """ plots gradient of all trainable weights"""

        gradients = self.gradients_of_weights(**kwargs)
        for lyr_name, gradient in gradients.items():
            if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                self._imshow(gradient, lyr_name + " Weight Gradients", save,  os.path.join(self.path, lyr_name))
            elif len(gradient) and np.ndim(gradient) < 3:
                plt.plot(gradient)
                plt.title(lyr_name + " Weight Gradients")
                plt.show()
            else:
                print("ignoring weight gradients for {} because it has shape {}".format(lyr_name, gradient.shape))

    def plot_act_grads(self, save: bool=None, **kwargs):
        """ plots activations of intermediate layers except input and output"""
        gradients = self.gradients_of_activations(**kwargs)

        for lyr_name, gradient in gradients.items():
            if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                self._imshow(gradient, lyr_name + " Activation Gradients", save, os.path.join(self.path, lyr_name))
            else:
                print("ignoring activation gradients for {} because it has shape {}".format(lyr_name, gradient.shape))

    def view_model(self, **kwargs):
        """ shows all activations, weights and gradients of the keras model."""
        self.plot_act_grads(**kwargs)
        self.plot_weight_grads(**kwargs)
        self.plot_activations(**kwargs)
        self.plot_weights()
        return

    def plot_act_along_lookback(self, activations, sample=0):

        activation = activations[sample, :, :]
        act_t = activation.transpose()

        fig, axis = plt.subplots()

        for idx, _name in enumerate(self.data_config['inputs']):
            axis.plot(act_t[idx, :], label=_name)

        axis.set_xlabel('Lookback')
        axis.set_ylabel('Input attention weight')
        axis.legend(loc="best")

        plt.show()
        return

    def plot_act_along_inputs(self, layer_name:str, name:str=None, **kwargs):

        pred, obs = self.predict(**kwargs)

        if isinstance(pred, list):
            pred = pred[0]
        if isinstance(obs, list):
            obs = obs[0]

        activation, data = self.activations(layer_names=layer_name, **kwargs)

        if isinstance(data, list):
            data = data[0]

        activation = activation[layer_name]
        data = data[:, 0, :]

        assert data.shape[1] == self.ins

        plt.close('all')

        for idx in range(self.ins):
            plt.close('all')

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
            fig.set_figheight(10)

            ax1.plot(data[:, idx], label=self.data_config['inputs'][idx])
            ax1.legend()
            ax1.set_title('activations w.r.t ' + self.data_config['inputs'][idx])
            ax1.set_ylabel(self.data_config['inputs'][idx])

            ax2.plot(pred.values, label='Prediction')
            ax2.plot(obs.values, '.', label='Observed')
            ax2.legend()

            im = ax3.imshow(activation[:, :, idx].transpose(), aspect='auto')
            ax3.set_ylabel('lookback')
            ax3.set_xlabel('samples')
            fig.colorbar(im, orientation='horizontal', pad=0.2)
            plt.subplots_adjust(wspace=0.005, hspace=0.005)
            if name is not None:
                plt.savefig(os.path.join(self.path, name) + str(idx), dpi=400, bbox_inches='tight')
            else:
                plt.show()

        return

    def plot2d_act_for_a_sample(self, activations, sample=0, name: str = None):
        fig, axis = plt.subplots()
        fig.set_figheight(8)
        # for idx, ax in enumerate(axis):
        im = axis.imshow(activations[sample, :, :].transpose(), aspect='auto')
        axis.set_xlabel('lookback')
        axis.set_ylabel('inputs')
        print(self.in_cols)
        axis.set_title('Activations of all inputs at different lookbacks for sample ' + str(sample))
        fig.colorbar(im)
        if name is not None:
            plt.savefig(os.path.join(self.path, name) + '_' + str(sample), dpi=400, bbox_inches='tight')
        else:
            plt.show()
        return

    def plot1d_act_for_a_sample(self, activations, sample=0, name=None):
        fig, axis = plt.subplots()

        for idx in range(self.lookback-1):
            axis.plot(activations[sample, idx, :].transpose(), label='lookback '+str(idx))
        axis.set_xlabel('inputs')
        axis.set_ylabel('activation weight')
        axis.set_title('Activations at different lookbacks for all inputs for sample ' + str(sample))
        if name is not None:
            plt.savefig(os.path.join(self.path, name) + '_' + str(sample), dpi=400, bbox_inches='tight')
        else:
            plt.show()

    def prepare_batches(self, data: pd.DataFrame, target: str):

        assert self.outs == 1

        x = np.zeros((len(data), self.lookback, data.shape[1] - 1))
        y = np.zeros((len(data), self.lookback, 1))

        for i, name in enumerate(list(data.columns[:-1])):
            for j in range(self.lookback):
                x[:, j, i] = data[name].shift(self.lookback - j - 1).fillna(method="bfill")

        for j in range(self.lookback):
            y[:, j, 0] = data[target].shift(self.lookback - j - 1).fillna(method="bfill")

        prediction_horizon = 1
        target = data[target].shift(-prediction_horizon).fillna(method="ffill").values

        x = x[self.lookback:]
        y = y[self.lookback:]
        target = target[self.lookback:]

        return x, y, target

    def _imshow(self, img, label: str = '', save=True, fname=None):
        assert np.ndim(img) == 2, "can not plot {} with shape {} and ndim {}".format(label, img.shape, np.ndim(img))
        plt.close('all')
        plt.imshow(img, aspect='auto')
        plt.colorbar()
        plt.title(label)
        self.save_or_show(save, fname)

    def _imshow_3d(self, activation, lyr_name, save=True):
        act_2d = []
        for i in range(activation.shape[0]):
            act_2d.append(activation[i, :])
        activation_2d = np.vstack(act_2d)
        self._imshow(activation_2d, lyr_name + " Activations (3d of {})".format(activation.shape),
                     save, os.path.join(self.path, lyr_name))

    def plot1d(self, array, label: str = '', save=True, fname=None):
        plt.close('all')
        plt.plot(array, '.')
        plt.title(label)
        self.save_or_show(save, fname)

    def save_or_show(self, save: bool = True, fname=None):
        if save:
            assert isinstance(fname, str)
            if "/" in fname:
                fname = fname.replace("/", "_")
            plt.savefig(os.path.join(self.path, fname))
        else:
            plt.show()

    def save_config(self, history: dict):

        test_indices = np.array(self.test_indices, dtype=int).tolist() if self.test_indices is not None else None
        train_indices = np.array(self.train_indices, dtype=int).tolist() if self.train_indices is not None else None

        config = dict()
        config['min_val_loss'] = int(np.min(history['val_loss'])) if 'val_loss' in history else None
        config['min_loss'] = int(np.min(history['loss'])) if 'val_loss' in history else None
        config['nn_config'] = self.nn_config
        config['data_config'] = self.data_config
        config['test_indices'] = test_indices
        config['train_indices'] = train_indices
        config['intervals'] = self.intervals
        config['method'] = self.method

        save_config_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(cls, config_path: str, data):
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        data_config = config['data_config']
        nn_config = config['nn_config']
        if 'intervals' in config:
            intervals = config['intervals']
        else:
            intervals = None

        cls.from_check_point = True

        # These paras neet to be set here because they are not withing init method
        cls.test_indices = config["test_indices"]
        cls.train_indices = config["train_indices"]

        return cls(data_config=data_config,
                   nn_config=nn_config,
                   data=data,
                   intervals=intervals,
                   path=os.path.dirname(config_path))

    def load_weights(self, w_file: str):
        # loads the weights of keras model from weight file `w_file`.
        cpath = os.path.join(self.path, w_file)
        self.k_model.load_weights(cpath)
        print("congratualtions, model loaded from weights")
        return

def unison_shuffled_copies(a, b, c):
    """makes sure that all the arrays are permuted similarly"""
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def get_index(idx_array, fmt='%Y%m%d%H%M'):
    """ converts a numpy 1d array into pandas DatetimeIndex type."""

    if not isinstance(idx_array, np.ndarray):
        raise TypeError

    return pd.to_datetime(idx_array.astype(str), format=fmt)
