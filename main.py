__all__ = ["Model"]

import numpy as np
import pandas as pd
from TSErrors import FindErrors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import json
import h5py

from nn_tools import NN
from models.global_variables import keras, tf, tcn
from utils import plot_results, plot_loss, maybe_create_path, save_config_file, get_index
from models.global_variables import LOSSES


np.random.seed(313)
if tf is not None:
    tf.compat.v1.disable_eager_execution()
    # tf.enable_eager_execution()
    import keract_mod as keract
    if int(tf.__version__[0]) == 1:
        tf.compat.v1.set_random_seed(313)
    elif int(tf.__version__[0]) > 1:
        tf.random.set_seed(313)

np.set_printoptions(suppress=True)  # to suppress scientific notation while printing arrays


class Model(NN):

    def __init__(self, data_config: dict,
                 nn_config: dict,
                 data,
                 intervals=None,
                 path: str = None):

        super(Model, self).__init__(data_config, nn_config)

        self.intervals = intervals
        self.data = data
        self.in_cols = self.data_config['inputs']
        self.out_cols = self.data_config['outputs']
        self.loss = LOSSES[self.nn_config['loss']]
        self.KModel = keras.models.Model if keras is not None else None
        self.path, self.act_path, self.w_path = maybe_create_path(path=path)

    @property
    def data(self):
        # so that .data attribute can be customized
        return self._data

    @data.setter
    def data(self, x):
        if isinstance(x, pd.DataFrame):
            _data = x[self.data_config['inputs'] + self.data_config['outputs']]
        else:
            _data = x
        self._data = _data

    @property
    def intervals(self):
        # so that itnervals can be set from outside the Model class. A use case can be when we want to implement
        # intervals during train time but not during test/predict time.
        return self._intervals

    @intervals.setter
    def intervals(self, x: list):
        self._intervals = x

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
    def ins(self):
        return len(self.in_cols)

    @property
    def outs(self):
        return len(self.out_cols)

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

    def fetch_data(self, data: pd.DataFrame,  st: int=0, en=None,
                   shuffle: bool = True,
                   write_data=True,
                   noise: int = 0,
                   indices: list = None,
                   scaler_key: str = '0',
                   return_dt_index=False):
        """

        :param data:
        :param st:
        :param en:
        :param shuffle:
        :param write_data: writes the created batches/generated data in h5 file. The filename will be data_scaler_key.h5
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

        if shuffle:
            x, y, label = unison_shuffled_copies(x, y, label)

        x = self.conform_shape(x)

        print('input_X shape:', x.shape)
        print('input_Y shape:', y.shape)
        print('label_Y shape:', label.shape)

        if return_dt_index:
            self.in_cols.remove("dt_index")
        else:
            x = x.astype(np.float32)

        if write_data:
            self.write_cache('data_' + scaler_key, x, y, label)

        return x, y, label

    def conform_shape(self, x):
        """
    makes sure that the shape of x corresponds to first layer of NN. This comes handy if the first
    layer is CNN in CNNLSTM case or first layer is Conv2DLSTM or LSTM or a dense layer. No change in shape
    will be performed when more than one `Input` layers exist.
        """
        input_layers = 0
        for lyr_name in self.layer_names:
            if "INPUT" in lyr_name.upper():
                input_layers += 1

        if input_layers > 1:
            return x
        elif input_layers == 1:

            first_layer_shape = self.first_layer_shape()
            return x.reshape(first_layer_shape)

        else:
            raise ValueError(" {} Input layers found".format(input_layers))

    def first_layer_shape(self) -> tuple:

        shape = []
        for d in self.k_model.layers[0].input.shape:
            if d is None:
                d = -1
            shape.append(d)
        shape = tuple(shape)
        return shape

    def fit(self, inputs, outputs, validation_data=None, **callbacks):

        _callbacks = list()

        _monitor = 'val_loss' if self.data_config['val_fraction'] > 0.0 else 'loss'
        _callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=self.w_path + "\\weights_{epoch:03d}_{val_loss:.4f}.hdf5",
            save_weights_only=True,
            monitor=_monitor,
            mode='min',
            save_best_only=True))

        _callbacks.append(keras.callbacks.EarlyStopping(
            monitor=_monitor, min_delta=self.nn_config['min_val_loss'],
            patience=self.nn_config['patience'], verbose=0, mode='auto'
        ))

        if 'tensorboard' in callbacks:
            _callbacks.append(keras.callbacks.TensorBoard(log_dir=self.path, histogram_freq=1))
        self.k_model.fit(inputs,
                         outputs,
                         epochs=self.nn_config['epochs'],
                         batch_size=self.data_config['batch_size'],
                         validation_split=self.data_config['val_fraction'],
                         validation_data=validation_data,
                         callbacks=_callbacks,
                         steps_per_epoch = self.data_config['steps_per_epoch']
                         )
        history = self.k_model.history

        self.save_config(history.history)

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
                    more = len(self.intervals) * self.lookback if self.intervals is not None else self.lookback
                    tot_obs = self.vals_in_intervals() - more
                else:
                    # data contains nans and target series are > 1, we want to make sure that they have same nan counts
                    tot_obs = self.data.shape[0] - int(self.data[self.out_cols[0]].isna().sum())
                    nans = self.data[self.out_cols].isna().sum()
                    assert np.all(nans.values == int(nans.sum() / self.outs))

            idx = np.arange(tot_obs - self.lookback)
            train_indices, test_idx = train_test_split(idx, test_size=self.data_config['test_fraction'], random_state=313)
            setattr(self, 'test_indices', list(test_idx))
            indices = list(train_indices)

        setattr(self, 'train_indices', indices)

        return indices

    def vals_in_intervals(self):

        if self.intervals is not None:
            interval_length = 0
            for interval in self.intervals:
                interval_length += interval[1] - interval[0]
        else:
            interval_length = self.data.shape[0]

        return interval_length

    def run_paras(self, **kwargs):

        x, y, label = self.fetch_data(self.data, **kwargs)
        return [x], label

    def process_results(self, true: list, predicted: list, name=None, **plot_args):

        errs = dict()
        out_name = ''
        for out, out_name in enumerate(self.out_cols):

            t = true[out]
            p = predicted[out]

            if np.isnan(t).sum() > 0:
                mask = np.invert(np.isnan(t))
                t = t[mask]
                p = p[mask]

            errors = FindErrors(t, p)
            errs[ out_name + '_errors'] = errors.calculate_all()
            errs[ out_name + '_stats'] = errors.stats()

            plot_results(t, p, name=os.path.join(self.path, name + out_name),
                         **plot_args)

        save_config_file(self.path, errors=errs, name=name + '_' + out_name)

        return

    def build_nn(self):

        print('building {} layer based model'.format(self.method))

        inputs, predictions = self.add_layers(None, self.nn_config['layers'])

        self.k_model = self.compile(inputs, predictions)

        if 'tcn' in self.nn_config['layers']:
            tcn.tcn_full_summary(self.k_model, expand_residual_blocks=True)

        return

    def get_val_data(self):
        """ This method can be overwritten in child classes. """
        val_data = None
        if self.data_config['val_data'] is not None:
            if isinstance(self.data_config['val_data'], str):
                val_data = NotImplementedError("Define the method `get_val_data` in your model class.")
            else:
                # `val_data` might have been provided (then use it as it is) or it is None
                val_data = self.data_config['val_data']

        return val_data

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        self.training = True

        indices = self.get_indices(indices)

        inputs, outputs = self.run_paras(st=st, en=en, indices=indices)

        val_data = self.get_val_data()

        history = self.fit(inputs, outputs, validation_data=val_data, **callbacks)

        plot_loss(history.history, name=os.path.join(self.path, "loss_curve"))

        self.training = False

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
            if np.ndim(first_input) == 2:
                dt_index = get_index(np.array(first_input[:, 0], dtype=np.int64))
                first_input = first_input[:, 1:]
            else:
                dt_index = get_index(np.array(first_input[:, -1, 0], dtype=np.int64))
                first_input = first_input[:, :, 1:].astype(np.float32)
            inputs[0] = first_input

        predicted = self.k_model.predict(x=inputs,
                                         batch_size=self.data_config['batch_size'],
                                         verbose=1)

        predicted, true_outputs = self.denormalize_data(first_input, predicted, true_outputs, scaler_key)

        if self.quantiles is None:
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
                df = pd.concat([t, p], axis=1)
                df.columns = ['true_' + str(out), 'pred_' + str(out)]
                df.to_csv(os.path.join(self.path, pref + '_' + str(out) + ".csv"), index_label='time')

            self.process_results(true_outputs, predicted, pref+'_', **plot_args)

            return true_outputs, predicted

        else:
            assert self.outs == 1
            self.plot_quantiles1(true_outputs, predicted)
            self.plot_quantiles2(true_outputs, predicted)
            self.plot_all_qs(true_outputs, predicted)

            return true_outputs, predicted

    def plot_quantile(self, true_outputs, predicted,  min_q:int, max_q, save=False):
        plt.close('all')
        plt.style.use('ggplot')

        st, en = 0, true_outputs.shape[0]
        q_name = "{:.1f}_{:.1f}".format(self.quantiles[min_q] * 100, self.quantiles[max_q] * 100)

        plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
        plt.fill_between(np.arange(st, en), predicted[st:en, min_q], predicted[st:en, max_q], alpha=0.2,
                         color='g', edgecolor=None, label=q_name + ' %')
        plt.legend(loc="best")
        if save:
            plt.savefig(os.path.join(self.path, "q_" + q_name + ".png"))
        else:
            plt.show()

    def plot_all_qs(self, true_outputs, predicted, save=False):
        plt.close('all')
        plt.style.use('ggplot')

        st, en = 0, true_outputs.shape[0]

        plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')

        for idx, q in enumerate(self.quantiles):
            q_name = "{:.1f}".format(self.quantiles[idx] * 100)
            plt.plot(np.arange(st, en), predicted[st:en, idx], label="q {} %".format(q_name))

        plt.legend(loc="best")
        if save:
            plt.savefig(os.path.join(self.path, "all_quantiles.png"))
        else:
            plt.show()
        return

    def plot_quantiles1(self, true_outputs, predicted, save=True):
        plt.close('all')
        plt.style.use('ggplot')

        st, en = 0, true_outputs.shape[0]
        for q in range(len(self.quantiles) - 1):
            st_q = "{:.1f}".format(self.quantiles[q] * 100)
            en_q = "{:.1f}".format(self.quantiles[-q] * 100)

            plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
            plt.fill_between(np.arange(st, en), predicted[st:en, q], predicted[st:en, -q], alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            if save:
                plt.savefig(os.path.join(self.path, 'q' + st_q + '_' + en_q + ".png"))
            else:
                plt.show()
        return

    def plot_quantiles2(self, true_outputs, predicted, save=True):
        plt.close('all')
        plt.style.use('ggplot')

        st, en = 0, true_outputs.shape[0]
        for q in range(len(self.quantiles) - 1):
            st_q = "{:.1f}".format(self.quantiles[q] * 100)
            en_q = "{:.1f}".format(self.quantiles[q+1] * 100)

            plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
            plt.fill_between(np.arange(st, en), predicted[st:en, q], predicted[st:en, q+1], alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            if save:
                plt.savefig(os.path.join(self.path, 'q' + st_q + '_' + en_q + ".png"))
            else:
                plt.show()
        return

    def denormalize_data(self, first_input, predicted, true_outputs, scaler_key):

        if self.outs == 1:
            # denormalize the data
            if np.ndim(first_input) == 4:
                first_input = first_input[:, -1, 0, :]
            elif np.ndim(first_input) == 5:
                first_input = first_input[:, -1, 0, -1, :]
            elif np.ndim(first_input) == 3:
                first_input = first_input[:, -1, :]
            elif np.ndim(first_input) == 2:
                first_input = first_input[:, :]
        else:
            if np.ndim(first_input) == 3:
                first_input = first_input[:, -1, :]
                true_outputs = np.stack(true_outputs, axis=1).reshape(-1, self.outs)
                predicted = np.stack(predicted, axis=1).reshape(-1, self.outs)

        in_obs = np.hstack([first_input, true_outputs])
        in_pred = np.hstack([first_input, predicted])
        scaler = self.scalers['scaler_'+scaler_key]
        in_obs_den = scaler.inverse_transform(in_obs)
        in_pred_den = scaler.inverse_transform(in_pred)
        true_outputs = in_obs_den[:, -self.outs:]
        predicted = in_pred_den[:, -self.outs:]

        if self.outs > 1:
            true_outputs = [true_outputs[:, i] for i in range(self.outs)]
            predicted = [predicted[:, i] for i in range(self.outs)]

        return predicted, true_outputs

    def compile(self, model_inputs, outputs):

        k_model = self.KModel(inputs=model_inputs, outputs=outputs)
        adam = keras.optimizers.Adam(lr=self.nn_config['lr'])
        k_model.compile(loss=self.loss, optimizer=adam, metrics=self.get_metrics())
        k_model.summary()

        try:
            keras.utils.plot_model(k_model, to_file=os.path.join(self.path, "model.png"), show_shapes=True, dpi=300)
        except AssertionError:
            print("dot plot of model could not be plotted")
        return k_model

    def get_metrics(self) -> list:
        """ returns the performance metrics specified"""
        _metrics = self.data_config['metrics']

        metrics = None
        if _metrics is not None:
            if not isinstance(_metrics, list):
                assert isinstance(_metrics, str)
                _metrics = [_metrics]

            from tf_losses import nse, kge, pbias

            METRICS = {'NSE': nse,
                       'KGE': kge,
                       'PBIAS': pbias}

            metrics = []
            for m in _metrics:
                if m.upper() in METRICS.keys():
                    metrics.append(METRICS[m.upper()])
                else:
                    metrics.append(m)

        return metrics

    def get_data(self, df, ins, outs):

        if len(self.first_layer_shape()) == 2:
            # for case when there is not lookback, i.e first layer is dense layer and takes 2D input
            input_x, input_y, label_y = df.iloc[:, 0:ins].values, df.iloc[:, -outs:].values,  df.iloc[:, -outs:].values

            assert self.lookback == 1, """lookback should be one for MLP/Dense layer based model, but it is {}
            """.format(self.lookback)

        else:
            # Provide lookback/history/seq length in input data
            input_x = []
            input_y = []
            label_y = []

            row_length = len(df)
            column_length = df.columns.size
            for i in range(row_length - self.lookback + 1):
                x_data = df.iloc[i:i + self.lookback, 0:column_length - outs]
                y_data = df.iloc[i:i + self.lookback - 1, column_length - outs:]
                label_data = df.iloc[i + self.lookback - 1, column_length - outs:]
                input_x.append(np.array(x_data))
                input_y.append(np.array(y_data))
                label_y.append(np.array(label_data))
            input_x = np.array(input_x, dtype=np.float64).reshape(-1, self.lookback, ins)
            input_y = np.array(input_y, dtype=np.float32).reshape(-1, self.lookback - 1, outs)
            label_y = np.array(label_y, dtype=np.float32).reshape(-1, outs)

        return self.check_nans(df, input_x, input_y, label_y, outs)

    def check_nans(self, df, input_x, input_y, label_y, outs):
        """ checks whether anns are present or not and checks shapes of arrays being prepared.
        """
        nans = df[self.out_cols].isna().sum()
        if int(nans.sum()) > 0:
            if self.data_config['ignore_nans']:
                print("\n{} Ignoring NANs in predictions {}\n".format(10*'*', 10*'*'))
            else:
                if self.method == 'dual_attention':
                    raise ValueError
                if outs > 1:
                    assert np.all(nans.values == int(nans.sum()/outs)), """output columns contains nan values at
                     different indices"""
                print('\n{} Removing Samples with nan labels  {}\n'.format(10*'*', 10*'*'))
                # y = df[df.columns[-1]]
                nan_idx = np.isnan(label_y) # y.isna()
                # nan_idx_t = nan_idx[self.lookback - 1:]
                # non_nan_idx = ~nan_idx_t.values
                non_nan_idx = ~nan_idx.reshape(-1,)
                label_y = label_y[non_nan_idx]
                input_x = input_x[non_nan_idx]
                input_y = input_y[non_nan_idx]

                assert np.isnan(label_y).sum() < 1, "label still contains {} nans".format(np.isnan(label_y).sum())

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], "shapes are not same"

        assert np.isnan(input_x).sum() == 0, "input still contains {} nans".format(np.isnan(input_x).sum())

        return input_x, input_y, label_y

    def activations(self, layer_names=None, **kwargs):
        # if layer names are not specified, this will get get activations of allparameters
        inputs, outputs = self.run_paras(**kwargs)

        activations = keract.get_activations(self.k_model, inputs, layer_names=layer_names, auto_compile=True)
        return activations, inputs

    def display_activations(self, layer_name: str = None, st=0, en=None, indices=None, **kwargs):
        # not working currently because it requres the shape of activations to be (1, output_h, output_w, num_filters)
        activations, _ = self.activations(st=st, en=en, indices=indices, layer_names=layer_name)
        if layer_name is None:
            activations = activations
        else:
            activations = activations[layer_name]

        keract.display_activations(activations=activations, **kwargs)

    def gradients_of_weights(self, **kwargs) -> dict:

        x, y = self.run_paras(**kwargs)

        return keract.get_gradients_of_trainable_weights(self.k_model, x, y)

    def gradients_of_activations(self, st=0, en=None, indices=None, layer_name=None) -> dict:

        x, y = self.run_paras(st=st, en=en, indices=indices)

        return keract.get_gradients_of_activations(self.k_model, x, y, layer_names=layer_name)

    def trainable_weights(self):
        """ returns all trainable weights as arrays in a dictionary"""
        weights = {}
        for weight in self.k_model.trainable_weights:
            if tf.executing_eagerly():
                weights[weight.name] = weight.numpy()
            else:
                weights[weight.name] = keras.backend.eval(weight)
        return weights

    def plot_weights(self, save=True):
        weights = self.trainable_weights()
        for _name, weight in weights.items():
            title = _name + " Weights"
            fname = _name + '_weights'

            if np.ndim(weight) == 2 and weight.shape[1] > 1:
                self._imshow(weight, title, save, fname)
            elif len(weight) > 1 and np.ndim(weight) < 3:
                self.plot1d(weight, title, save, fname)
            else:
                print("ignoring weight for {} because it has shape {}".format(_name, weight.shape))

    def plot_activations(self, save: bool = False, **kwargs):
        """ plots activations of intermediate layers except input and output.
        If called without any arguments then it will plot activations of all layers."""
        activations, _ = self.activations(**kwargs)

        for lyr_name, activation in activations.items():
            if np.ndim(activation) == 2 and activation.shape[1] > 1:
                self._imshow(activation, lyr_name + " Activations", save, os.path.join(self.path, lyr_name))
            elif np.ndim(activation) == 3:
                self._imshow_3d(activation, lyr_name, save=save)
            elif np.ndim(activation) == 2:  # this is now 1d
                # shape= (?, 1)
                self.plot1d(activation, label=lyr_name+' Activations', save=save,
                            fname=os.path.join(self.path, lyr_name+'_activations'))
            else:
                print("ignoring activations for {} because it has shape {}, {}".format(lyr_name, activation.shape,
                                                                                       np.ndim(activation)))

    def plot_weight_grads(self, save: bool = True, **kwargs):
        """ plots gradient of all trainable weights"""

        gradients = self.gradients_of_weights(**kwargs)
        for lyr_name, gradient in gradients.items():

            title = lyr_name + "Weight Gradients"
            fname = os.path.join(self.path, lyr_name+'_weight_grads')

            if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                self._imshow(gradient, title, save,  fname)
            elif len(gradient) and np.ndim(gradient) < 3:
                self.plot1d(gradient, title, save, fname)
            else:
                print("ignoring weight gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                           np.ndim(gradient)))

    def plot_act_grads(self, save: bool = None, **kwargs):
        """ plots activations of intermediate layers except input and output"""
        gradients = self.gradients_of_activations(**kwargs)

        for lyr_name, gradient in gradients.items():
            fname = os.path.join(self.path, lyr_name + "_activation gradients")
            title = lyr_name + " Activation Gradients"
            if np.ndim(gradient) == 2:
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

    def view_model(self, **kwargs):
        """ shows all activations, weights and gradients of the keras model."""
        self.plot_act_grads(**kwargs)
        self.plot_weight_grads(**kwargs)
        self.plot_activations(**kwargs)
        self.plot_weights()
        return

    def plot_act_along_lookback(self, activations, sample=0):

        assert isinstance(activations, np.ndarray)

        activation = activations[sample, :, :]
        act_t = activation.transpose()

        fig, axis = plt.subplots()

        for idx, _name in enumerate(self.in_cols):
            axis.plot(act_t[idx, :], label=_name)

        axis.set_xlabel('Lookback')
        axis.set_ylabel('Input attention weight')
        axis.legend(loc="best")

        plt.show()
        return

    def plot_act_along_inputs(self, layer_name: str, name: str = None, vmin=0, vmax=0.8, **kwargs):

        assert isinstance(layer_name, str), "layer_name must be a string, not of {} type".format(type(layer_name))

        predictions, observations = self.predict(**kwargs)

        activation, data = self.activations(layer_names=layer_name, **kwargs)

        if isinstance(data, list):
            data = data[0]

        activation = activation[layer_name]
        data = data[:, 0, :]

        assert data.shape[1] == self.ins

        plt.close('all')

        for out in range(self.outs):
            pred = predictions[out]
            obs = observations[out]
            out_name = self.out_cols[out]

            for idx in range(self.ins):

                fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
                fig.set_figheight(10)

                ax1.plot(data[:, idx], label=self.in_cols[idx])
                ax1.legend()
                ax1.set_title('activations w.r.t ' + self.in_cols[idx])
                ax1.set_ylabel(self.in_cols[idx])

                ax2.plot(pred.values, label='Prediction')
                ax2.plot(obs.values, '.', label='Observed')
                ax2.legend()

                im = ax3.imshow(activation[:, :, idx].transpose(), aspect='auto', vmin=vmin, vmax=vmax)
                ax3.set_ylabel('lookback')
                ax3.set_xlabel('samples')
                fig.colorbar(im, orientation='horizontal', pad=0.2)
                plt.subplots_adjust(wspace=0.005, hspace=0.005)
                if name is not None:
                    _name = out_name + '_' + name
                    plt.savefig(os.path.join(self.act_path, _name) + self.in_cols[idx], dpi=400, bbox_inches='tight')
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

    def prepare_batches(self, df: pd.DataFrame, ins, outs):

        assert self.outs == 1
        target = self.data_config['outputs'][0]

        x = np.zeros((len(df), self.lookback, df.shape[1] - 1))
        y = np.zeros((len(df), self.lookback, 1))

        for i, name in enumerate(list(df.columns[:-1])):
            for j in range(self.lookback):
                x[:, j, i] = df[name].shift(self.lookback - j - 1).fillna(method="bfill")

        for j in range(self.lookback):
            y[:, j, 0] = df[target].shift(self.lookback - j - 1).fillna(method="bfill")

        prediction_horizon = 1
        target = df[target].shift(-prediction_horizon).fillna(method="ffill").values

        input_x = x[self.lookback:]
        input_y = y[self.lookback:]
        label_y = target[self.lookback:]

        # return x, y, target
        return self.check_nans(df, input_x, input_y, label_y, outs)

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

        save_config_file(indices={'test_indices': test_indices,
                                 'train_indices': train_indices}, path=self.path)

        config = dict()
        config['min_val_loss'] = int(np.min(history['val_loss'])) if 'val_loss' in history else None
        config['min_loss'] = int(np.min(history['loss'])) if 'val_loss' in history else None
        config['nn_config'] = self.nn_config
        config['data_config'] = self.data_config
        config['intervals'] = self.intervals
        config['method'] = self.method
        config['quantiles'] = self.quantiles
        config['loss'] = self.k_model.loss.__name__ if self.k_model is not None else None

        save_config_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(cls, config_path: str, data):
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        idx_file = os.path.join(os.path.dirname(config_path), 'indices.json')
        with open(idx_file, 'r') as fp:
            indices = json.load(fp)

        data_config = config['data_config']
        nn_config = config['nn_config']
        if 'intervals' in config:
            intervals = config['intervals']
        else:
            intervals = None

        cls.from_check_point = True

        # These paras neet to be set here because they are not withing init method
        cls.test_indices = indices["test_indices"]
        cls.train_indices = indices["train_indices"]

        return cls(data_config=data_config,
                   nn_config=nn_config,
                   data=data,
                   intervals=intervals,
                   path=os.path.dirname(config_path))

    def load_weights(self, w_file: str):
        # loads the weights of keras model from weight file `w_file`.
        cpath = os.path.join(self.w_path, w_file)
        self.k_model.load_weights(cpath)
        print("{} Successfully loaded weights {}".format('*'*10, '*'*10))
        return

    def write_cache(self, _fname, input_x, input_y, label_y):
        fname = os.path.join(self.path, _fname)
        h5 = h5py.File(fname, 'w')
        h5.create_dataset('input_X', data=input_x)
        h5.create_dataset('input_Y', data=input_y)
        h5.create_dataset('label_Y', data=label_y)
        h5.close()


def unison_shuffled_copies(a, b, c):
    """makes sure that all the arrays are permuted similarly"""
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
