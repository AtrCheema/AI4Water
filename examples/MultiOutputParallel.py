# This file shows contruction of different models for multi output with following assumptions.
# Each output comes from one parallel structure/layer of # NN.
# Each parallel NN is not connected during forward propagation and each parallel NN receives separate input.
# The final loss is calculated by adding the loss from each of the parallel NN.
# The target/observations for each of the parallell NN is not present concurrently.
# This means the different target values are present at different time stamps.
# The number of inputs and outputs to and from each NN are equal (but not same)

from dl4seq import Model
from dl4seq.utils.utils import check_min_loss
from examples import LSTMAutoEnc_Config

from tensorflow.python.keras.engine import training_utils

import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras


class MultiOutputParallel(Model):
    """
    This structure makes parallel NNs for multiple outputs. Each of the parallel NNs are independent from
    each other during forward propagation.
    """

    def __init__(self, **kwargs):
        super(MultiOutputParallel, self).__init__(**kwargs)
        self.tr_outs = [True for _ in range(self.outs)]
        self.val_outs = [True for _ in range(self.outs)]

    def train_data(self, **kwargs):

        outs = len(self.config['outputs'])

        x_data = []
        y_data = []
        for out in range(outs):

            self.out_cols = [self.config['outputs'][out]]  # because fetch_data depends upon self.outs
            scaler_key = str(out) if 'scaler_key' not in kwargs else kwargs['scaler_key']
            kwargs['scaler_key'] = scaler_key
            x, _, labels = self.fetch_data(data=self.data[out][self.in_cols + self.out_cols], **kwargs)

            x_data.append(x)
            y_data.append(labels)

        self.out_cols = self.config['outputs']  # setting the actual output columns back to original

        return x_data, y_data

    def build(self):

        self.config['tr_outs'] = self.tr_outs
        self.config['val_outs'] = self.val_outs

        inputs = []
        predictions = []

        for out in range(self.outs):

            site_inputs = keras.layers.Input(shape=(self.lookback, self.ins))
            _, site_predictions = self.add_layers(self.config['model']['layers']['nn_' + str(out)], site_inputs)

            inputs.append(site_inputs)
            predictions.append(site_predictions)

        self._model = self.compile(inputs, predictions)

        return

    def train_test_split(self, st, en, indices):

        indices = self.get_indices(indices)

        x, y = self.train_data(st=st, en=en, indices=indices)

        (x, y, sample_weights, val_x, val_y, val_sample_weights) = training_utils.split_training_and_validation_data(
            x, y, None, self.config['val_fraction'])

        return x, y, val_x, val_y

    def fit(self, st=0, en=None, indices=None, **callbacks):
        # Instantiate an optimizer.
        optimizer = keras.optimizers.Adam(learning_rate=self.config['lr'])
        # Instantiate a loss function.
        loss_fn = self.loss

        # Prepare the training dataset.
        batch_size = self.config['batch_size']

        outs = len(self.config['outputs'])

        x, y, val_x, val_y = self.train_test_split(st, en, indices)

        train_data = tuple(x + y)
        val_data = tuple(val_x + val_y)

        print("{} samples used for training and {} samples for validation".format(x[0].shape[0], val_x[0].shape[0]))
        self.info['train_examples'] = x[0].shape[0]
        self.info['val_examples'] = val_x[0].shape[0]

        train_datasets = tf.data.Dataset.from_tensor_slices(train_data)
        train_datasets = train_datasets.shuffle(buffer_size=1024).batch(batch_size)

        val_datasets = tf.data.Dataset.from_tensor_slices(val_data)
        val_datasets = val_datasets.shuffle(buffer_size=1024).batch(batch_size)

        tr_epoch_losses = {key: [] for key in ['loss']+ ['_'+str(i) for i in range(outs)]}
        val_epoch_losses = {key: [] for key in ['loss']+ ['_'+str(i) for i in range(outs)]}
        skipped_batches = None

        @tf.function
        def train_step(train__x, complete_y):

            # skip_flag = False
            with tf.GradientTape() as taape:
                _masks = []
                _y_trues = []
                for out in range(self.outs):
                    mask = tf.greater(tf.reshape(complete_y[out], (-1,)), 0.0)  # # # (batch_size,)
                    _masks.append(mask)
                    _y_trues.append(complete_y[out][mask])

                _predictions = self._model(train__x, training=True)  # predictions for this minibatch

                losses = {}
                for out in range(self.outs):
                    y_obj = _predictions[out][_masks[out]]
                    # if len(y_obj) < 1:
                    #   skip_flag = True
                    losses['_' + str(out)] = keras.backend.mean(loss_fn(_y_trues[out], y_obj))

                # Compute the loss value for this minibatch.
                # loss_val = tf.reduce_sum(list(losses.values()))
                _x = tf.stack(list(losses.values()))
                _x = tf.boolean_mask(_x, self.tr_outs)
                loss_val = tf.reduce_sum(tf.boolean_mask(_x, tf.math.is_finite(_x)))

            losses.update({'loss': float(loss_val)})

            _grads = taape.gradient(loss_val, self._model.trainable_weights)  # list

            grads = [tf.clip_by_value(g, -1.0, 1.0) for g in _grads]

            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

            return losses  # , skip_flag

        @tf.function
        def val_step(_val_x, complete_y):
            masks = []
            y_trues = []
            for out in range(self.outs):
                mask = tf.greater(tf.reshape(complete_y[out], (-1,)), 0.0)  # # # (batch_size,)
                masks.append(mask)
                y_trues.append(complete_y[out][mask])

            predictions = self._model(_val_x, training=False)  # predictions for this minibatch

            val_step_losses = {}
            for out in range(self.outs):
                y_obj = predictions[out][masks[out]]

                val_step_losses['_' + str(out)] = keras.backend.mean(loss_fn(y_trues[out], y_obj))

            # Compute the loss value for this minibatch.
            _x = tf.stack(list(val_step_losses.values()))
            _x = tf.boolean_mask(_x, self.val_outs)
            loss_val = tf.reduce_sum(tf.boolean_mask(_x, tf.math.is_finite(_x)))
            val_step_losses.update({'loss': loss_val})

            return val_step_losses

        start = time.time()

        for epoch in range(self.config['epochs']):

            skipped_batches = 0
            tr_batch_losses = {key: [] for key in ['loss']+ ['_'+str(i) for i in range(outs)]}
            # Iterate over the batches of the dataset.
            for tr_step, data in enumerate(train_datasets):

                # skip_loop = False
                train_x = list(data[0:outs])  # list of all training x data for current batch
                train_y = list(data[outs:])  # list of all training y data for current batch

                tr_losses = train_step(train_x, train_y)
                # ##print(tr_step, skip_loop.numpy())

                for key, loss in tr_losses.items():
                    tr_batch_losses[key].append(loss)

            # end of one training epoch
            for key, loss in tr_batch_losses.items():
                tr_epoch_losses[key].append(np.nanmean(loss))

            # start of validation steps
            # val_skipped_batches = 0
            val_batch_losses = {key: [] for key in ['loss']+ ['_'+str(i) for i in range(outs)]}
            for step, data in enumerate(val_datasets):
                # skip_loop = False
                val_x = list(data[0:outs])  # list of all training x data for current batch
                val_y = list(data[outs:])  # list of all training y data for current batch

                val_losses = val_step(val_x, val_y)

                for key, loss in val_losses.items():
                    val_batch_losses[key].append(float(loss))

            for key, loss in val_batch_losses.items():
                val_epoch_losses[key].append(np.nanmean(loss))

            msg = ' '
            save_this_epoch = False
            msg, save_this_epoch = check_min_loss(tr_epoch_losses['loss'], epoch, msg, save_this_epoch, None)
            msg, save_this_epoch = check_min_loss(val_epoch_losses['loss'], epoch, msg, save_this_epoch, True)

            msg = msg + "{} ouf of {} minibatches skipped".format(skipped_batches, tr_step)

            if save_this_epoch:
                self._model.save_weights(filepath=os.path.join(self.path + "\\weights_{:03d}_{:.4f}.h5"
                                                                .format(epoch, np.nanmean(val_batch_losses['loss']))))
            print(epoch, msg)

        self.info['training_batches'] = int(tr_step)
        self.info['training_time_in_minutes'] = int( (time.time()-start) / 60.0)
        _history = self.at_train_end(skipped_batches, tr_epoch_losses, val_epoch_losses)

        print("Training time: {}".format(time.time()-start))
        return _history

    def at_train_end(self, skipped_batches: int, tr_epoch_losses, val_epoch_losses):
        # compiles losses and saves them

        # lets save the last epoch
        self._model.save_weights(filepath=os.path.join(self.path + "\\weights_last_epoch.h5"))

        _history = {'loss': tr_epoch_losses['loss'], 'val_loss': val_epoch_losses['loss']}

        val_losses = dict()
        for k, v in val_epoch_losses.items():
            val_losses['val_' + k] = v

        self.plot_loss(_history)
        self.info['skipped_batches'] = skipped_batches

        self.save_config(_history)

        # collect all losses in one dictionary
        tr_epoch_losses.update(val_losses)

        # save all the losses or performance metrics
        df = pd.DataFrame.from_dict(tr_epoch_losses)
        df.to_csv(os.path.join(self.path, "losses.csv"))

        return _history

    def denormalize_data(self, first_input, predicted: list, true_outputs: list, scaler_key: str):

        if np.ndim(first_input) == 5:
            # for ConvLSTM cases
            first_input = first_input[:, -1, 0, -1, :]
        else:
            first_input = first_input[:, -1, :]

        true_y = []
        pred_y = []

        for out in range(self.outs):

            in_obs = np.hstack([first_input, true_outputs[out]])
            in_pred = np.hstack([first_input, predicted[out]])
            scaler = self.scalers['scaler_' + scaler_key]
            in_obs_den = scaler.inverse_transform(in_obs)
            in_pred_den = scaler.inverse_transform(in_pred)
            true_y.append(in_obs_den[:, -1:])
            pred_y.append(in_pred_den[:, -1:])

        return pred_y, true_y


class ConvLSTMMultiOutput(MultiOutputParallel):

    def train_data(self, **kwargs):

        outs = len(self.config['outputs'])

        x_data = []
        y_data = []
        for out in range(outs):

            self.out_cols = [self.config['outputs'][out]]  # because fetch_data depends upon self.outs
            x, prev_y, labels = self.fetch_data(data=self.data[out], **kwargs)

            sub_seq = self.config['subsequences']
            sub_seq_lens = int(self.lookback / sub_seq)
            examples = x.shape[0]

            x = x.reshape((examples, sub_seq, 1, sub_seq_lens, self.ins))

            x_data.append(x)
            y_data.append(labels)

        self.out_cols = self.config['outputs']  # setting the actual output columns back to original
        return x_data, y_data

    def build(self):

        inputs = []
        predictions = []

        assert self.lookback % self.config['subsequences'] == int(0), """lookback must be multiple of subsequences,
        lookback is {} while number of subsequences are {}""".format(self.lookback, self.config['subsequences'])

        for out in range(self.outs):

            sub_seq = self.config['subsequences']
            sub_seq_lens = int(self.lookback / sub_seq)

            site_inputs = keras.layers.Input(shape=(sub_seq, 1, sub_seq_lens, self.ins))
            _, site_predictions = self.add_layers(self.config['layers']['nn_' + str(out)], site_inputs)

            inputs.append(site_inputs)
            predictions.append(site_predictions)

        self._model = self.compile(inputs, predictions)

        return

class LSTMAutoEncMultiOutput(MultiOutputParallel):

    def __init__(self, nn_config, **kwargs):
        # because composite attribute is used in this Model
        self.composite = nn_config['composite']

        super(LSTMAutoEncMultiOutput, self).__init__(nn_config=nn_config, **kwargs)

    def train_data(self, **kwargs):

        outs = len(self.config['outputs'])

        x_data = []
        y_data = []
        for out in range(outs):

            self.out_cols = [self.config['outputs'][out]]  # because fetch_data depends upon self.outs
            x, train_y_3, labels = self.fetch_data(data=self.data[out], **kwargs)

            if self.composite:
                outputs = [x, labels]
            else:
                outputs = labels

            x_data.append(x)
            y_data.append(outputs)

        self.out_cols = self.config['outputs']  # setting the actual output columns back to original
        return x_data, y_data


def make_multi_model(input_model,  from_config=False, config_path=None, weights=None,
                     batch_size=8, lookback=19, lr=1.52e-5, **kwargs):

    val_fraction = 0.2

    fpath = os.path.join(os.path.dirname(os.getcwd()), 'data')
    df_1 = pd.read_csv(os.path.join(fpath, 'data_1.csv'))
    df_3 = pd.read_csv(os.path.join(fpath, 'data_3.csv'))
    df_8 = pd.read_csv(os.path.join(fpath, 'data_8.csv'))
    df_12 = pd.read_csv(os.path.join(fpath, 'data_12.csv'))

    assert df_1.shape == df_3.shape == df_8.shape == df_12.shape

    chl_1_nonan_idx = df_1[:-lookback][~df_1['obs_chla_1'][:-lookback].isna().values].index  # 175
    chl_3_nonan_idx = df_3[:-lookback][~df_3['obs_chla_3'][:-lookback].isna().values].index  # 181
    chl_8_nonan_idx = df_8[:-lookback][~df_8['obs_chla_8'][:-lookback].isna().values].index  # 380
    chl_12_nonan_idx = df_12[:-lookback][~df_12['obs_chla_12'][:-lookback].isna().values].index  # 734
    # len(list(set().union(chl_1_nonan_idx.to_list(), chl_3_nonan_idx.to_list(), chl_8_nonan_idx.to_list(), chl_12_nonan_idx.to_list()
    # ))) = 1162

    train_idx_chl_1, test_idx_chl_1 = train_test_split(chl_1_nonan_idx, test_size=val_fraction,
                                                       random_state=313)
    train_idx_chl_3, test_idx_chl_3 = train_test_split(chl_3_nonan_idx, test_size=val_fraction,
                                                       random_state=313)
    train_idx_chl_8, test_idx_chl_8 = train_test_split(chl_8_nonan_idx, test_size=val_fraction,
                                                       random_state=313)
    train_idx_chl_12, test_idx_chl_12 = train_test_split(chl_12_nonan_idx, test_size=val_fraction,
                                                         random_state=313)

    _train_idx = list(set().union(train_idx_chl_1.to_list(),
                                  train_idx_chl_3.to_list(),
                                  train_idx_chl_8.to_list(),
                                  train_idx_chl_12.to_list()
                                  ))  # 863
    _test_idx = list(set().union(test_idx_chl_1.to_list(),
                                 test_idx_chl_3.to_list(),
                                 test_idx_chl_8.to_list(),
                                 test_idx_chl_12.to_list()
                                 ))   # 406

    df_1.index = pd.to_datetime(df_1['date'])
    df_3.index = pd.to_datetime(df_3['date'])
    df_8.index = pd.to_datetime(df_8['date'])
    df_12.index = pd.to_datetime(df_12['date'])

    if from_config:
        _model = input_model.from_config(batch_size=batch_size,
                        inputs=['tmin', 'tmax', 'slr', 'FLOW_INcms', 'SED_INtons', 'WTEMP(C)',
                             'CBOD_INppm', 'DISOX_Oppm', 'H20VOLUMEm3',# 'ORGP_INppm'
                             ],
                        outputs=['obs_chla_1', 'obs_chla_3', 'obs_chla_8', 'obs_chla_12'],
                        val_fraction=val_fraction,
                        lookback=lookback,
                        lr=lr,
                        allow_nan_labels=1,
                        data=[df_1, df_3, df_8, df_12],
                                         **kwargs,)
        _model.load_weights(weights)
    else:
        _model = input_model(
                             data=[df_1, df_3, df_8, df_12],
                             batch_size=batch_size,
                             inputs=['tmin', 'tmax', 'slr', 'FLOW_INcms', 'SED_INtons', 'WTEMP(C)',
                                     'CBOD_INppm', 'DISOX_Oppm', 'H20VOLUMEm3',  # 'ORGP_INppm'
                                     ],
                             outputs=['obs_chla_1', 'obs_chla_3', 'obs_chla_8', 'obs_chla_12'],
                             val_fraction=val_fraction,
                             lookback=lookback,
                             lr=lr,
                             allow_nan_labels=1,
                             **kwargs
                             )
    return _model, _train_idx, _test_idx


if __name__ == "__main__":

    model, train_idx, test_idx = make_multi_model(LSTMAutoEncMultiOutput,
                                                  layers=LSTMAutoEnc_Config,
                                                  batch_size=12,
                                                  lookback=3,
                                                  lr=0.00047681,
                                                  epochs=2)
    history = model.fit(indices=train_idx)

    # cpath = "D:\\playground\\paper_with_codes\\dl_ts_prediction\\results\\convlstm_parallel\\20200918_1549\\config.json"
    # model, train_idx, test_idx = make_multi_model(ConvLSTMMultiOutput, from_config=True,
    #                                               config_path=cpath, weights="weights_341_0.0393.h5")
    #
    # y, obs = model.predict(indices=train_idx, pref='341_train', use_datetime_index=False)
    #
    # test_y, test_obs = model.predict(indices=test_idx, pref='341_test', use_datetime_index=False)
