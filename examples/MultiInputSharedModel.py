# This file shows how to build an NN model using different basic models which are available in models dicrectory.
# The basic assumptions are following
# A single model is used which receives different inputs and procduces corresponding outputs.
# Each of the multiple input is present in a separate file which is read in a dataframe and all the dataframes are
# passed as a list for `data` attribute.

import pandas as pd
import numpy as np
import os

from dl4seq.utils.utils import plot_loss, get_index, make_model
from dl4seq import Model


class MultiInputSharedModel(Model):

    def test_data(self, data, **kwargs):
        x, _, labels = self.fetch_data(data=data, **kwargs)
        return [x], [labels]

    def train_data(self, **kwargs):

        x_data = []
        y_data = []
        for out in range(2):

            self.out_cols = [self.data_config['outputs'][out]]  # because fetch_data depends upon self.outs

            x, _, labels = self.fetch_data(data=self.data[out], **kwargs)

            x_data.append(x)
            y_data.append(labels)

        self.out_cols = self.data_config['outputs']  # setting the actual output columns back to original

        x_data = np.vstack(x_data)
        y_data = np.vstack(y_data)

        return x_data, y_data

    def val_data(self, **kwargs):

        self.out_cols = [self.data_config['outputs'][-1]]  # because fetch_data depends upon self.outs
        x, _, labels = self.fetch_data(data=self.data[-1], **kwargs)

        self.out_cols = self.data_config['outputs']  # setting the actual output columns back to original

        return x, labels

    def build(self):

        inputs, predictions = self.add_layers(self.nn_config['layers'])

        self._model = self.compile(inputs, predictions)

        return

    def train(self, st=0, en=None, indices=None, **callbacks):

        train_data = self.train_data(st=st, en=en, indices=indices)

        val_data = self.val_data(st=st, en=en, indices=indices)

        history = self.fit(train_data[0], train_data[1], validation_data=val_data, **callbacks)

        plot_loss(history.history, name=os.path.join(self.path, "loss_curve"))

        return history.history

    def denormalize_data(self, first_input, predicted: list, true_outputs:list, scaler_key:str):

        return predicted, true_outputs[0]

    def deindexify_input_data(self, inputs, sort=False, use_datetime_index=False):
        first_input = inputs[0]
        dt_index = np.arange(len(first_input))  # default case when datetime_index is not present in input data
        if use_datetime_index:
            # remove the first of first inputs which is datetime index
            dt_index = get_index(np.array(first_input[:, 0], dtype=np.int64))
            first_input = first_input[..., 1:].astype(np.float32)
            inputs[0] = first_input
            if self.num_input_layers > 1:
                for i in range(1, self.num_input_layers):
                    inputs[i] = inputs[i][..., 1:]

        return first_input, inputs, dt_index

    def predict(self, st=0, en=None, indices=None, scaler_key: str = '5', pref: str = 'test',
                use_datetime_index=True, pp=True, **plot_args):
        out_cols = self.out_cols

        predictions = []
        observations = []

        for idx, out in enumerate(out_cols):

            self.out_cols = [self.data_config['outputs'][idx]]  # because fetch_data depends upon self.outs
            inputs, true_outputs = self.test_data(st=st, en=en, indices=indices, scaler_key=scaler_key,
                                                   use_datetime_index=use_datetime_index, data=self.data[idx])
            self.out_cols = self.data_config['outputs']  # setting the actual output columns back to original

            first_input, inputs, dt_index = self.deindexify_input_data(inputs, use_datetime_index=use_datetime_index)

            predicted = self._model.predict(x=inputs,
                                             batch_size=self.data_config['batch_size'],
                                             verbose=1)

            predicted, true_outputs = self.denormalize_data(first_input, predicted, true_outputs, scaler_key)

            horizons = self.data_config['forecast_length']
            if self.quantiles is None:
                true_outputs = pd.DataFrame(true_outputs.reshape(-1, horizons), index=dt_index,
                                            columns=['true_' + str(i) for i in range(horizons)]).sort_index()
                predicted = pd.DataFrame(predicted.reshape(-1, horizons), index=dt_index,
                                         columns=['pred_' + str(i) for i in range(horizons)]).sort_index()

                if pp:
                    df = pd.concat([true_outputs, predicted], axis=1)
                    df.to_csv(os.path.join(self.path, pref + '_' + str(out) + ".csv"), index_label='time')

                    self.out_cols = [out]
                    self.process_results([true_outputs], [predicted], pref + '_', **plot_args)

            else:
                self.plot_quantiles1(true_outputs, predicted)
                self.plot_quantiles2(true_outputs, predicted)
                self.plot_all_qs(true_outputs, predicted)

            predictions.append(predicted)
            observations.append(true_outputs)

        self.out_cols = out_cols
        return observations, predictions


def make_multi_model(input_model,  from_config=False, config_path=None, weights=None,
                     prefix=None,
                     batch_size=8, lookback=19, lr=1.52e-5, ignore_nans=True, **kwargs):

    data_config, nn_config = make_model(batch_size=batch_size,
                                                         lookback=lookback,
                                                         lr=lr,
                                                         ignore_nans=ignore_nans,
                                                         **kwargs)

    data_config['inputs'] = ['tmin', 'tmax', 'slr', 'FLOW_INcms', 'SED_INtons', 'WTEMP(C)',
                             'CBOD_INppm', 'DISOX_Oppm', 'H20VOLUMEm3']
    data_config['outputs'] = ['obs_chla_1', 'obs_chla_3', 'obs_chla_8'  # , 'obs_chla_12'
                              ]

    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    df_1 = pd.read_csv(os.path.join(fpath, 'data_1.csv'))
    df_3 = pd.read_csv(os.path.join(fpath, 'data_3.csv'))
    df_8 = pd.read_csv(os.path.join(fpath, 'data_8.csv'))
    # df_12 = pd.read_csv(os.path.join(fpath, 'data_12.csv'))

    df_1.index = pd.to_datetime(df_1['date'])
    df_3.index = pd.to_datetime(df_3['date'])
    df_8.index = pd.to_datetime(df_8['date'])
    # df_12.index = pd.to_datetime(df_12['date'])

    if from_config:
        _model = input_model.from_config(config_path=config_path,
                                         data=[df_1, df_3, df_8  # , df_12
                                               ])
        _model.build_nn()
        _model.load_weights(weights)
    else:
        _model = input_model(data_config=data_config,
                             nn_config=nn_config,
                             data=[df_1, df_3, df_8  # , df_12
                                   ],
                             prefix=prefix
                             )
    return _model


if __name__ == "__main__":

    _layers = {'lstm_0': {'config': {'units': 62,  'activation': 'leakyrelu',  'dropout': 0.4, 'recurrent_dropout': 0.4,
                                        'return_sequences': True, 'return_state': False, 'name': 'lstm_0'}},
               'lstm_1': {'config':  {'units': 32,  'activation': 'leakyrelu',  'dropout': 0.4, 'recurrent_dropout': 0.4,
                                        'return_sequences': False, 'return_state': False, 'name': 'lstm_1'}},
               'Dense_0': {'config':  {'units': 16, 'activation': 'leakyrelu'}},
               'Dropout': {'config':  {'rate': 0.4}},
               'Dense_1': {'config':  {'units': 1}}
               }

    model = make_multi_model(MultiInputSharedModel,
                             batch_size=4,
                             lookback=3,
                             lr=0.000216,
                             layers=_layers,
                             epochs=300,
                             ignore_nans=False,
                             )
    model.build()
    model.train(st=0, en=5500)
    # model.predict()

    # cpath = "D:\\playground\\paper_with_codes\\dl_ts_prediction_copy\\results\\lstm_hyper_opt_shared_weights\\20200922_0147\\config.json"
    # w_file = "weights_150_0.0086.hdf5"
    #
    # model = make_multi_model(MultiInputSharedModel, from_config=True, config_path=cpath, weights=w_file)
    # model.predict(st=0, en=5500)
