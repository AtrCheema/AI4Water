import pandas as pd

from run_model import make_model
from models import Model, LSTMModel, CNNModel, CNNLSTMModel, LSTMCNNModel, ConvLSTMModel, InputAttentionModel
from models import DualAttentionModel


def make_and_run(input_model, lookback=12, epochs=4, **kwargs):

    input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                      'rel_hum']
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml']

    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=lookback,
                                                         lr=0.001,
                                                         inputs=input_features,
                                                         outputs = outputs,
                                                         epochs=epochs,
                                                         **kwargs)
    df = pd.read_csv('../data/all_data_30min.csv')

    model = input_model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  intervals=total_intervals
                  )

    model.build_nn()

    _ = model.train_nn(indices='random')

    _ = model.predict(use_datetime_index=False)

    return

dense_config = {32: {'units': 64, 'activation': 'relu', 'dropout_layer': 0.3},
                                 16: {'units': 32, 'activation': 'relu', 'dropout_layer': 0.3},
                                 8: {'units': 16, 'activation': 'relu'},
                                 1: {'units': 1}
                                 }

make_and_run(Model, lookback=1, dense_config=dense_config)

##
# LSTM based model
make_and_run(LSTMModel)


##
# CNN based model
make_and_run(CNNModel)

##
# CNNLSTM based model
lstm_config = {'units': 64,  # for more options https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
                                 'activation': 'relu',  # activation inside LSTM
                                 'dropout': 0.4,
                                 'recurrent_dropout': 0.5,
                                 'return_sequences':False
               }
make_and_run(CNNLSTMModel, lstm_config=lstm_config)

##
# LSTMCNNModel based model
make_and_run(LSTMCNNModel)

##
# ConvLSTMModel based model
make_and_run(ConvLSTMModel)

##
# InputAttentionModel based model
make_and_run(InputAttentionModel)

##
# DualAttentionModel based model
make_and_run(DualAttentionModel)