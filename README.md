# dl_ts_prediction
Different deep learning based architechtures for time series forecasting.  
This repo provides a framework to build layered models from python dictionary and it provides several helper tools 
which fasten the process of  modeling time-series forcasting. 

Most of the models in this repository have been adopted from other repositories in order to create an `all in one` code.
I have tried to reference the original repositories as well.

Currently following models are implemented

| Name                          | Name in this repository  | Reference |
| -------------------------- | ------------- | ---------- |
| MLP  | `Model` | |
| LSTM | `LSTMModel` | |
| CNN  | * |  |
| LSTM CNN | * | * |
| CNN LSTM |  `CNNLSTMModel` |  |
| Autoencoder  | `AutoEncoder` |  |
| Temporal Convolutional Networks (TCN)  | * | [paper](https://www.nature.com/articles/s41598-020-65070-5) [code](https://github.com/philipperemy/keras-tcn) |
| Iterpretable Multivariate LSTM (IMV-LSTM)  | `IMVLSTMModel` | [paper](https://arxiv.org/pdf/1905.12034.pdf) [code](https://github.com/KurochkinAlexey/IMV_LSTM) |
| HARHN  | `HARHNModel` | [paper](https://arxiv.org/abs/1806.00685) [code](https://github.com/KurochkinAlexey/Hierarchical-Attention-Based-Recurrent-Highway-Networks-for-Time-Series-Prediction)|
| Neural Beats  | `NBeatsModel` | [paper](https://arxiv.org/pdf/1905.10437.pdf) |
| Dual Attention | `DualAttentionModel` | [paper](https://arxiv.org/pdf/1704.02971.pdf) [code]() |
| Input Attention  | `InputAttentionModel` | |
| ConvLSTM | `ConvLSTMModel` | [paper](https://arxiv.org/abs/1506.04214v1) [Code](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/) |

* These models can be constructed by stacking layers in a python dictionary as shown later here. The remaining models 
can be used as shown below
## How to use

```python
import pandas as pd 
from models import InputAttentionModel  # import any of the above model
from run_model import make_model  # helper function to make inputs for model

data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                     lookback=15,
                                                     lr=0.001)
df = pd.read_csv('data/all_data_30min.csv')

model = InputAttentionModel(data_config=data_config,
              nn_config=nn_config,
              data=df,
              intervals=total_intervals
              )


model.build_nn()

history = model.train_nn(indices='random')

preds, obs = model.predict()
acts = model.view_model()
```

## Build using python dictionary

We can construct a normal layered model using keras layers by placing the layers in a dictionary. The keys in the
dictionary must be a keras layer and optionally can have an identifier separated by an underscore `_` in order to 
differentiate it from other similar layers in the model. The input arguments in the layer must be enclosed in a 
dictionary themselves. To find out what input arguments can be used, check documentation of corresponding layer in 
`Tensorflow` docs. 

### multi-layer perceptron

```python
from run_model import make_model
from models import Model

import pandas as pd

layers = {"Dense_0": {'units': 64, 'activation': 'relu'},
          "Dropout_0": {'rate': 0.3},
          "Dense_1": {'units': 32, 'activation': 'relu'},
          "Dropout_1": {'rate': 0.3},
          "Dense_2": {'units': 16, 'activation': 'relu'},
          "Dense_3": {'units': 1}
          }

data_config, nn_config, _ = make_model(batch_size=16,
    lookback=1,
    lr=0.001,
    epochs=2)
nn_config['layers'] = layers

df = pd.read_csv('data/all_data_30min.csv')

_model = Model(data_config=data_config,
              nn_config=nn_config,
              data=df
              )

_model.build_nn()
```
![MLP based model](imgs/mlp.png "Title")


### LSTM based model
```python
from run_model import make_model
from models import LSTMModel
import pandas as pd

layers = {"LSTM_0": {'units': 64, 'return_sequences': True},
          "LSTM_1": {'units': 32},
          "Dropout": {'rate': 0.3},
          "Dense": {'units': 1}
          }
data_config, nn_config, _ = make_model(batch_size=16,
    lookback=1,
    lr=0.001,
    epochs=2)
nn_config['layers'] = layers

df = pd.read_csv("data/all_data_30min.csv")

_model = LSTMModel(data_config=data_config,
              nn_config=nn_config,
              data=df
              )

_model.build_nn()
```

![LSTM based model](imgs/lstm.png "Title")

### 1d CNN based model
If a layer does not receive any input arguments, still an empty dictioanry must be provided.  
Activation functions can also be used as a separate layer.
```python
layers = {"Conv1D_9": {'filters': 64, 'kernel_size': 2},
          "dropout": {'rate': 0.3},
          "Conv1D_1": {'filters': 32, 'kernel_size': 2},
          "maxpool1d": {'pool_size': 2},
          'flatten': {}, # This layer does not receive any input arguments
          'leakyrelu': {},  # activation function can also be used as a separate layer
          "Dense": {'units': 1}
          }
```

![CNN based model](imgs/cnn.png "Title")

### LSTM -> CNN based model
```python
layers = {"LSTM": {'units': 64, 'return_sequences': True},
          "Conv1D_0": {'filters': 64, 'kernel_size': 2},
          "dropout": {'rate': 0.3},
          "Conv1D_1": {'filters': 32, 'kernel_size': 2},
          "maxpool1d": {'pool_size': 2},
          'flatten': {},
          'leakyrelu': {},
          "Dense": {'units': 1}
          }
```
![LSTM->CNN based model](imgs/lstm_cnn.png "Title")


### ConvLSTM based model
```python
layers = {'convlstm2d': {'filters': 64, 'kernel_size': (1, 3), 'activation': 'relu'},
          'flatten': {},
          'repeatvector': {'n': 1},
          'lstm':   {'units': 128,   'activation': 'relu', 'dropout': 0.3, 'recurrent_dropout': 0.4 },
          'Dense': {'units': 1}
          }
```
![ConvLSTM based model](imgs/convlstm.png "Title")


### CNN -> LSTM
If a layer is to be enclosed in `TimeDistributed` layer, just add the layer followed by `TimeDistributed` as shown below.
In following, 3 `Conv1D` layer are enclosed in `TimeDistributed` layer. Similary `Flatten` and `MaxPool1D` are also 
wrapped in `TimeDistributed` layer.
```python
layers = {
    "TimeDistributed_0": {},
    'conv1d_0': {'filters': 64, 'kernel_size': 2},
    'LeakyRelu_0': {},
    "TimeDistributed_1": {},
    'conv1d_1': {'filters': 32, 'kernel_size': 2},
    'elu_1': {},
    "TimeDistributed_2": {},
    'conv1d_2': {'filters': 16, 'kernel_size': 2},
    'tanh_2': {},
    "TimeDistributed_3": {},
    "maxpool1d": {'pool_size': 2},
    "TimeDistributed_4": {},
    'flatten': {},
    'lstm_0':   {'units': 64, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5, 'return_sequences': True,
               'name': 'lstm_0'},
    'Relu_1': {},
    'lstm_1':   {'units': 32, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5, 'name': 'lstm_1'},
    'sigmoid_2': {},
    'Dense': {'units': 1}
}
```
![CNN -> LSTM model](imgs/cnn_lstm.png "Title")


### LSTM based auto-encoder
```python
layers = {
    'lstm_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
    "leakyrelu_0": {},
    'RepeatVector': {'n': 11},
    'lstm_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
    "relu_1": {},
    'Dense': {'units': 1}
}
```
![LSTM auto-encoder](imgs/lstm_autoenc.png "Title")


### TCN layer
```python
layers = {"tcn": {'nb_filters': 64,
                  'kernel_size': 2,
                  'nb_stacks': 1,
                  'dilations': [1, 2, 4, 8, 16, 32],
                  'padding': 'causal',
                  'use_skip_connections': True,
                  'return_sequences': False,
                  'dropout_rate': 0.0},
          'Dense': {'units': 1}
          }
```
![TCN layer](imgs/tcn.png "Title")


For more examples see `docs`.

## Disclaimer
Athough the purpose of this repo is purportedly `all_in_one` model, however there is no `one_for_all` model. For each
deep learning proble, the model needs to be build accordingly. I made this repo to teach myself deep learning for time
series prediction. 