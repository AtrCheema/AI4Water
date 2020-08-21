# dl_ts_prediction
Different deep learning based architechtures for time series forecasting.

Most of the models in this repository have been adopted from other repositories in order to create an `all in one` code.
I have tried to reference the original repositories as well.

Currently following models are implemented

| Name                          | Name in this repository  | Reference |
| -------------------------- | ------------- | ---------- |
| LSTM | `Model` | |
| CNN  | `CNNModel` | |
| LSTM CNN | `LSTMCNNModel` | |
| CNN LSTM |  `CNNLSTMModel` | |
| LSTM Autoencoder  | `LSTMAutoEncoder` | |
| Temporal Convolutional Networks (TCN)  | `TCNModel` | [paper](https://www.nature.com/articles/s41598-020-65070-5) [code](https://github.com/philipperemy/keras-tcn) |
| Iterpretable Multivariate LSTM (IMV-LSTM)  | `IMVLSTMModel` | [paper](https://arxiv.org/pdf/1905.12034.pdf) [code](https://github.com/KurochkinAlexey/IMV_LSTM) |
| HARHN  | `HARHNModel` | [paper](https://arxiv.org/abs/1806.00685) [code](https://github.com/KurochkinAlexey/Hierarchical-Attention-Based-Recurrent-Highway-Networks-for-Time-Series-Prediction)|
| Neural Beats  | `NBeatsModel` | [paper](https://arxiv.org/pdf/1905.10437.pdf) |
| Dual Attention | `DualAttentionModel` | [paper](https://arxiv.org/pdf/1704.02971.pdf) [code]() |
| Input Attention  | `InputAttentionModel` | |

## How to use

```python
import pandas as pd 
from models import Model  # import any of the above model
from run_model import make_model  # helper function to make inputs for model

data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                     lookback=15,
                                                     lr=0.001)
df = pd.read_csv('data/all_data_30min.csv')

model = Model(data_config=data_config,
              nn_config=nn_config,
              data=df,
              intervals=total_intervals
              )


model.build_nn()

history = model.train_nn(indices='random')

preds, obs = model.predict()
acts = model.view_model()
```

For more examples see `docs`.

## Disclaimer
Athough the purpose of this repo is purportedly `all_in_one` model, however there is no `one_for_all` model. For each
deep learning proble, the model needs to be build accordingly. I made this repo to teach myself deep learning for time
series prediction. 