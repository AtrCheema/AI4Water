# dl_ts_prediction
Different deep learning based architechtures for time series forecasting.

Most of the models in this repository have been adopted from other repositories in order to create an `all in one` code. I have tried to reference the original repositories as well.

Currently following models are implemented

| Name                          | Name in this repository  |
| -------------------------- | ------------- |
| LSTM | `Model` |
| CNN  | `CNNModel` |
| LSTM CNN | `LSTMCNNModel` |
| CNN LSTM |  `CNNLSTMModel` |
| LSTM Autoencoder  | `LSTMAutoEncoder` |
| Temporal Convolutional Networks (TCN)  | `TCNModel` |
| Iterpretable Multivariate LSTM (IMV-LSTM)  | `IMVLSTMModel` |
| HARHN  | `HARHNModel` |
| Neural Beats  | `NBeatsModel` |
| Dual Attention | `DualAttentionModel` |
| Input Attention  | `InputAttentionModel` |

** How to use

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