# dl4seq

[![Build Status](https://travis-ci.com/AtrCheema/dl4seq.svg?branch=master)](https://travis-ci.com/AtrCheema/dl4seq)  
Different deep learning based architechtures for time series forecasting.  

The purpose of the repository is
* compliment the functionality of keras by making pre and post processing easeier for time-series prediction problems
* save, load/reload or build models from readable json file.
* both of above functionalities should be available without complicating simple keras implementation

This repo provides a framework to build layered models using python dictionary and with several helper tools 
which fasten the process of  modeling time-series forcasting. The purpose is to cut the time to write boiler plate code
in developing deep learning based models.

Most of the models in this repository have been adopted from other repositories in order to create an `all in one` code.
I have tried to reference the original repositories as well.

This repository is for you if you want to
* avoid pre and post post processing of data to build NNs for 1D or time series data.
* want to save and reload models in readable json config file.
* Customize some of the utilities provided here while retaining others e.g using your own normalization and denormalization 

Currently following models are implemented

| Name                          | Name in this repository  | Reference |
| -------------------------- | ------------- | ---------- |
| MLP  | `Model` | |
| LSTM | ٭ | |
| CNN  | * |  |
| LSTM CNN | * |  |
| CNN LSTM |  * |  |
| Autoencoder  | * |  |
| ConvLSTM | * | [paper](https://arxiv.org/abs/1506.04214v1) [Code](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/) |
| Temporal Convolutional Networks (TCN)  | * | [paper](https://www.nature.com/articles/s41598-020-65070-5) [code](https://github.com/philipperemy/keras-tcn) |
| Iterpretable Multivariate LSTM (IMV-LSTM)  | `IMVLSTMModel` | [paper](https://arxiv.org/pdf/1905.12034.pdf) [code](https://github.com/KurochkinAlexey/IMV_LSTM) |
| HARHN  | `HARHNModel` | [paper](https://arxiv.org/abs/1806.00685) [code](https://github.com/KurochkinAlexey/Hierarchical-Attention-Based-Recurrent-Highway-Networks-for-Time-Series-Prediction)|
| Neural Beats  | `NBeatsModel` | [paper](https://arxiv.org/pdf/1905.10437.pdf) |
| Dual Attention | `DualAttentionModel` | [paper](https://arxiv.org/pdf/1704.02971.pdf) [code]() |
| Input Attention  | `InputAttentionModel` | |

`*` These models can be constructed by stacking layers in a python dictionary as shown later here. The remaining models 
can be used as shown [here](https://github.com/AtrCheema/dl4seq/blob/master/examples/build_dl_models.md)

## Installation

using github link for the latest code

	python -m pip install git+https://github.com/AtrCheema/dl4seq.git

using setup file, go to folder where repo is downloaded

    python setup.py install

## How to use

```python
import pandas as pd 
from dl4seq import InputAttentionModel  # import any of the above model
from dl4seq.utils import make_model  # helper function to make inputs for model

data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                     lookback=15,
                                                     lr=0.001)
df = pd.read_csv('data/all_data_30min.csv')

model = InputAttentionModel(data_config=data_config,
              nn_config=nn_config,
              data=df,
              intervals=total_intervals
              )

model.build()

history = model.train(indices='random')

preds, obs = model.predict()
acts = model.view_model()
```



## Disclaimer
Athough the purpose of this repo is purportedly `all_in_one` model, however there is no `one_for_all` model. For each
deep learning proble, the model needs to be build accordingly. I made this repo to teach myself deep learning for time
series prediction. 