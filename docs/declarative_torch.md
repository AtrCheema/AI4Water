This page describes how to build Neural Networks for pytorch using pythron dictionary in `AI4Water`.
The user can use any layer provided by pytorch such as `Linear` or `LSTM`. Similarly the user
can use any input argument allowed by the particular layer e.g. `bidirectional` for 
[`LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and `out_features` for 
[`Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).

All the examples presented here are similar which were shown for [tensorflow's case](https://ai4water.readthedocs.io/en/latest/build_dl_models.html)
### multi-layer perceptron

```python

from AI4Water.utils.datasets import arg_beach
from AI4Water import Model

layers = {
    "Linear_0": {"in_features": 13, "out_features": 64},
    "ReLU_0": {},
    "Dropout_0": 0.3,
    "Linear_1": {"in_features": 64, "out_features": 32},
    "ReLU_1": {},
    "Dropout_1": 0.3,
    "Linear_2": {"in_features": 32, "out_features": 16},
    "Linear_3": {"in_features": 16, "out_features": 1},
}

model = Model(
    model={'layers': layers},
    data=arg_beach())
```

If we want to do slicing of the outputs of one layer, we can use pythons's `lambda` function. 
In fact any `callable` object can be proivded

### LSTM based model
```python
layers ={
    'LSTM_0': {"config": {'input_size': 13, 'hidden_size': 64, "batch_first": True},
               "outputs": ['lstm0_output', 'states_0']},  # LSTM in pytorch returns two values see docs
    'LSTM_1': {"config": {'input_size': 64, 'hidden_size': 32, "batch_first": True, "dropout": 0.3},
               "outputs": ["lstm1_output", 'states_1'],
               "inputs": "lstm0_output"},
    'slice': {"config": lambda x: x[:, -1, :],   #  we want to get the output from last lookback step.
              "inputs": "lstm1_output"},
    "Linear": {"in_features": 32, "out_features": 1},
}
```