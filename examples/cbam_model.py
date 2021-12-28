# Put channel and spatial attention of CBAM model for time-series prediction


from ai4water import Model
from ai4water.datasets import arg_beach


layers = {
    "Conv1D": {"filters": 64, "kernel_size": 7},
    "MaxPool1D": {},
    "ChannelAttention": {"conv_dim": "1d", "in_planes": 32},
    "SpatialAttention": {"conv_dim": "1d"},

    "Flatten": {},
    "Dense": 1
}

model = Model(
    model={'layers': layers},
    lookback=10,
    train_data='random',
    data=arg_beach())

history = model.fit()
