# Put channel and spatial attention of CBAM model for time-series prediction


from AI4Water import Model
from AI4Water.utils.datasets import arg_beach


layers = {
    "Conv1D": {"filters": 64, "kernel_size": 7},
    "MaxPool1D": {},
    "ChannelAttention": {"conv_dim": "1d", "in_planes": 32},
    "SpatialAttention": {"conv_dim": "1d"},

    "Flatten": {},
    "Dense": 1,
    "Reshape": {"target_shape": (1,1)}
}

model = Model(
    model={'layers':layers},
    lookback=10,
    data=arg_beach())

history = model.fit(indices="random")
