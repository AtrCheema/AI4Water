# Put channel and spatial attention of CBAM model for time-series prediction


from AI4Water import Model
from AI4Water.utils.datasets import arg_beach


layers = {
    "Conv1D": {"config": {"filters": 64, "kernel_size": 7}},
    "MaxPool1D": {"config": {}},
    "ChannelAttention": {"config": {"conv_dim": "1d", "in_planes": 32}},
    "SpatialAttention": {"config": {"conv_dim": "1d"}},

    "Flatten": {"config": {}},
    "Dense": {"config": {"units": 1}},
    "Reshape": {"config": {"target_shape": (1,1)}}
}

model = Model(
    model={'layers':layers},
    lookback=10,
    data=arg_beach())

history = model.fit(indices="random")
