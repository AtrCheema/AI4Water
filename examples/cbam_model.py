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

data = arg_beach()
input_features=data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

model = Model(
    model={'layers': layers},
    lookback=10,
    train_data='random',
    input_features=input_features,
    output_features=output_features,
)

history = model.fit(data=data)
