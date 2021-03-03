# Put channel and spatial attention of CBAM model for time-series prediction


from dl4seq import Model
from dl4seq.data import load_30min

inputs = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
                  'input11']
outputs = ['target7']

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
    inputs=inputs,
    outputs=outputs,
    data=load_30min())

history = model.fit(indices="random")
