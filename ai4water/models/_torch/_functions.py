
from typing import Union

from ..utils import _check_length


def MLP(
        units: Union[int, list] = 32,
        num_layers: int = 1,
        input_shape: tuple = None,
        num_outputs: int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        mode: str = "regression",
        output_activation: str = None,
        **kwargs
):
    """makes layers for MLP model for tensorflow"""
    assert num_layers>=1

    assert input_shape is not None

    if isinstance(units, int):
        units = [units]*num_layers

    if not isinstance(activation, list):
        activation = [activation] * num_layers

    if not isinstance(dropout, list):
        dropout = [dropout] * num_layers

    in_feat = input_shape[-1]
    out_feat = units[0]

    layers = {}
    for i in range(num_layers):

        lyr = {f"Linear_{i}": {"in_features": in_feat, "out_features": out_feat}}

        in_feat = out_feat
        out_feat = units[i]

        layers.update(lyr)

        if activation[i]:
            layers.update({f'{activation[i]}_{i}': {}})

        if dropout[i]:
            layers.update({f'Dropout_{i}': dropout[i]})


    layers = _make_output_layer(
        layers,
        mode,
        out_feat,
        num_outputs,
        output_activation
    )

    return {"layers": layers}


def LSTM(
        units: Union[int, list] = 32,
        num_layers: int = 1,
        input_shape: tuple = None,
        num_outputs: int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = 0.,
        mode: str = "regression",
        output_activation: str = None,
        **kwargs
):
    """
    helper function to make LSTM layers for pytorch
    """
    assert isinstance(input_shape, (tuple, list))
    assert len(input_shape)==2

    if dropout is None:
        dropout = 0.

    layers = {
        'LSTM_0': {"config": dict(
            input_size=input_shape[1],
            hidden_size= units,
            batch_first= True,
            num_layers= num_layers,
            dropout=dropout,
            **kwargs
        ),
            "outputs": ['lstm_output', 'states_0']
        },
    'slice': {"config": lambda x: x[:, -1, :],   #  we want to get the output from last lookback step.
              "inputs": "lstm_output"},
    }

    layers = _make_output_layer(
        layers,
        mode,
        units,
        num_outputs,
        output_activation
    )

    return {"layers": layers}


def CNN(
        filters: Union[int, list] = None,
        out_channels = 32,
        kernel_size: Union[int, tuple, list] = 3,
        convolution_type: str = "1D",
        num_layers: int = 1,
        padding: Union[str, list] = "same",
        stride: Union[int, list] = 1,
        pooling_type: Union[str, list] = None,
        pool_size: Union[int, list] = 2,
        batch_normalization: Union[bool, list] = None,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        input_shape: tuple = None,
        num_outputs: int = 1,
        mode: str = "regression",
        output_activation: str = None,
        **kwargs,
):
    """
    helper function to make CNN model with pytorch as backend
    """
    out_channels = _check_length(out_channels, num_layers)
    #activation = _check_length(activation, num_layers)
    padding = _check_length(padding, num_layers)
    stride = _check_length(stride, num_layers)
    pooling_type = _check_length(pooling_type, num_layers)
    pool_size = _check_length(pool_size, num_layers)
    kernel_size = _check_length(kernel_size, num_layers)
    batch_normalization = _check_length(batch_normalization, num_layers)
    dropout = _check_length(dropout, num_layers)

    assert isinstance(input_shape, (tuple, list))
    assert len(input_shape)==2

    in_feat = input_shape[-1]
    out_feat = out_channels[0]

    layers = {}
    for idx in range(num_layers):

        pool_type = pooling_type[idx]
        batch_norm = batch_normalization[idx]

        config = {
            "kernel_size": kernel_size[idx],
            "stride": stride[idx],
            "padding": padding[idx],
            "in_channels": in_feat,
            "out_channels": out_feat
        }

        if kwargs is not None:
            config.update(kwargs)

        layers.update({f"Conv1d_{idx}": config})

        if pool_type:
            pool_lyr = f"{pool_type}{convolution_type}"
            _lyr = {pool_lyr: {"pool_size": pool_size[idx]}}
            layers.update(_lyr)

        if batch_norm:
            layers.update({"BatchNormalization": {}})

        if dropout[idx]:
            layers.update({f'Dropout_{idx}': dropout[idx]})

    layers = _make_output_layer(
        layers,
        mode,
        out_feat,
        num_outputs,
        output_activation
    )

    return {"layers": layers}


def _make_output_layer(
        layers,
        mode,
        in_features,
        num_outputs,
        output_activation
):
    if output_activation is None and mode == "classification":
        # for binary it is better to use sigmoid
        if num_outputs > 2:
            output_activation = "softmax"
        else:
            output_activation = "sigmoid"
            num_outputs = 1

    layers.update(
        {"Linear_out": {"in_features": in_features, 'out_features': num_outputs}})
    return layers
