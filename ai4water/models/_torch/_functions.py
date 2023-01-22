
from typing import Union

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
