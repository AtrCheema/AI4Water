
from typing import Union, List

from ai4water.backend import tf
from ..utils import _make_output_layer, _check_length


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

    units = _check_length(units, num_layers)
    dropout = _check_length(dropout, num_layers)
    activation = _check_length(activation, num_layers)

    if input_shape is None:
        layers = {}
    else:
        layers =   {"Input": {"shape": input_shape}}

    for idx, lyr in enumerate(range(num_layers)):

        config = {"units": units[idx],
                  "activation": activation[idx],
                  }
        config.update(kwargs)

        _lyr = {f"Dense_{lyr}": config}
        layers.update(_lyr)

        _dropout = dropout[idx]

        if  _dropout and _dropout > 0.0:
            layers.update({"Dropout": {"rate": _dropout}})

    layers.update({"Flatten": {}})

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )

    return {'layers': layers}


def LSTM(
        units: Union[int, list] = 32,
        num_layers:int = 1,
        input_shape: tuple = None,
        num_outputs:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        mode:str = "regression",
        output_activation:str = None,
        **kwargs
):
    """helper function to make LSTM Model for tensorflow
    """

    units, dropout, activation, layers = _precheck(
        num_layers, input_shape, units, dropout, activation)

    for idx, lyr in enumerate(range(num_layers)):

        return_sequences = False
        if idx+1 != num_layers:
            return_sequences = True

        config = {"units": units[idx],
                  "activation": activation[idx],
                  "return_sequences": return_sequences,
                                }
        config.update(kwargs)

        _lyr = {f"LSTM_{lyr}": config}
        layers.update(_lyr)

        _dropout = dropout[idx]
        if  _dropout and _dropout > 0.0:
            layers.update({"Dropout": {"rate": _dropout}})

    layers.update({"Flatten": {}})

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )

    return {'layers': layers}


def AttentionLSTM(
        units: Union[int, list] = 32,
        num_layers: int = 1,
        input_shape: tuple = None,
        num_outputs: int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        atten_units:int = 128,
        atten_activation:str = "tanh",
        mode: str = "regression",
        output_activation: str = None,
        **kwargs
)->dict:
    """helper function to make AttentionLSTM Model for tensorflow"""

    units, dropout, activation, layers = _precheck(
        num_layers, input_shape, units, dropout, activation)

    for idx, lyr in enumerate(range(num_layers)):

        config = {"units": units[idx],
                  "activation": activation[idx],
                  "return_sequences": True,
                  }
        config.update(kwargs)

        _lyr = {f"LSTM_{lyr}": config}
        layers.update(_lyr)

        _dropout = dropout[idx]
        if _dropout and _dropout > 0.0:
            layers.update({"Dropout": {"rate": _dropout}})

    attn_layer = {"SelfAttention":
                      {"config": {"units": atten_units, "activation": atten_activation},
                       "outputs": ["atten_outputs", "atten_weights"]}}
    layers.update(attn_layer)

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation,
        inputs="atten_outputs"
    )

    return {'layers': layers}


def CNN(
        filters: Union[int, list] = 32,
        kernel_size: Union[int, tuple, list] = 3,
        convolution_type: str = "1D",
        num_layers: int = 1,
        padding: Union[str, list] = "same",
        strides: Union[int, list]= 1,
        pooling_type: Union[str, list] = None,
        pool_size: Union[int, list] = 2,
        batch_normalization: Union[bool, list] = None,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        input_shape: tuple = None,
        num_outputs:int = 1,
        mode: str = "regression",
        output_activation:str = None,
        **kwargs
)->dict:
    """helper function to make convolution neural network based model for tensorflow
    """

    assert num_layers>=1

    assert convolution_type in ("1D", "2D", "3D")
    assert pooling_type in ("MaxPool", "AveragePooling", None)

    filters = _check_length(filters, num_layers)
    activation = _check_length(activation, num_layers)
    padding = _check_length(padding, num_layers)
    strides = _check_length(strides, num_layers)
    pooling_type = _check_length(pooling_type, num_layers)
    pool_size = _check_length(pool_size, num_layers)
    kernel_size = _check_length(kernel_size, num_layers)
    batch_normalization = _check_length(batch_normalization, num_layers)
    dropout = _check_length(dropout, num_layers)

    if input_shape is None:
        layers = {}
    else:
        assert len(input_shape) >= 2
        layers =   {"Input": {"shape": input_shape}}

    for idx, lyr in enumerate(range(num_layers)):

        pool_type = pooling_type[idx]
        batch_norm = batch_normalization[idx]

        config = {
            "filters": filters[idx],
            "kernel_size": kernel_size[idx],
            "activation": activation[idx],
            "strides": strides[idx],
            "padding": padding[idx]
        }

        config.update(kwargs)

        _lyr = {f"Conv{convolution_type}_{lyr}": config}

        layers.update(_lyr)

        if pool_type:
            pool_lyr = f"{pool_type}{convolution_type}"
            _lyr = {pool_lyr: {"pool_size": pool_size[idx]}}
            layers.update(_lyr)

        if batch_norm:
            layers.update({"BatchNormalization": {}})

        _dropout = dropout[idx]
        if  _dropout and _dropout > 0.0:
            layers.update({"Dropout": {"rate": _dropout}})

    layers.update({"Flatten": {}})

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )

    return {'layers': layers}



def CNNLSTM(
        input_shape:tuple,
        sub_sequences=3,
        cnn_layers:int = 2,
        lstm_layers:int = 1,
        filters:Union[int, list]=32,
        kernel_size: Union[int, tuple, list]=3,
        max_pool:bool=False,
        units: Union[int, tuple, list] = 32,
        num_outputs:int = 1,
        mode:str = "regression",
        output_activation:str = None,
)->dict:
    """
    helper function to make CNNLSTM model for tensorflow

    """

    assert len(input_shape) == 2
    layers =   {"Input": {"shape": input_shape}}
    lookback = input_shape[-2]
    input_features = input_shape[-1]
    time_steps = lookback // sub_sequences
    new_shape = sub_sequences, time_steps, input_features

    layers.update({"Reshape": {"target_shape": new_shape}})

    filters = _check_length(filters, cnn_layers)
    kernel_size = _check_length(kernel_size, cnn_layers)

    units = _check_length(units, lstm_layers)

    for idx, cnn_lyr in enumerate(range(cnn_layers)):

        layers.update({f"TimeDistributed_{idx}": {}})

        config = {"filters": filters[idx],
                  "kernel_size": kernel_size[idx],
                  "padding": "same"}
        layers.update({f"Conv1D_{idx}": config})

        if max_pool:
            layers.update({f"TimeDistributed_mp{idx}": {}})

            layers.update({f"MaxPool1D_{idx}": {}})

    layers.update({"TimeDistributed_e": {}})
    layers.update({'Flatten': {}})

    for lstm_lyr in range(lstm_layers):

        return_sequences = False
        if lstm_lyr+1 != lstm_layers:
            return_sequences = True

        config = {"units": units[lstm_lyr],
                  "return_sequences": return_sequences
                  }
        layers.update({f"LSTM_{lstm_lyr}": config})

    layers.update({"Flatten": {}})

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )

    return {"layers": layers}


def LSTMAutoEncoder(
        input_shape:tuple,
        encoder_layers:int = 1,
        decoder_layers:int = 1,
        encoder_units: Union[int, list]=32,
        decoder_units: Union[int, list]=32,
        num_outputs: int = 1,
        prediction_mode: bool = True,
        mode:str = "regression",
        output_activation: str = None,
        **kwargs
)->dict:
    """
    helper function to make LSTM based AutoEncoder model for tensorflow
    """

    assert len(input_shape)>=2
    assert encoder_layers >= 1
    assert decoder_layers >= 1

    lookback = input_shape[-2]

    encoder_units = _check_length(encoder_units, encoder_layers)
    decoder_units = _check_length(decoder_units, decoder_layers)

    layers =   {"Input": {"shape": input_shape}}

    for idx, enc_lyr in enumerate(range(encoder_layers)):

        return_sequences = False
        if idx + 1 != encoder_layers:
            return_sequences = True

        config = {
            "units": encoder_units[idx],
            "return_sequences": return_sequences
        }
        lyr = {f"LSTM_e{idx}": config}
        layers.update(lyr)

    layers.update({'RepeatVector': lookback})

    for idx, dec_lyr in enumerate(range(decoder_layers)):

        return_sequences = False
        if idx + 1 != decoder_units:
            return_sequences = True

        config = {
            "units": decoder_units[idx],
            "return_sequences": return_sequences
        }
        lyr = {f"LSTM_d{idx}": config}
        layers.update(lyr)

    layers.update({"Flatten": {}})

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )
    return {'layers': layers}


def TCN(
        input_shape,
        filters:int = 32,
        kernel_size: int = 2,
        nb_stacks: int = 1,
        dilations = [1, 2, 4, 8, 16, 32],
        num_outputs:int = 1,
        mode="regression",
        output_activation: str = None,
        **kwargs
)->dict:
    """helper function for building temporal convolution network with tensorflow

    """

    layers = {"Input": {"shape": input_shape}}

    config = {'nb_filters': filters,
              'kernel_size': kernel_size,
              'nb_stacks': nb_stacks,
              'dilations': dilations,
              'padding': 'causal',
              'use_skip_connections': True,
              'return_sequences': False,
              'dropout_rate': 0.0}

    config.update(kwargs)
    layers.update({"TCN": config})

    layers.update({"Flatten": {}})

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )
    return {'layers': layers}




def TFT(
        input_shape,
        hidden_units: int = 32,
        num_heads: int = 3,
        dropout:float = 0.1,
        num_outputs:int = 1,
        use_cudnn:bool = False,
        mode:str="regression",
        output_activation:str = None,
)->dict:
    """helper function for temporal fusion transformer based model for tensorflow
    """

    num_encoder_steps = input_shape[-2]
    input_features = input_shape[-1]

    params = {
        'total_time_steps': num_encoder_steps,
        'num_encoder_steps': num_encoder_steps,
        'num_inputs': input_features,
        'category_counts': [],
        'input_obs_loc': [],  # leave empty if not available
        'static_input_loc': [],  # if not static inputs, leave this empty
        'known_regular_inputs': list(range(input_features)),
        'known_categorical_inputs': [],  # leave empty if not applicable
        'hidden_units': hidden_units,
        'dropout_rate': dropout,
        'num_heads': num_heads,
        'use_cudnn': use_cudnn,
        'future_inputs': False,
        'return_sequences': True,
    }

    layers = {
        "Input": {"config": {"shape": (params['total_time_steps'], input_features)}},
        "TemporalFusionTransformer": {"config": params},
        "lambda": {"config": tf.keras.layers.Lambda(lambda _x: _x[Ellipsis, -1, :])},
    }

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )

    return {'layers': layers}


def TabTransformer(
        num_numeric_features: int,
        cat_vocabulary: dict,
        hidden_units=32,
        num_heads: int = 4,
        depth: int = 4,
        dropout: float = 0.1,
        num_dense_lyrs: int = 2,
        prenorm_mlp:bool = True,
        post_norm: bool = True,
        final_mlp_units: Union[int, List[int]] = 16,
        num_outputs: int = 1,
        mode: str = "regression",
        output_activation: str = None,
        seed:int = 313,
):
    """
    TabTransformer for tensorflow
    """
    layers = _make_input_lyrs(num_numeric_features, len(cat_vocabulary))
    layers.update(
        {"TabTransformer": {"config": dict(
            cat_vocabulary=cat_vocabulary,
            num_numeric_features=num_numeric_features,
            hidden_units = hidden_units,
            num_heads = num_heads,
            depth= depth,
            dropout = dropout,
            num_dense_lyrs = num_dense_lyrs,
            prenorm_mlp = prenorm_mlp,
            post_norm = post_norm,
            final_mlp_units = final_mlp_units,
            seed = seed,
        ),
        "inputs": ["Input_num", "Input_cat"],
        "outputs": ['transformer_output', 'imp']}}
    )

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation,
        inputs="transformer_output"
    )

    return {"layers": layers}


def FTTransformer(
        num_numeric_features:int,
        cat_vocabulary:dict=None,
        hidden_units = 32,
        num_heads: int = 4,
        depth:int = 4,
        dropout: float = 0.1,
        num_dense_lyrs: int = 2,
        lookup_kws:dict = None,
        post_norm:bool = True,
        final_mlp_units:int = 16,
        num_outputs: int = 1,
        mode: str = "regression",
        output_activation: str = None,
        seed:int = 313,
)->dict:
    """
    FTTransformer for tensorflow
    """
    if cat_vocabulary:
        layers = _make_input_lyrs(num_numeric_features, len(cat_vocabulary))
        inputs = ["Input_num", "Input_cat"]
    else:
        layers = _make_input_lyrs(num_numeric_features)
        inputs = "Input_num"

    body = {

        "FTTransformer": {"config": dict(
            cat_vocabulary=cat_vocabulary,
            num_numeric_features=num_numeric_features,
            hidden_units=hidden_units,
            num_heads=num_heads,
            depth=depth,
            lookup_kws=lookup_kws,
            num_dense_lyrs=num_dense_lyrs,
            post_norm=post_norm,
            final_mlp_units=final_mlp_units,
            dropout=dropout,
            seed=seed
        ),
            "inputs": inputs,
            "outputs": ['transformer_output', 'imp']}
    }

    layers.update(body)

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation,
        inputs='transformer_output'
    )

    return {"layers": layers}



def _make_input_lyrs(
        num_numeric_features:int,
        num_cat_features:int=None,
        numeric_dtype:tf.DType=tf.float32,
        cat_dtype:tf.DType = tf.string
)->dict:

    if num_cat_features:
        lyrs = {
            "Input_num": {"shape": (num_numeric_features,), "dtype": numeric_dtype},
            "Input_cat": {"shape": (num_cat_features,), "dtype": cat_dtype}
        }
    else:
        lyrs = {
            "Input_num": {"shape": (num_numeric_features,), "dtype": numeric_dtype}
        }

    return lyrs


def _precheck(num_layers, input_shape, units, dropout, activation):
    assert num_layers>=1

    if input_shape is None:
        layers = {}
    else:
        layers = {"Input": {"shape": input_shape}}
        assert len(input_shape)>=2

    units = _check_length(units, num_layers)
    dropout = _check_length(dropout, num_layers)
    activation = _check_length(activation, num_layers)
    return units, dropout, activation, layers
