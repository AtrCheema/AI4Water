
from typing import Union


def MLP(
        input_shape:tuple,
        units: Union[int, list] = 32,
        num_layers:int = 1,
        output_features:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        problem:str = "regression",
        **kwargs
)->dict:
    """helper function to make multi layer perceptron model

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model
    units : Union[int, list], default=32
        number of units in Dense layer
    num_layers : int, optional, (default, 32)
        number of Dense_ layers to use
    output_features : int, optional
        number of output features from the network
    activation : Union[str, list], optional
        activation function to use.
    dropout : Union[float, list], optional
        dropout to use in Dense layer
    problem : str, optional
        either ``regression`` or ``classification``
    **kwargs :
        any additional keyword arguments for Dense_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key

    Examples
    --------
    >>> MLP((10, ),32)

    >>> MLP((10, ), 32,  3)

    >>> MLP((5, 10), 32,  3)

    >>> MLP((10, 1), [32, 16, 8], 3, activation="relu")

    >>> MLP((10, ), 32, 3, use_bias=True)

    .. _Dense:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

    """
    assert num_layers>=1

    units = _check_length(units, num_layers)
    dropout = _check_length(dropout, num_layers)
    activation = _check_length(activation, num_layers)

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

    layers.update({"Dense_out": {"units": output_features,
                   "activation": None if problem=="regression" else "softmax"
                   }})
    return {'layers': layers}


def LSTM(
        input_shape:tuple,
        units: Union[int, list] = 32,
        num_layers:int = 1,
        output_features:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        problem:str = "regression",
        **kwargs
):
    """helper function to make LSTM Model

    Parameters
    ----------
    input_shape :
        shape of input tensor to the model
    units : Union[int, list], optional (default 32)
        number of units in LSTM layer
    num_layers :
        number of lstm layers to use
    output_features : int, optinoal (default=1)
        number of output features
    activation : Union[str, list], optional
        activation function to use in LSTM
    dropout :
        if > 0.0, a dropout layer is added after each LSTM layer
    problem : str, optional
        either ``regression`` or ``classification``
    **kwargs :
        any keyword argument for LSTM_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key

    Examples
    --------
    >>> LSTM((5, 10))
    # to build a model with stacking of LSTM layers
    >>> LSTM((5, 10), 32, num_layers=2)

    .. _LSTM:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
    """
    assert num_layers>=1
    assert len(input_shape)>=2

    units = _check_length(units, num_layers)
    dropout = _check_length(dropout, num_layers)
    activation = _check_length(activation, num_layers)

    layers =   {"Input": {"shape": input_shape}}

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

    layers.update({"Dense_out": {"units": output_features,
                   "activation": None if problem=="regression" else "softmax"
                   }})
    return {'layers': layers}


def CNN(
        input_shape:tuple,
        convolution_type: str = "1D",
        filters: Union[int, list] = 32,
        kernel_size: Union[int, tuple, list] = 3,
        num_layers: int = 1,
        padding: Union[str, list] = "same",
        strides: Union[int, list]= 1,
        pooling_type: Union[str, list] = None,
        pool_size: Union[int, list] = 2,
        batch_normalization: Union[bool, list] = None,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        output_features:int = 1,
        problem: str = "regression",
        **kwargs
)->dict:
    """helper function to make convolution neural network based model.

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model
    convolution_type : str, optional, (default="1D")
        either ``1D`` or ``2D`` or ``3D``
    filters : Union[int, list], optional
        number of filters in convolution layer. If given as list, it should
        be equal to ``num_layers``.
    kernel_size : Union[int, list], optional
        kernel size in (each) convolution layer
    num_layers : int, optional
        number of convolution layers to use. Should be > 0.
    padding : Union[str, list], optional
        padding to use in (each) convolution layer
    strides : Union[int, list], optional
        strides to use in (each) convolution layer
    pooling_type : str, optional
        either "MaxPool" or "AveragePooling"
    pool_size : Union[int, list], optional
        only valid if pooling_type is not None
    batch_normalization :
        whether to use batch_normalization after each convolution or
        convolution+pooling layer. If true, a batch_norm_ layer
        is added.
    activation : Union[str, list], optional
        activation function to use in convolution layer
    dropout : Union[float, list], optional
        if > 0.0, a dropout layer is added after each LSTM layer
    output_features : int, optional
        number of output features
    problem : str, optional
        either ``regression`` or ``classification``
    **kwargs :
        any keyword argument for Convolution_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key

    Examples
    --------
    >>> CNN((5, 10), "1D", 32, 2)

    >>> CNN((5, 10), "1D", 32, 2, pooling_type="MaxPool")


    .. _Convolution:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
    .. _batch_norm:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

    """

    assert num_layers>=1
    assert len(input_shape) >= 2

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

    layers.update({"Dense_out": {"units": output_features,
                   "activation": None if problem=="regression" else "softmax"
                   }})
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
        output_features:int = 1,
        problem:str = "regression"
)->dict:
    """
    helper function to make CNNLSTM model. It adds one or more 1D convolutional
    layers before one or more LSTM layers.

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model
    sub_sequences : int
        number of sub_sequences in which to divide the input before applying
        Conv1D on it.
    cnn_layers : int , optional (default=2)
        number of cnn layers
    lstm_layers :
        number of lstm layers
    filters : Union[int, list], optional
        number of filters in (each) cnn layer
    kernel_size : Union[int, tuple, list], optional
        kernel size in (each) cnn layer
    max_pool : bool, optional (default=True)
        whether to use max_pool after every cnn layer or not
    units : Union[int, list], optional (default=32)
        number of units in (each) lstm layer
    output_features : int, optional (default=1)
        number of output features
    problem :
        either ``regression`` or ``classification``

    Returns
    -------
    dict :
        a dictionary with ``layers`` as key

    Examples
    --------

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

    layers.update({"Dense_out": {"units": output_features,
                   "activation": None if problem=="regression" else "softmax"
                   }})
    return {"layers": layers}


def LSTMAutoEncoder(
        input_shape:tuple,
        encoder_layers:int = 1,
        decoder_layers:int = 1,
        encoder_units: Union[int, list]=32,
        decoder_units: Union[int, list]=32,
        output_features: int = 1,
        mode: str = "prediction",
        problem:str = "regression",
        **kwargs
)->dict:
    """
    helper function to make LSTM based AutoEncoder model.

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model
    encoder_layers : int, optional (default=1)
        number of encoder LSTM layers
    decoder_layers : int, optional (default=1)
        number of decoder LSTM layers
    encoder_units : Union[int, list], optional, (default=32)
        number of units in (each) encoder LSTM
    decoder_units : Union[int, list], optional, (default=32)
        number of units in (each) decoder LSTM
    mode : str, optional (default="prediction")
        either "prediction" or "reconstruction"
    output_features : int, optional
        number of output features
    problem : str, optional
        either ``regression`` or ``classification``
    **kwargs

    Returns
    -------
    dict :
        a dictionary with ``layers`` as key

    Examples
    --------
    >>> LSTMAutoEncoder((5, 10), 2, 2, 32, 32)

    >>> LSTMAutoEncoder((5, 10), 2, 2, [64, 32], [32, 64])
    """
    assert len(input_shape)>=2
    assert encoder_layers >= 1
    assert decoder_layers >= 1

    lookback = input_shape[-2]

    encoder_units = _check_length(encoder_units, encoder_layers)
    decoder_units = _check_length(decoder_units, decoder_layers)

    layers =   {"Input": {"shape": input_shape}}

    for idx, enc_lyr in enumerate(range(encoder_layers)):

        config = {
            "units": encoder_units[idx]
        }
        lyr = {f"LSTM_e{idx}": config}
        layers.update(lyr)

    layers.update({'RepeatVector': lookback})

    for idx, dec_lyr in enumerate(range(decoder_layers)):

        config = {
            "units": decoder_units[idx]
        }
        lyr = {f"LSTM_d{idx}": config}
        layers.update(lyr)

    layers.update({"Flatten": {}})

    layers.update({"Dense_out": {"units": output_features,
                   "activation": None if problem=="regression" else "softmax"
                   }})
    return {'layers': layers}


def TCN(
        input_shape,
        filters:int = 32,
        kernel_size: int = 2,
        nb_stacks: int = 1,
        dilations = [1, 2, 4, 8, 16, 32],
        output_features:int = 1,
        problem="regression",
        **kwargs
)->dict:
    """helper function for building temporal convolution network

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model
    filters : int, optional (default=32)
        number of filters
    kernel_size : int, optional (default=2)
        kernel size
    nb_stacks : int, optional (default=
        number of stacks of tcn layer
    dilations :
        dilation rate
    output_features : int, optional
        number of output features
    problem : str, optional
        either ``regression`` or ``classification``
    **kwargs
        any additional keyword argument

    Returns
    -------
    dict :
        a dictionary with ``layers`` as key
    Examples
    --------
    >>> TCN((5, 10), 32)

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

    layers.update({"Dense_out": {"units": output_features,
                   "activation": None if problem=="regression" else "softmax"
                   }})
    return {'layers': layers}


def TFT(
        input_shape,
        hidden_units: int = 32,
        num_heads: int = 3,
        dropout:float = 0.1,
        output_features:int = 1,
        use_cudnn=False,
        problem="regression",
)->dict:
    """helper function for temporal fusion transformer based model

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model
    hidden_units : int, optional (default=32)
        number of hidden units
    num_heads : int, optional (default=1)
        number of attention heads
    dropout : int, optional (default=0.1)
        droput rate
    output_features : int, optional (default=1)
        number of output features
    use_cudnn : bool, optional (default=False)
        whether to use cuda or not
    problem : str, optional (default="regression")
        either ``regression`` or ``classification``

    Returns
    -------
    dict :
        a dictionary with ``layers`` as key

    Examples
    --------
    """
    import tensorflow as tf

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

    layers.update({"Dense_out": {"units": output_features,
                   "activation": None if problem=="regression" else "softmax"
                   }})

    return {'layers': layers}


def _check_length(parameter, num_layers):

    if not isinstance(parameter, list):
        parameter = [parameter for _ in range(num_layers)]
    else:
        assert len(parameter)==num_layers

    return parameter