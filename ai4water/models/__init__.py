
from typing import Union, List


from .utils import _make_output_layer


def TabTransformer(
        cat_vocabulary: dict,
        num_numeric_features: int,
        hidden_units=32,
        num_heads: int = 4,
        depth: int = 4,
        dropout: float = 0.1,
        num_dense_lyrs: int = 1,
        prenorm_mlp:bool = True,
        post_norm: bool = True,
        final_mlp_units: Union[int, List[int]] = 16,
        num_outputs: int = 1,
        mode: str = "regression",
        output_activation: str = None,
        seed:int = 313,
        backend:str = "tensorflow"
)->dict:
    """
    Tab Transformer following the work of `Huang et al., 2021 <https://arxiv.org/abs/2012.06678>_`

    Parameters
    ----------
    cat_vocabulary : dict
        a dictionary whose keys are names of categorical features and values
        are lists which consist of unique values of categorical features.
        You can use the function :fun:`ai4water.models.utils.gen_cat_vocab` to create this for your
        own data. The length of dictionary should be equal to number of
        categorical features.
    num_numeric_features : int
        number of numeric features to be used as input.
    hidden_units : int, optional (default=32)
        number of hidden units
    num_heads : int, optional (default=4)
        number of attention heads
    depth : int (default=4)
        number of transformer blocks to be stacked on top of each other
    dropout : int, optional (default=0.1)
        droput rate in transformer
    post_norm : bool (default=True)
    prenorm_mlp : bool (default=True)
    num_dense_lyrs : int (default=2)
        number of dense layers in MLP block
    final_mlp_units : int (default=16)
        number of units/neurons in final MLP layer i.e. the MLP layer
        after Transformer block
    num_outputs : int, optional (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``num_outputs``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the num_outputs are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
    seed : int
    backend : str
        either ``tensorflow`` or ``pytorch``

    Returns
    -------
    dict :
        a dictionary with ``layers`` as key

    Examples
    ----------
    >>> from ai4water import Model
    >>> from ai4water.models import TabTransformer
    >>> from ai4water.utils.utils import TrainTestSplit
    >>> from ai4water.models.utils import gen_cat_vocab
    >>> from ai4water.datasets import mg_photodegradation
    ...
    ... # bring the data as DataFrame
    >>> data, _, _ = mg_photodegradation()
    ... # Define categorical and numerical features and label
    >>> NUMERIC_FEATURES = data.columns.tolist()[0:9]
    >>> CAT_FEATURES = ["Catalyst_type", "Anions"]
    >>> LABEL = "Efficiency (%)"
    ... # create vocabulary of unique values of categorical features
    >>> cat_vocab = gen_cat_vocab(data)
    ... # make sure the data types are correct
    >>> data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype(float)
    >>> data[CAT_FEATURES] = data[CAT_FEATURES].astype(str)
    >>> data[LABEL] = data[LABEL].astype(float)
    >>> # split the data into training and test set
    >>> splitter = TrainTestSplit(seed=313)
    >>> train_data, test_data, _, _ = splitter.split_by_random(data)
    ...
    ... # build the model
    >>> model = Model(model=TabTransformer(
    ...     cat_vocabulary=cat_vocab,num_numeric_features=len(NUMERIC_FEATURES),
    ...     hidden_units=16, final_mlp_units=[84, 42]))
    ... # make a list of input arrays for training data
    >>> train_x = [train_data[NUMERIC_FEATURES].values, train_data['Catalyst_type'].values,
    ...            train_data['Anions'].values]
    >>> test_x = [test_data[NUMERIC_FEATURES].values, test_data['Catalyst_type'].values,
    ...           test_data['Anions'].values]
    ...
    >>> h = model.fit(x=train_x, y= train_data[LABEL].values,
    ...               validation_data=(test_x, test_data[LABEL].values), epochs=1)
    """

    kws = dict(
        cat_vocabulary=cat_vocabulary,
        num_numeric_features=num_numeric_features,
        hidden_units=hidden_units,
        num_heads = num_heads,
        depth = depth,
        dropout = dropout,
        num_dense_lyrs = num_dense_lyrs,
        prenorm_mlp = prenorm_mlp,
        post_norm = post_norm,
        final_mlp_units = final_mlp_units,
        num_outputs = num_outputs,
        mode = mode,
        output_activation = output_activation,
        seed = seed)

    if backend=="tensorflow":
        from .tensorflow import TabTransformer
        return TabTransformer(**kws)
    else:
        raise NotImplementedError


def FTTransformer(
        cat_vocabulary:dict,
        num_numeric_features:int,
        hidden_units = 32,
        num_heads: int = 4,
        depth:int = 4,
        dropout: float = 0.1,
        num_dense_lyrs:int = 2,
        post_norm:bool = True,
        final_mlp_units:int = 16,
        num_outputs: int = 1,
        mode: str = "regression",
        output_activation: str = None,
        seed: int = 313,
        backend:str = "tensorflow"
)->dict:
    """
    FT Transformer following the work of `Gorishniy et al., 2021 <https://arxiv.org/pdf/2106.11959v2.pdf>`_

    Parameters
    ----------
    cat_vocabulary : dict
        a dictionary whose keys are names of categorical features and values
        are lists which consist of unique values of categorical features.
        You can use the function :fun:`ai4water.models.utils.gen_cat_vocab` to create this for your
        own data. The length of dictionary should be equal to number of
        categorical features.
    num_numeric_features : int
        number of numeric features to be used as input.
    hidden_units : int, optional (default=32)
        number of hidden units
    num_heads : int, optional (default=4)
        number of attention heads
    depth : int (default=4)
        number of transformer blocks to be stacked on top of each other
    dropout : int, optional (default=0.1)
        droput rate in transformer
    post_norm : bool (default=True)
    num_dense_lyrs : int (default=2)
        number of dense layers in MLP block
    final_mlp_units : int (default=16)
        number of units/neurons in final MLP layer i.e. the MLP layer
        after Transformer block
    num_outputs : int, optional (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``num_outputs``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the num_outputs are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
    seed : int
    backend : str
        either ``tensorflow`` or ``pytorch``

    Returns
    -------
    dict :
        a dictionary with ``layers`` as key

    Examples
    ----------
    >>> from ai4water import Model
    >>> from ai4water.models import FTTransformer
    >>> from ai4water.datasets import mg_photodegradation
    >>> from ai4water.models.utils import gen_cat_vocab
    >>> from ai4water.utils.utils import TrainTestSplit
    >>> # bring the data as DataFrame
    >>> data, _, _ = mg_photodegradation()
    ... # Define categorical and numerical features and label
    >>> NUMERIC_FEATURES = data.columns.tolist()[0:9]
    >>> CAT_FEATURES = ["Catalyst_type", "Anions"]
    >>> LABEL = "Efficiency (%)"
    ... # create vocabulary of unique values of categorical features
    >>> cat_vocab = gen_cat_vocab(data)
    ... # make sure the data types are correct
    >>> data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype(float)
    >>> data[CAT_FEATURES] = data[CAT_FEATURES].astype(str)
    >>> data[LABEL] = data[LABEL].astype(float)
    ... # split the data into training and test set
    >>> splitter = TrainTestSplit(seed=313)
    >>> train_data, test_data, _, _ = splitter.split_by_random(data)
    ... # build the model
    >>> model = Model(model=FTTransformer(cat_vocab, len(NUMERIC_FEATURES)))
    ... # make a list of input arrays for training data
    >>> train_x = [train_data[NUMERIC_FEATURES].values,
    ...   train_data['Catalyst_type'].values,
    ...   train_data['Anions'].values]
    ...
    >>> test_x = [test_data[NUMERIC_FEATURES].values,
    ...       test_data['Catalyst_type'].values,
    ...       test_data['Anions'].values]
    ... # train the model
    >>> h = model.fit(x=train_x, y= train_data[LABEL].values,
    ...              validation_data=(test_x, test_data[LABEL].values),
    ...              epochs=1)
    """

    kws = dict(
        cat_vocabulary=cat_vocabulary,
        num_numeric_features=num_numeric_features,
        hidden_units=hidden_units,
        num_heads = num_heads,
        depth = depth,
        dropout = dropout,
        num_dense_lyrs = num_dense_lyrs,
        post_norm = post_norm,
        final_mlp_units = final_mlp_units,
        num_outputs = num_outputs,
        mode = mode,
        output_activation = output_activation,
        seed = seed)

    if backend=="tensorflow":
        from .tensorflow import FTTransformer
        return FTTransformer(**kws)
    else:
        raise NotImplementedError


def MLP(
        units: Union[int, list] = 32,
        num_layers:int = 1,
        input_shape: tuple = None,
        output_features:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        mode:str = "regression",
        output_activation:str = None,
        **kwargs
)->dict:
    """helper function to make multi layer perceptron model.
    This model consists of stacking layers of Dense_ layers. The number of
    dense layers are defined by ``num_layers``. Each layer can be optionaly
    followed by a Dropout_ layer.

    Parameters
    ----------
    units : Union[int, list], default=32
        number of units in Dense layer
    num_layers : int, optional, (default, 1)
        number of Dense_ layers to use as hidden layers, excluding output layer.
    input_shape : tuple, optional (default=None)
        shape of input tensor to the model. If specified, it should exclude batch_size
        for example if model takes inputs (num_examples, num_features) then
        we should define the shape as (num_features,). The batch_size dimension
        is always None.
    output_features : int, (default=1)
        number of output features from the network
    activation : Union[str, list], optional (default=None)
        activation function to use.
    dropout : Union[float, list], optional
        dropout to use in Dense layer
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``output_features``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the output_features are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
    **kwargs :
        any additional keyword arguments for Dense_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key which can be fed to ai4water's Model

    Examples
    --------
    >>> from ai4water import Model
    >>> from ai4water.models import MLP
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]
    ... # build a basic MLP
    >>> MLP(32)
    ... # MLP with 3 Dense layers
    >>> MLP(32,  3)
    ... # we can specify input shape as 3d (first dimension is always None)
    >>> MLP(32,  3, (5, 10))
    ... # we can also specify number of units for each layer
    >>> MLP([32, 16, 8], 3, (13, 1))
    ... # we can feed any argument which is accepted by Dense layer
    >>> mlp =  MLP(32, 3, (13, ), use_bias=True, activation="relu")
    ... # we can feed the output of MLP to ai4water's Model
    >>> model = Model(model=mlp, input_features=input_features,
    >>>               output_features=output_features)
    >>> model.fit(data=data)

    .. _Dense:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

    .. _Dropout:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

    """
    check_backend("MLP")

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
        output_features,
        output_activation
    )

    return {'layers': layers}


def LSTM(
        units: Union[int, list] = 32,
        num_layers:int = 1,
        input_shape: tuple = None,
        output_features:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        mode:str = "regression",
        output_activation:str = None,
        **kwargs
):
    """helper function to make LSTM Model

    Parameters
    ----------
    units : Union[int, list], optional (default 32)
        number of units in LSTM layer
    num_layers : int (default=1)
        number of lstm layers to use
    input_shape : tuple, optional (default=None)
        shape of input tensor to the model. If specified, it should exclude batch_size
        for example if model takes inputs (num_examples, lookback, num_features) then
        we should define the shape as (lookback, num_features). The batch_size dimension
        is always None.
    output_features : int, optinoal (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    activation : Union[str, list], optional
        activation function to use in LSTM
    dropout :
        if > 0.0, a dropout layer is added after each LSTM layer
    mode : str, optional
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``output_features``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the output_features are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
    **kwargs :
        any keyword argument for LSTM_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key

    Examples
    --------
    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]
    # a simple LSTM model with 32 neurons/units
    >>> LSTM(32)
    # to build a model with stacking of LSTM layers
    >>> LSTM(32, num_layers=2)
    # we can build ai4water's model and train it
    >>> lstm = LSTM(32)
    >>> model = Model(model=lstm, input_features=input_features,
    >>>               output_features=output_features, ts_args={"lookback": 5})
    >>> model.fit(data=data)

    .. _LSTM:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
    """
    check_backend("LSTM")

    assert num_layers>=1

    if input_shape is None:
        layers = {}
    else:
        layers = {"Input": {"shape": input_shape}}
        assert len(input_shape)>=2

    units = _check_length(units, num_layers)
    dropout = _check_length(dropout, num_layers)
    activation = _check_length(activation, num_layers)

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
        output_features,
        output_activation
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
        output_features:int = 1,
        mode: str = "regression",
        output_activation:str = None,
        **kwargs
)->dict:
    """helper function to make convolution neural network based model.

    Parameters
    ----------
    filters : Union[int, list], optional
        number of filters in convolution layer. If given as list, it should
        be equal to ``num_layers``.
    kernel_size : Union[int, list], optional
        kernel size in (each) convolution layer
    convolution_type : str, optional, (default="1D")
        either ``1D`` or ``2D`` or ``3D``
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
    input_shape : tuple, optional (default=None)
        shape of input tensor to the model. If specified, it should exclude batch_size
        for example if model takes inputs (num_examples, lookback, num_features) then
        we should define the shape as (lookback, num_features). The batch_size dimension
        is always None.
    output_features : int, optional, (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    mode : str, optional
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``output_features``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the output_features are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
    **kwargs :
        any keyword argument for Convolution_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key

    Examples
    --------
    >>> CNN(32, 2, "1D", input_shape=(5, 10))

    >>> CNN(32, 2, "1D", pooling_type="MaxPool", input_shape=(5, 10))


    .. _Convolution:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
    .. _batch_norm:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

    """
    check_backend("CNN")

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
        output_features,
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
        output_features:int = 1,
        mode:str = "regression",
        output_activation:str = None,
)->dict:
    """
    helper function to make CNNLSTM model. It adds one or more 1D convolutional
    layers before one or more LSTM layers.

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model. If specified, it should exclude batch_size
        for example if model takes inputs (num_examples, lookback, num_features) then
        we should define the shape as (lookback, num_features). The batch_size dimension
        is always None.
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
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``output_features``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the output_features are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.

    Returns
    -------
    dict :
        a dictionary with ``layers`` as key

    Examples
    --------
    >>> from ai4water import Model
    >>> from ai4water.models import CNNLSTM
    >>> from ai4water.datasets import busan_beach
    ... # define data and input/output features
    >>> data = busan_beach()
    >>> inputs = data.columns.tolist()[0:-1]
    >>> outputs = [data.columns.tolist()[-1]]
    >>> lookback_steps = 9
    ... # get configuration of CNNLSTM as dictionary which can be given to Model
    >>> model_config = CNNLSTM(input_shape=(lookback_steps, len(inputs)), sub_sequences=3)
    ... # build the model
    >>> model = Model(model=model_config, input_features=inputs,
    ...    output_features=outputs, ts_args={"lookback": lookback_steps})
    ... # train the model
    >>> model.fit(data=data)

    """
    check_backend("CNNLSTM")

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
        output_features,
        output_activation
    )

    return {"layers": layers}


def LSTMAutoEncoder(
        input_shape:tuple,
        encoder_layers:int = 1,
        decoder_layers:int = 1,
        encoder_units: Union[int, list]=32,
        decoder_units: Union[int, list]=32,
        output_features: int = 1,
        prediction_mode: bool = True,
        mode:str = "regression",
        output_activation: str = None,
        **kwargs
)->dict:
    """
    helper function to make LSTM based AutoEncoder model.

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model. This shape should exclude batch_size
        for example if model takes inputs (num_examples, num_features) then
        we should define the shape as (num_features,). The batch_size dimension
        is always None.
    encoder_layers : int, optional (default=1)
        number of encoder LSTM layers
    decoder_layers : int, optional (default=1)
        number of decoder LSTM layers
    encoder_units : Union[int, list], optional, (default=32)
        number of units in (each) encoder LSTM
    decoder_units : Union[int, list], optional, (default=32)
        number of units in (each) decoder LSTM
    prediction_mode : bool, optional (default="prediction")
        either "prediction" or "reconstruction"
    output_features : int, optional
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``output_features``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the output_features are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
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

    check_backend("LSTMAutoEncoder")

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

    layers = _make_output_layer(
        layers,
        mode,
        output_features,
        output_activation
    )
    return {'layers': layers}


def TCN(
        input_shape,
        filters:int = 32,
        kernel_size: int = 2,
        nb_stacks: int = 1,
        dilations = [1, 2, 4, 8, 16, 32],
        output_features:int = 1,
        mode="regression",
        output_activation: str = None,
        **kwargs
)->dict:
    """helper function for building temporal convolution network

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model. This shape should exclude batch_size
        for example if model takes inputs (num_examples, num_features) then
        we should define the shape as (num_features,). The batch_size dimension
        is always None.
    filters : int, optional (default=32)
        number of filters
    kernel_size : int, optional (default=2)
        kernel size
    nb_stacks : int, optional (default=
        number of stacks of tcn layer
    dilations :
        dilation rate
    output_features : int, optional
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``output_features``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the output_features are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
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
    check_backend("TCN")

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
        output_features,
        output_activation
    )
    return {'layers': layers}


def TFT(
        input_shape,
        hidden_units: int = 32,
        num_heads: int = 3,
        dropout:float = 0.1,
        output_features:int = 1,
        use_cudnn:bool = False,
        mode:str="regression",
        output_activation:str = None,
)->dict:
    """helper function for temporal fusion transformer based model

    Parameters
    ----------
    input_shape : tuple
        shape of input tensor to the model. This shape should exclude batch_size
        for example if model takes inputs (num_examples, num_features) then
        we should define the shape as (num_features,). The batch_size dimension
        is always None.
    hidden_units : int, optional (default=32)
        number of hidden units
    num_heads : int, optional (default=1)
        number of attention heads
    dropout : int, optional (default=0.1)
        droput rate
    output_features : int, optional (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    use_cudnn : bool, optional (default=False)
        whether to use cuda or not
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``output_features``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the output_features are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 output_features as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to output_features
        and ``softmax`` as activation.
    Returns
    -------
    dict :
        a dictionary with ``layers`` as key

    Examples
    --------
    >>> from ai4water import Model
    >>> from ai4water.models import TFT
    >>> from ai4water.datasets import busan_beach
    >>> model = Model(model=TFT(input_shape=(14, 13)),
    ...                   ts_args={"lookback": 14})
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

    layers = _make_output_layer(
        layers,
        mode,
        output_features,
        output_activation
    )

    return {'layers': layers}


def _check_length(parameter, num_layers):

    if not isinstance(parameter, list):
        parameter = [parameter for _ in range(num_layers)]
    else:
        assert len(parameter)==num_layers

    return parameter


def check_backend(model:str, backend:str="tf")->None:
    if backend=="tf":
        try:
            import tensorflow as tf
        except Exception as e:
            raise Exception(f"""You must have install tensorflow to use {model} model. 
            Importing tensorflow raised following error \n{e}""")
    elif backend == "torch":
        try:
            import torch
        except Exception as e:
            raise Exception(f"""You must have install PyTorch to use {model} model. 
            Importing torch raised following error \n{e}""")
    return
