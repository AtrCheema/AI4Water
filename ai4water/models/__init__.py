
from typing import Union, List


from .utils import _make_output_layer


def TabTransformer(
        num_numeric_features: int,
        cat_vocabulary: dict,
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
        backend:str = "tf"
)->dict:
    """
    Tab Transformer following the work of `Huang et al., 2021 <https://arxiv.org/abs/2012.06678>_`

    Parameters
    ----------
    num_numeric_features : int
        number of numeric features to be used as input.
    cat_vocabulary : dict
        a dictionary whose keys are names of categorical features and values
        are lists which consist of unique values of categorical features.
        You can use the function :fun:`ai4water.models.utils.gen_cat_vocab` to create this for your
        own data. The length of dictionary should be equal to number of
        categorical features.
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
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    seed : int
        seed for reproducibility
    backend : str
        either ``tf`` or ``pytorch``

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
    ...     num_numeric_features=len(NUMERIC_FEATURES), cat_vocabulary=cat_vocab,
    ...     hidden_units=16, final_mlp_units=[84, 42]))
    ... # make a list of input arrays for training data
    >>> train_x = [train_data[NUMERIC_FEATURES].values, train_data[CAT_FEATURES]]
    >>> test_x = [test_data[NUMERIC_FEATURES].values, test_data[CAT_FEATURES].values]
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

    if backend=="tf":
        from ._tensorflow import TabTransformer
        return TabTransformer(**kws)
    else:
        raise NotImplementedError


def FTTransformer(
        num_numeric_features:int,
        cat_vocabulary:dict = None,
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
        backend:str = "tf"
)->dict:
    """
    FT Transformer following the work of `Gorishniy et al., 2021 <https://arxiv.org/pdf/2106.11959v2.pdf>`_

    Parameters
    ----------
    num_numeric_features : int
        number of numeric features to be used as input.
    cat_vocabulary : dict
        a dictionary whose keys are names of categorical features and values
        are lists which consist of unique values of categorical features.
        You can use the function :fun:`ai4water.models.utils.gen_cat_vocab` to create this for your
        own data. The length of dictionary should be equal to number of
        categorical features. If it is None, then it is supposed that
        no categoical variables are available and the model will expect only
        numerical input features.
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
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    seed : int
    backend : str
        either ``tf`` or ``pytorch``

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
    >>> model = Model(model=FTTransformer(len(NUMERIC_FEATURES), cat_vocab))
    ... # make a list of input arrays for training data
    >>> train_x = [train_data[NUMERIC_FEATURES].values,
    ...   train_data[CAT_FEATURES].values]
    ...
    >>> test_x = [test_data[NUMERIC_FEATURES].values,
    ...       test_data[CAT_FEATURES].values]
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

    if backend=="tf":
        from ._tensorflow import FTTransformer
        return FTTransformer(**kws)
    else:
        raise NotImplementedError


def MLP(
        units: Union[int, list] = 32,
        num_layers:int = 1,
        input_shape: tuple = None,
        num_outputs:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        mode:str = "regression",
        output_activation:str = None,
        backend:str = "tf",
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
        number of Dense_ or Linear_ layers to use as hidden layers, excluding output layer.
    input_shape : tuple, optional (default=None)
        shape of input tensor to the model. If specified, it should exclude batch_size
        for example if model takes inputs (num_examples, num_features) then
        we should define the shape as (num_features,). The batch_size dimension
        is always None.
    num_outputs : int, (default=1)
        number of output features from the network
    activation : Union[str, list], optional (default=None)
        activation function to use.
    dropout : Union[float, list], optional
        dropout to use in Dense layer
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``num_outputs``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the num_outputs are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str (default='tf')
        either ``tf`` or ``pytorch``
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

    similary for pytorch as backend we can build the model as below

    >>> model = Model(model=MLP(32, 2, (13,), backend="pytorch"),
    ...           backend="pytorch",
    ...           input_features = input_features,
    ...           output_features = output_features)
    >>> model.fit(data=data)

    .. _Dense:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

    .. _Linear:
        https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    .. _Dropout:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

    """
    check_backend("MLP", backend=backend)

    kws = dict(
        units = units,
        num_layers = num_layers,
        input_shape = input_shape,
        num_outputs = num_outputs,
        activation = activation,
        dropout = dropout,
        mode = mode,
        output_activation = output_activation,
        **kwargs
    )

    if backend == "tf":
        from ._tensorflow import MLP
        return MLP(**kws)
    else:
        from ._torch import MLP
        return MLP(**kws)


def LSTM(
        units: Union[int, list] = 32,
        num_layers:int = 1,
        input_shape: tuple = None,
        num_outputs:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        mode:str = "regression",
        output_activation:str = None,
        backend:str = "tf",
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
    num_outputs : int, optinoal (default=1)
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
        then the activation of output layer is decided based upon ``num_outputs``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the num_outputs are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str
        the type of backend to use. Allowed vlaues are ``tf`` or ``pytorch``.
    **kwargs :
        any keyword argument for LSTM layer of `tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>`_
        or `pytorch <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`_ layer

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

    Similary for pytorch as backend the interface will be same
    except that we need to explicitly tell the backend as below

    >>> model = Model(model=LSTM(32, 2, (5, 13), backend="pytorch"),
    ...           backend="pytorch",
    ...               input_features = input_features,
    ...               output_features = output_features,
    ...           ts_args={'lookback':5})
    >>> model.fit(data=data)
    """
    check_backend("LSTM", backend=backend)
    kws = dict(
        units = units,
        num_layers = num_layers,
        input_shape = input_shape,
        num_outputs = num_outputs,
        activation = activation,
        dropout = dropout,
        mode = mode,
        output_activation = output_activation,
        **kwargs
    )

    if backend == "tf":
        from ._tensorflow import LSTM
        return LSTM(**kws)
    else:
        from ._torch import LSTM
        return LSTM(**kws)


def AttentionLSTM(
        units: Union[int, list] = 32,
        num_layers:int = 1,
        input_shape: tuple = None,
        num_outputs:int = 1,
        activation: Union[str, list] = None,
        dropout: Union[float, list] = None,
        atten_units:int = 128,
        atten_activation:str = "tanh",
        mode:str = "regression",
        output_activation:str = None,
        backend:str = "tf",
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
    num_outputs : int, optinoal (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    activation : Union[str, list], optional
        activation function to use in LSTM
    dropout :
        if > 0.0, a dropout layer is added after each LSTM layer
    atten_units : int
        number of units in SelfAttention layer
    atten_activation : str
        activation function in SelfAttention layer
    mode : str, optional
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``num_outputs``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the num_outputs are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str
        the type of backend to use. Allowed vlaues are ``tf`` or ``pytorch``.
    **kwargs :
        any keyword argument for LSTM layer of `tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>`_
        or `pytorch <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key

    Examples
    --------
    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> from ai4water.models import AttentionLSTM
    >>> data = busan_beach()
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]
    # a simple Attention LSTM model with 32 neurons/units
    >>> AttentionLSTM(32)
    # to build a model with stacking of LSTM layers
    >>> AttentionLSTM(32, num_layers=2)
    # we can build ai4water's model and train it
    >>> lstm = AttentionLSTM(32)
    >>> model = Model(model=lstm, input_features=input_features,
    >>>               output_features=output_features, ts_args={"lookback": 5})
    >>> model.fit(data=data)

    """
    check_backend("AttentionLSTM", backend=backend)
    kws = dict(
        units = units,
        num_layers = num_layers,
        input_shape = input_shape,
        num_outputs = num_outputs,
        activation = activation,
        dropout = dropout,
        atten_units = atten_units,
        atten_activation = atten_activation,
        mode = mode,
        output_activation = output_activation,
        **kwargs
    )

    if backend == "tf":
        from ._tensorflow import AttentionLSTM
        return AttentionLSTM(**kws)
    else:
        raise NotImplementedError


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
        backend:str = "tf",
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
    num_outputs : int, optional, (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    mode : str, optional
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``num_outputs``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the num_outputs are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str
        either "tf" or "pytorch"
    **kwargs :
        any keyword argument for Convolution_ layer

    Returns
    -------
    dict :
        a dictionary with 'layers' as key

    Examples
    --------
    >>> from ai4water import Model
    >>> from ai4water.models import CNN
    >>> from ai4water.datasets import busan_beach
    ...
    >>> data = busan_beach()
    >>> input_features = data.columns.tolist()[0:-1]
    >>> output_features = data.columns.tolist()[-1:]
    >>> model_config = CNN(32, 2, "1D", input_shape=(5, 10))
    >>> model = Model(model=model_config, ts_args={"lookback": 5}, backend="pytorch",
    ...         input_features=input_features, output_features=output_features)
    ...
    >>> model.fit(data=data)
    >>> model_config = CNN(32, 2, "1D", pooling_type="MaxPool", input_shape=(5, 10))
    >>> model = Model(model=model_config, ts_args={"lookback": 5}, backend="pytorch",
    ...         input_features=input_features, output_features=output_features)
    ...
    >>> model.fit(data=data)

    .. _Convolution:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
    .. _batch_norm:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

    """
    check_backend("CNN", backend=backend)

    kws = dict(
        filters=filters,
        kernel_size=kernel_size,
        convolution_type=convolution_type,
        num_layers=num_layers,
        padding=padding,
        strides=strides,
        pooling_type=pooling_type,
        pool_size=pool_size,
        batch_normalization=batch_normalization,
        activation=activation,
        dropout=dropout,
        input_shape=input_shape,
        num_outputs=num_outputs,
        mode=mode,
        output_activation=output_activation,
        **kwargs
    )

    if backend == "tf":
        from ._tensorflow import CNN
        return CNN(**kws)
    else:

        for arg in ['strides', 'kernel_size']:
            kws.pop(arg)

        from ._torch import CNN
        return CNN(**kws)


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
        backend:str = "tf",
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
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str

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
    ...    num_outputs=outputs, ts_args={"lookback": lookback_steps})
    ... # train the model
    >>> model.fit(data=data)

    """
    check_backend("CNNLSTM", backend=backend)

    kws = dict(
        input_shape=input_shape,
        sub_sequences=sub_sequences,
        cnn_layers=cnn_layers,
        lstm_layers=lstm_layers,
        filters=filters,
        kernel_size=kernel_size,
        max_pool=max_pool,
        units=units,
        num_outputs=num_outputs,
        mode=mode,
        output_activation=output_activation
    )
    if backend == "tf":
        from ._tensorflow import CNNLSTM
        return CNNLSTM(**kws)
    else:
        raise NotImplementedError


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
        backend:str = "tf",
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
    num_outputs : int, optional
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
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str
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

    check_backend("LSTMAutoEncoder", backend=backend)

    kws = dict(
        input_shape=input_shape,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        num_outputs=num_outputs,
        prediction_mode=prediction_mode,
        mode=mode,
        output_activation=output_activation,
        **kwargs
    )
    if backend == "tf":
        from ._tensorflow import LSTMAutoEncoder
        return LSTMAutoEncoder(**kws)
    else:
        raise NotImplementedError


def TCN(
        input_shape,
        filters:int = 32,
        kernel_size: int = 2,
        nb_stacks: int = 1,
        dilations = [1, 2, 4, 8, 16, 32],
        num_outputs:int = 1,
        mode="regression",
        output_activation: str = None,
        backend:str = "tf",
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
    num_outputs : int, optional
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
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str
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
    check_backend("TCN", backend=backend)

    kws = dict(
        input_shape=input_shape,
        filters=filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        num_outputs=num_outputs,
        mode=mode,
        output_activation=output_activation,
        **kwargs
    )
    if backend == "tf":
        from ._tensorflow import TCN
        return TCN(**kws)
    else:
        raise NotImplementedError


def TFT(
        input_shape,
        hidden_units: int = 32,
        num_heads: int = 3,
        dropout:float = 0.1,
        num_outputs:int = 1,
        use_cudnn:bool = False,
        mode:str="regression",
        output_activation:str = None,
        backend:str = "tf",
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
    num_outputs : int, optional (default=1)
        number of output features. If ``mode`` is ``classification``, this refers
        to number of classes.
    use_cudnn : bool, optional (default=False)
        whether to use cuda or not
    mode : str, optional (default="regression")
        either ``regression`` or ``classification``
    output_activation : str, optional (default=None)
        activation of the output layer. If not given and the mode is clsasification
        then the activation of output layer is decided based upon ``num_outputs``
        argument. In such a case, for binary classification, sigmoid with 1 output
        neuron is preferred. Therefore, even if the num_outputs are 2,
        the last layer will have 1 neuron and activation function is ``sigmoid``.
        Although the user can set ``softmax`` for 2 num_outputs as well
        (binary classification) but this seems superfluous and is slightly
        more expensive.
        For multiclass, the last layer will have neurons equal to num_outputs
        and ``softmax`` as activation.
    backend : str
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

    kws = dict(
        input_shape=input_shape,
        hidden_units=hidden_units,
        num_heads=num_heads,
        dropout=dropout,
        use_cudnn=use_cudnn,
        num_outputs=num_outputs,
        mode=mode,
        output_activation=output_activation
    )
    if backend == "tf":
        from ._tensorflow import TFT
        return TFT(**kws)
    else:
        raise NotImplementedError


def check_backend(model:str, backend:str="tf")->None:

    if backend=="tf":
        try:
            import tensorflow as tf
        except Exception as e:
            raise Exception(f"""
            You must have installed tensorflow to use {model} model. 
            Importing tensorflow raised following error \n{e}""")

    elif backend == "pytorch":
        try:
            import torch
        except Exception as e:
            raise Exception(f"""
            You must have installed PyTorch to use {model} model. 
            Importing pytorch raised following error \n{e}""")

    return
