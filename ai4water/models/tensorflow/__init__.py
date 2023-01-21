
from ..utils import _make_output_layer

try:
    import tensorflow as tf
except (ModuleNotFoundError, ImportError):
    tf = None

if tf is not None:
    from .private_layers import Conditionalize
    from .private_layers import MCLSTM
    from .private_layers import EALSTM
    from .private_layers import NumericalEmbeddings
    from .private_layers import CatEmbeddings
    from .private_layers import TransformerBlocks
    from .nbeats_keras import NBeats
    from .tft_layer import TemporalFusionTransformer


def TabTransformer(

):
    """
    Tab Transformer following the work of `Huang et al., 2021 <https://arxiv.org/abs/2012.06678>_`
    """
    return


def FTTransformer(
        cat_vocabulary:dict,
        num_numeric_features:int,
        hidden_units = 32,
        num_heads: int = 4,
        depth:int = 4,
        mlp_units: int = 32,
        dropout: float = 0.1,
        post_norm:bool = True,
        final_mpl_units:int = 16,
        num_outputs: int = 1,
        mode: str = "regression",
        output_activation: str = None,
)->dict:
    """
    FT Transformer following the work of `Doroshny et al., 2021 <https://arxiv.org/pdf/2106.11959v2.pdf>_`

    Parameters
    ----------
    cat_vocabulary : dict
        a dictionary whose keys are names of categorical features and values
        are lists which consist of unique values of categorical features.
        You can use the function `gen_cat_vocab` to create this for your
        own data. The length of dictionary should be equal to number of
        categorical features.
    num_numeric_features : int
        number of numeric features to be used as input.
    hidden_units : int, optional (default=32)
        number of hidden units
    num_heads : int, optional (default=1)
        number of attention heads
    depth : int
        number of transformer blocks to be stacked on top of each other
    dropout : int, optional (default=0.1)
        droput rate in transformer
    post_norm : bool (default=True)
    mlp_units : int
        number of units/neurons in Dense/MLP layers which are inside Transformer
    final_mpl_units : int (default=16)
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

    >>> test_x = [test_data[NUMERIC_FEATURES].values,
    ...       test_data['Catalyst_type'].values,
    ...       test_data['Anions'].values]

    >>> h = model.fit(x=train_x, y= train_data[LABEL].values,
    ...              validation_data=(test_x, test_data[LABEL].values),
    ...              epochs=1)
    """

    layers = _make_input_lyrs(num_numeric_features, list(cat_vocabulary.keys()))

    body = {
        "CatEmbeddings": {"config": {'vocabulary': cat_vocabulary,
                                     'embed_dim': hidden_units},
                          "inputs": list(cat_vocabulary.keys())},

        "NumericalEmbeddings": {"config": {'num_features': num_numeric_features,
                                           'emb_dim': hidden_units},
                                "inputs": "Input_num"},

        "Concatenate": {"config": {"axis": 1},
                        "inputs": ["NumericalEmbeddings", "CatEmbeddings"]},

        "TransformerBlocks": {"config": dict(embed_dim=hidden_units,
                                             num_heads=num_heads,
                                             num_blocks=depth,
                                             mlp_units=mlp_units,
                                             post_norm=post_norm,
                                             dropout=dropout),
                              "inputs": "Concatenate",
                              "outputs": ['transformer_output', 'imp']},

        "lambda": {"config": tf.keras.layers.Lambda(lambda x: x[:, 0, :]),
                   "inputs": "transformer_output"},

        "LayerNormalization_0": {"config": {}, "inputs": "lambda"},
        "Dense_0": {"config": {"units": final_mpl_units}, "inputs": "LayerNormalization_0"}
    }

    layers.update(body)

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )

    return {"layers": layers}


def _make_input_lyrs(
        num_numeric_features:int,
        cat_features:list,
        numeric_dtype:tf.DType=tf.float32,
        cat_dtype:tf.DType = tf.string
)->dict:

    lyrs = {"Input_num": {"shape": (num_numeric_features,), "dtype": numeric_dtype}}

    for feature_name in cat_features:
        lyr = {f"Input_{feature_name}": {"shape": (), "dtype": cat_dtype, "name": feature_name}}
        lyrs.update(lyr)

    return lyrs