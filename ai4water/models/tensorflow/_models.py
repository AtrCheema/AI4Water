
from typing import Union, List

from ai4water.backend import tf
from ..utils import _make_output_layer


def TabTransformer(
        cat_vocabulary: dict,
        num_numeric_features: int,
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
)->dict:
    """
    Tab Transformer following the work of `Huang et al., 2021 <https://arxiv.org/abs/2012.06678>_`
    """

    layers = _make_input_lyrs(num_numeric_features, list(cat_vocabulary.keys()))

    body = {
        "CatEmbeddings": {"config": {'vocabulary': cat_vocabulary,
                                     'embed_dim': hidden_units},
                          "inputs": list(cat_vocabulary.keys())},

        "LayerNormalization_0": {"config": {}, "inputs": "Input_num"},

        "TransformerBlocks": {"config": dict(embed_dim=hidden_units,
                                             num_heads=num_heads,
                                             num_blocks=depth,
                                             num_dense_lyrs=num_dense_lyrs,
                                             post_norm=post_norm,
                                             prenorm_mlp=prenorm_mlp,
                                             dropout=dropout),
                              "inputs": "CatEmbeddings",
                              "outputs": ['transformer_output', 'imp']},

        "Flatten": {"config": {},
                    "inputs": "transformer_output"},

        "Concatenate": {"config": {},
                        "inputs": ["LayerNormalization_0", "Flatten"]},
    }

    body.update(_make_mlp(final_mlp_units, "Concatenate", seed=seed))

    layers.update(body)

    layers = _make_output_layer(
        layers,
        mode,
        num_outputs,
        output_activation
    )
    return {"layers": layers}


def _make_mlp(
        units,
        inputs:str,
        norm_lyr:str = "BatchNormalization",
        dropout=0.1,
        activation:str = "selu",
        seed=313,
)->dict:

    lyrs = {norm_lyr: {"config": {}, "inputs": inputs}}

    if isinstance(units, int):
        units = [units]

    for idx, unit in enumerate(units):
        lyrs.update({
            f"Dense_{idx}": {"config": {"units": unit, "activation": activation}},
            f"Dropout_{idx}": {"config": {"rate": dropout, 'seed': seed}}
        })

    return lyrs


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
        seed:int = 313,
)->dict:

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
                                             num_dense_lyrs=num_dense_lyrs,
                                             post_norm=post_norm,
                                             dropout=dropout,
                                             seed=seed
                                             ),
                              "inputs": "Concatenate",
                              "outputs": ['transformer_output', 'imp']},

        "lambda": {"config": tf.keras.layers.Lambda(lambda x: x[:, 0, :]),
                   "inputs": "transformer_output"},

        "LayerNormalization_0": {"config": {}, "inputs": "lambda"},
        "Dense_0": {"config": {"units": final_mlp_units}, "inputs": "LayerNormalization_0"}
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