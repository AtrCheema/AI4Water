
from ai4water.backend import pd


def gen_cat_vocab(
        data,
        cat_columns:list = None,
)->dict:

    if data.ndim != 2:
        raise TypeError('Expected a 2-dimensional dataframe or array')

    assert isinstance(data, pd.DataFrame)

    vocab = {}
    if cat_columns:
        for feature in cat_columns:
            vocab[feature] = sorted(list(data[feature].unique()))
    else:
        for col in data.columns:
            if data.loc[:, col].dtype == 'O':
                vocab[col] = sorted(list(data.loc[:, col].unique()))

    return vocab


def _make_output_layer(
        layers:dict,
        mode:str="regression",
        num_outputs:int=1,
        output_activation:str=None,
)->dict:
    if output_activation is None and mode == "classification":
        # for binary it is better to use sigmoid
        if num_outputs > 2:
            output_activation = "softmax"
        else:
            output_activation = "sigmoid"
            num_outputs = 1

    layers.update(
        {"Dense_out": {"units": num_outputs,
                       "activation": output_activation}})
    return layers
