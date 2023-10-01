
import json

from ai4water.backend import np, pd, sklearn


def consider_intervals(data, intervals):
    _source = data
    if intervals is not None:
        if isinstance(data, pd.DataFrame):
            try:  # if indices in intervals are of same type as that of index
                # -1 so that .loc and .iloc give same results, however this is not possible
                # with DatetimeIndex
                if isinstance(data.index, pd.DatetimeIndex):
                    _source = pd.concat([data.loc[st:en] for st, en in intervals])
                else:
                    _source = pd.concat([data.loc[st:en - 1] for st, en in intervals])
            except TypeError:  # assuming indices in intervals are integers
                _source = pd.concat([data.iloc[st:en] for st, en in intervals])

    return _source


def load_data_from_hdf5(data_type, data):
    import h5py

    f = h5py.File(data, mode='r')

    g = f[data_type]
    weight_names = list(g.keys())
    weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

    f.close()

    return weight_values


def check_for_classification(label: np.ndarray, to_categorical):
    OneHotEncoder = sklearn.preprocessing.OneHotEncoder

    assert isinstance(label, np.ndarray), f"""
                            classification problem for label of type {label.__class__.__name__} not implemented yet"""

    # for clsasification, it should be 2d and integer
    labels = label.reshape(-1, label.shape[1])
    if to_categorical:
        assert labels.shape[1] == 1
        labels = OneHotEncoder(sparse=False).fit_transform(labels)
    # else:   # mutlti_label/binary problem
    #     # todo, is only binary_crossentropy is binary/multi_label problem?
    #     pass #assert self.loss_name() in ['binary_crossentropy']
    return labels.astype(np.int32)


def decode(json_string):
    return json.loads(json_string, object_hook=_decode_helper)


def _decode_helper(obj):
    """A decoding helper that is TF-object aware."""
    if isinstance(obj, dict) and 'class_name' in obj:

        if obj['class_name'] == '__tuple__':
            return tuple(_decode_helper(i) for i in obj['items'])
        elif obj['class_name'] == '__ellipsis__':
            return Ellipsis
    return obj