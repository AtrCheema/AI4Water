

__all__ = ['DataSetUnion']

from typing import Union

from ai4water.backend import np, os

from ._main import _DataSet


class DataSetUnion(_DataSet):
    """A Union of datasets concatenated in parallel. A DataSetUnion of four DataSets will be as follows:

        ===========  ===========  ===========  ===========
        DataSet1     DataSet2     DataSet3     DataSet4
        ===========  ===========  ===========  ===========

    """
    def __init__(
            self,
            *datasets,
            stack_y : bool = False,
            verbosity : int = 1,
            **named_datasets
    ) -> None:
        """
        DataSets must be passed either as positional arguments or as keyword arguments
        but not both.

        Parameters
        ----------
        datasets : 
            DataSets to be concatenated in parallel.
        stack_y : bool
            whether to stack y/outputs of individual datasets as one array or not
        verbosity : bool
            controls amount of information being printed
        named_datasets : 
            DataSets to be concatenated in parallel.


        Examples
        ---------
        >>> import pandas as pd
        >>> from ai4water.preprocessing import DataSet, DataSetUnion
        >>> df1 = pd.DataFrame(np.random.random((100, 10)),
        ...              columns=[f"Feat_{i}" for i in range(10)])
        >>> df2 = pd.DataFrame(np.random.random((200, 10)),
        ...              columns=[f"Feat_{i}" for i in range(10)])
        >>> ds1 = DataSet(df1)
        >>> ds2 = DataSet(df2)
        >>> ds = DataSetUnion(ds1, ds2)
        >>> train_x, train_y = ds.training_data()
        >>> val_x, val_y = ds.validation_data()
        >>> test_x, test_y = ds.test_data()

        Note
        ----
        DataSets must be provided either as positional arguments or as keyword arguments
        using named_datasets and not both.
        """

        self.stack_y = stack_y
        self.verbosity = verbosity

        self.as_list = False
        self.as_dict = False

        self._datasets = {}

        if datasets:
            assert not named_datasets, f"""provide either datasets or named_datasets, not both"""
            self.as_list = True
            for idx, ds in enumerate(datasets):
                assert isinstance(ds, _DataSet), f"""
                                {ds} is not a valid dataset. {ds.__class__.__name__}"""
                self._datasets[idx] = ds

        if named_datasets:
            assert not datasets, f"""provide either datasets or named_datasets, not both"""
            self.as_dict = True
            for name, ds in named_datasets.items():
                assert isinstance(ds, _DataSet), f"""
                                {ds} is not a valid dataset. {ds.__class__.__name__}"""
                self._datasets[name] = ds

        self.examples = {}

        _DataSet.__init__(self, config={}, path=os.getcwd())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = self._datasets[self.index]
        except KeyError:
            self.index = 0
            raise StopIteration

        self.index += 1
        return item

    def __getitem__(self, item: int):
        return self._datasets[item]

    @property
    def mode(self):
        return list(set([ds.mode for ds in self._datasets.values()]))[0]

    @property
    def num_datasets(self) -> int:
        return len(self._datasets)

    @property
    def teacher_forcing(self):
        return all([ds.teacher_forcing for ds in self._datasets.values()])

    @property
    def is_binary(self):
        return all([ds.is_binary for ds in self._datasets.values()])

    @property
    def is_multilabel(self):
        return all([ds.is_multilabel for ds in self._datasets.values()])

    @property
    def is_multiclass(self):
        return all([ds.is_multiclass for ds in self._datasets.values()])

    @property
    def input_features(self):
        _input_features = {k:v.input_features for k,v in self._datasets.items()}
        if self.as_list:
            return list(_input_features.values())
        return _input_features

    @property
    def output_features(self):
        _output_features = {k:v.output_features for k,v in self._datasets.items()}
        if self.as_list:
            return list(_output_features.values())
        return _output_features

    @property
    def indexes(self):  # todo, needs testing
        # multiple DataSets will have separate indexes so keys in idx will get updated
        idx = {}
        for ds in self._datasets.values():
            for k, v in ds.indexes.items():
                #print(ds_n, k, type(v))
                idx[k] = v
        return idx

    @property
    def ts_args(self)->dict:
        _ts_args = {k:ds.ts_args for k,ds in self._datasets.items()}
        return _ts_args

    def training_data(self, key="train", **kwargs)->Union[list, dict]:
        if self.teacher_forcing:
            x, prev_y, y = self._get_x_yy('training_data', key, **kwargs)
            return self.return_x_yy(x, prev_y, y, 'Training')

        else:
            x, y = self._get_xy('training_data', key, **kwargs)
            return self.return_xy(x, y, 'Training')

    def validation_data(self, key="val", **kwargs):
        if self.teacher_forcing:
            x, prev_y, y = self._get_x_yy('validation_data', key, **kwargs)
            return self.return_x_yy(x, prev_y, y, 'Validation')

        else:
            x, y = self._get_xy('validation_data', key, **kwargs)
            return self.return_xy(x, y, 'Validation')

    def test_data(self, key="test", **kwargs):
        if self.teacher_forcing:
            x, prev_y, y = self._get_x_yy('test_data', key, **kwargs)
            return self.return_x_yy(x, prev_y, y, 'Test')

        else:
            x, y = self._get_xy('test_data', key, **kwargs)
            return self.return_xy(x, y, 'Test')

    def _get_x_yy(self, method, key, **kwargs):
        x, prev_y, y = {}, {}, {}
        exs = {}

        for ds_name, ds in self._datasets.items():
            x[ds_name], prev_y[ds_name], y[ds_name] = getattr(ds, method)(key, **kwargs)

            exs[ds_name] = {'x': len(x[ds_name]), 'y': len(y[ds_name])}

        self.examples[method] = exs

        if self.as_list:
            return list(x.values()), list(prev_y.values()), list(y.values())
        return x, prev_y, y

    def _get_xy(self, method, key, **kwargs):
        x, y = {}, {}
        exs = {}

        for ds_name, ds in self._datasets.items():
            _x, _y = getattr(ds, method)(key, **kwargs)
            x[ds_name] = _x

            # if one ds does not return y, better to ignroe this y, because when we
            # stack ys together, they must be same
            if _y.size > 0:
                y[ds_name] = _y
                exs[ds_name] = {'x': len(x[ds_name]), 'y': len(y[ds_name])}
            else:
                exs[ds_name] = {'x': len(x[ds_name]), 'y': 0}

        self.examples[method] = exs

        # it is possible that y has only 1 member because all other DataSets don'y have
        # any y and may have already been purged.
        if self.stack_y and len(y)>1:
            assert np.allclose(*list(y.values()))
            y = {'y': list(y.values())[0]}

        if self.as_list:
            return list(x.values()), list(y.values())

        return x, y
