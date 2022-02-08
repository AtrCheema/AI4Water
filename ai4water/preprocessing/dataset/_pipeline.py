

__all__ = ['DataSetPipeline']

import os

import numpy as np

from ._main import _DataSet


class DataSetPipeline(_DataSet):
    """A collection of DataSets concatenated one after the other. A DataSetPipeLine
    of four DataSets will be as follows:

        ||     DataSet1   ||
        ||     DataSet2   ||
        ||     DataSet3   ||
        ||     DataSet4   ||

    The only condition for different datasets is that they have the same output dimension.

    """
    def __init__(
            self,
            *datasets,
            verbosity=1
    ) -> None:
        """

        """
        self.verbosity = verbosity

        self._datasets = []
        for ds in datasets:
            assert isinstance(ds, _DataSet), f"""
                            {ds} is not a valid dataset"""
            self._datasets.append(ds)

        self.examples = {}

        _DataSet.__init__(self, config={}, path=os.getcwd())

    @property
    def num_datasets(self) -> int:
        return len(self._datasets)

    @property
    def teacher_forcing(self):
        return all([ds.teacher_forcing for ds in self._datasets])

    @property
    def mode(self):
        return all([ds.mode for ds in self._datasets])

    @property
    def is_binary(self):
        return all([ds.is_binary for ds in self._datasets])

    @property
    def input_features(self):
        _input_features = [ds.input_features for ds in self._datasets]
        return _input_features

    @property
    def output_features(self):
        _output_features = [ds.output_features for ds in self._datasets]
        return _output_features

    def training_data(self, key="train", **kwargs):
        if self.teacher_forcing:
            x, prev_y, y = self._get_x_yy('training_data')
            return x, prev_y, y
        else:
            x, y = self._get_xy('training_data')
            return self.return_xy(np.row_stack(x), np.row_stack(y), "Training")

    def validation_data(self, key="val", **kwargs):
        if self.teacher_forcing:
            x, prev_y, y = self._get_x_yy('validation_data')
            return x, prev_y, y
        else:
            x, y = self._get_xy('validation_data')
            return self.return_xy(x, y, "Validation")

    def test_data(self, key="test", **kwargs):
        if self.teacher_forcing:

            x, prev_y, y = self._get_x_yy('test_data')
            return x, prev_y, y
        else:

            x, y = self._get_xy('test_data')
            return self.return_xy(x, y, "Test")

    def _get_x_yy(self, method):
        x, prev_y, y = [], [], []
        exs = {}

        for idx, ds in enumerate(self._datasets):
            _x, _prev_y, _y = getattr(ds, method)()
            x.append(_x)
            prev_y.append(_prev_y)
            y.append(_y)
            exs[idx] = {'x': len(x), 'y': len(y)}

        self.examples[method] = exs

        if not all([i.size for i in x]):
            x = conform_shape(x)
            prev_y = conform_shape(prev_y)
            y = conform_shape(y)

        return np.row_stack(x), np.row_stack(prev_y), np.row_stack(y)

    def _get_xy(self, method):

        x, y = [], []
        exs = {}
        for idx, ds in enumerate(self._datasets):
            _x, _y = getattr(ds, method)()
            x.append(_x)
            y.append(_y)
            exs[idx] = {'x': len(x), 'y': len(y)}

        self.examples[method] = exs

        if not all([i.size for i in x]):
            x = conform_shape(x)
            y = conform_shape(y)

        return np.row_stack(x), np.row_stack(y)


def conform_shape(alist:list):
    desired_shape = list([i.shape for i in alist if i.size != 0][0])
    desired_shape[0] = 0
    return [np.zeros(desired_shape) if arr.size == 0 else arr for arr in alist]
