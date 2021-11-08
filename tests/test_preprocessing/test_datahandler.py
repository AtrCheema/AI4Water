import os
import unittest
import random
import sys
import site   # so that ai4water directory is in path
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ai4water import Model
from ai4water.preprocessing import DataHandler, SiteDistributedDataHandler
from ai4water.preprocessing.datahandler import MultiLocDataHandler
from ai4water.datasets import arg_beach

os.environ['PYTHONHASHSEED'] = '313'
random.seed(313)
np.random.seed(313)


# todo, check last dimension of x,y
# todo test with 3d y

def _check_xy_equal_len(x, prev_y, y, lookback, num_ins, num_outs, num_examples, data_type='training'):

    feat_dim = 1
    if lookback > 1:
        assert x.shape[1] == lookback
        feat_dim = 2

    assert x.shape[
               feat_dim] == num_ins, f"for {data_type} x's shape is {x.shape} while num_ins of dataloader are {num_ins}"

    if y is not None:
        assert y.shape[1] == num_outs, f"for {data_type} y's shape is {y.shape} while num_outs of dataloader are {num_outs}"
    else:
        assert num_outs == 0
        y = x  # just for next statement to run

    if prev_y is None:
        prev_y = x  # just for next statement to run

    assert x.shape[0] == y.shape[0] == prev_y.shape[
        0], f"for {data_type} xshape: {x.shape}, yshape: {y.shape}, prevyshape: {prev_y.shape}"

    if num_examples:
        assert x.shape[
                   0] == num_examples, f'for {data_type} x contains {x.shape[0]} samples while expected samples are {num_examples}'
    return


def assert_xy_equal_len(x, prev_y, y, data_loader, num_examples=None, data_type='training'):

    if isinstance(x, np.ndarray):
        _check_xy_equal_len(x, prev_y, y, data_loader.lookback, data_loader.num_ins, data_loader.num_outs, num_examples,
                            data_type=data_type)

    elif isinstance(x, list):

        while len(y)<len(x):
            y.append(None)

        for idx, i in enumerate(x):
            _check_xy_equal_len(i, prev_y[idx], y[idx], data_loader.lookback[idx], data_loader.num_ins[idx],
                                data_loader.num_outs[idx], num_examples, data_type=data_type
                                )

    elif isinstance(x, dict):
        for key, i in x.items():
            _check_xy_equal_len(i, prev_y.get(key, None), y.get(key, None), data_loader.lookback[key], data_loader.num_ins[key],
                                data_loader.num_outs[key], num_examples, data_type=data_type
                                )

    elif x is None: # all should be None
        assert all(v is None for v in [x, prev_y, y])
    else:
        raise ValueError


def _check_num_examples(train_x, val_x, test_x, val_ex, test_ex, tot_obs):
    val_examples = 0
    if val_ex:
        val_examples = val_x.shape[0]

    test_examples = 0
    if test_ex:
        test_examples = test_x.shape[0]

    xyz_samples = train_x.shape[0] + val_examples + test_examples
    # todo, whould be equal
    assert xyz_samples == tot_obs, f"""  
    data_loader has {tot_obs} examples while sum of train/val/test examples are {xyz_samples}."""


def check_num_examples(train_x, val_x, test_x, val_ex, test_ex, data_loader):

    if isinstance(train_x, np.ndarray):
        _check_num_examples(train_x, val_x, test_x, val_ex, test_ex, data_loader.tot_obs_for_one_df())

    elif isinstance(train_x, list):
        for idx in range(len(train_x)):
            _check_num_examples(train_x[idx], val_x[idx], test_x[idx], val_ex, test_ex,
                                data_loader.tot_obs_for_one_df()[idx])
    return


def check_inverse_transformation(data, data_loader, y, cols, key):
    if cols is None:
        # not output columns, so not checking
        return

    # check that after inverse transformation, we get correct y.
    if data_loader.source_is_df:
        train_y_ = data_loader.inverse_transform(data=pd.DataFrame(y.reshape(-1, len(cols)), columns=cols), key=key)
        train_y_, index = data_loader.deindexify(train_y_, key=key)
        compare_individual_item(data, key, cols, train_y_, data_loader)

    elif data_loader.source_is_list:
        #for idx in range(data_loader.num_sources):
        #    y_ = y[idx].reshape(-1, len(cols[idx]))
        train_y_ = data_loader.inverse_transform(data=y, key=key)

        train_y_, _ = data_loader.deindexify(train_y_, key=key)

        for idx, y in enumerate(train_y_):
            compare_individual_item(data[idx], f'{key}_{idx}', cols[idx], y, data_loader)

    elif data_loader.source_is_dict:
        train_y_ = data_loader.inverse_transform(data=y, key=key)
        train_y_, _ = data_loader.deindexify(train_y_, key=key)

        for src_name, val in train_y_.items():
            compare_individual_item(data[src_name], f'{key}_{src_name}', cols[src_name], val, data_loader)


def compare_individual_item(data, key, cols, y, data_loader):

    if y is None:
        return

    train_index = data_loader.indexes[key]

    if y.__class__.__name__ in ['DataFrame']:
        y = y.values

    for i, v in zip(train_index, y):
        if len(cols) == 1:
            if isinstance(train_index, pd.DatetimeIndex):
                # if true value in data is None, y's value should also be None
                if np.isnan(data[cols].loc[i]).item():
                    assert np.isnan(v).item()
                else:
                    _t = round(data[cols].loc[i].item(), 0)
                    _p = round(v.item(), 0)
                    if not np.allclose(data[cols].loc[i].item(), v.item()):
                        print(f'true: {_t}, : pred: {_p}, index: {i}, col: {cols}')
            else:
                if isinstance(v, np.ndarray):
                    v = v.item()
                _true = data[cols].loc[i]
                _p = round(v, 3)
                if not np.allclose(v, _true):
                    print(f'true: {_true}, : pred: {_p}, index: {i}, col: {cols}')
        else:
            if isinstance(train_index, pd.DatetimeIndex):
                assert abs(data[cols].loc[i].sum() - np.nansum(v)) <= 0.00001, f'{data[cols].loc[i].sum()},: {v}'
            else:
                assert abs(data[cols].iloc[i].sum() - v.sum()) <= 0.00001


def check_kfold_splits(data_handler):

    if data_handler.source_is_df:

        splits = data_handler.KFold_splits()

        for (train_x, train_y), (test_x, test_y) in splits:

            ... # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    return

def assert_uniquenes(train_y, val_y, test_y, out_cols, data_loader):

    if isinstance(train_y, list):
        assert isinstance(val_y, list)
        assert isinstance(test_y, list)
        train_y = train_y[0]
        val_y = val_y[0]
        test_y = test_y[0]

    if isinstance(train_y, dict):
        train_y = list(train_y.values())[0]
        assert isinstance(val_y, dict)
        isinstance(test_y, dict)
        val_y = list(val_y.values())[0]
        test_y = list(test_y.values())[0]

    if out_cols is not None:
        b = train_y.reshape(-1, )
        if val_y is None:
            a = test_y.reshape(-1, )
        else:
            a = val_y.reshape(-1, )

        if not len(np.intersect1d(a, b)) == 0:
            raise ValueError(f'train and val have overlapping values')

    if data_loader.val_data != 'same' and out_cols is not None and val_y is not None and test_y is not None:

        a = test_y.reshape(-1,)
        b = val_y.reshape(-1,)
        assert len(np.intersect1d(a, b)) == 0, 'test and val have overlapping values'

    return


def build_and_test_loader(data, config, out_cols, train_ex=None, val_ex=None, test_ex=None, save=True,
                          assert_uniqueness=True, check_examples=True,
                          true_train_y=None, true_val_y=None, true_test_y=None):

    config['teacher_forcing'] = True  # todo

    if 'val_fraction' not in config:
        config['val_fraction'] = 0.3
    if 'test_fraction' not in config:
        config['test_fraction'] = 0.3

    data_loader = DataHandler(data=data, save=save, verbosity=0, **config)
    #dl = DataLoader.from_h5('data.h5')
    train_x, prev_y, train_y = data_loader.training_data(key='train')
    assert_xy_equal_len(train_x, prev_y, train_y, data_loader, train_ex)

    val_x, prev_y, val_y = data_loader.validation_data(key='val')
    assert_xy_equal_len(val_x, prev_y, val_y, data_loader, val_ex, data_type='validation')

    test_x, prev_y, test_y = data_loader.test_data(key='test')
    assert_xy_equal_len(test_x, prev_y, test_y, data_loader, test_ex, data_type='test')

    if check_examples:
        check_num_examples(train_x, val_x, test_x, val_ex, test_ex, data_loader)

    if isinstance(data, str):
        data = data_loader.data

    check_inverse_transformation(data, data_loader, train_y, out_cols, 'train')

    if val_ex:
        check_inverse_transformation(data, data_loader, val_y, out_cols, 'val')

    if test_ex:
        check_inverse_transformation(data, data_loader, test_y, out_cols, 'test')

    check_kfold_splits(data_loader)

    if assert_uniqueness:
        assert_uniquenes(train_y, val_y, test_y, out_cols, data_loader)

    if true_train_y is not None:
        assert np.allclose(train_y, true_train_y)

    if true_val_y is not None:
        assert np.allclose(val_y, true_val_y)

    if true_test_y is not None:
        assert np.allclose(test_y, true_test_y)

    return data_loader


class TestAllCases(object):

    def __init__(self, input_features,  output_features, lookback=3, allow_nan_labels=0, save=True):
        self.input_features = input_features
        self.output_features = output_features
        self.lookback = lookback
        self.allow_nan_labels=allow_nan_labels
        self.save=save
        self.run_all()

    def run_all(self):
        all_methods = [m for m in dir(self) if callable(getattr(self, m)) and not m.startswith('_') and m not in ['run_all']]
        for m in all_methods:
            getattr(self, m)()
        return

    def test_basic(self):
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback}

        tr_examples = 49 - (self.lookback - 2) if self.lookback>1 else 49
        val_examples = 22 - (self.lookback - 2) if self.lookback>1 else 22
        test_examples = 30 - (self.lookback - 2) if self.lookback>1 else 30

        if self.output_features == ['c']:
            tty = np.arange(202, 250).reshape(-1, 1, 1)
            tvy = np.arange(250, 271).reshape(-1, 1, 1)
            ttesty = np.arange(271, 300).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, test_examples,
                                       save=self.save,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       true_test_y=ttesty,
                                       check_examples=True,
                                       )
        assert loader.source_is_df

        return

    def test_with_random(self):
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random'}

        tr_examples = 49 - (self.lookback - 2) if self.lookback>1 else 49
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 20, 30,
                                       save=self.save,
                                       )

        assert loader.source_is_df

        return

    def test_drop_remainder(self):
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'batch_size': 8,
                  'drop_remainder': True,
                  'train_data': 'random'}

        loader = build_and_test_loader(data, config, self.output_features,
                                       48, 16, 24,
                                       check_examples=False,
                                       save=self.save,
                                       )

        assert loader.source_is_df

        return

    def test_with_same_val_data(self):
        # val_data is "same" as and train_data is make based upon fractions.
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'val_data': 'same'}

        if self.output_features == ['c']:
            tty = np.arange(202, 271).reshape(-1, 1, 1)
            tvy = np.arange(271, 300).reshape(-1, 1, 1)
            ttesty = np.arange(271, 300).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 71 - (self.lookback - 1) if self.lookback > 1 else 71
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 29, 29,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       true_test_y=ttesty,
                                       save=self.save,
                                       check_examples=False
                                       )

        assert loader.source_is_df
        return

    def test_with_same_val_data_and_random(self):
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random',
                  'val_data': 'same'}

        tr_examples = 70 - (self.lookback - 1) if self.lookback > 1 else 70
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 30, 30,
                                       check_examples=False,
                                       save=self.save
                                       )
        assert loader.source_is_df
        return

    def test_with_no_val_data(self):
        # we dont' want to have any validation_data
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'val_fraction': 0.0}

        if self.output_features == ['c']:
            tty = np.arange(202, 271).reshape(-1, 1, 1)
            ttesty = np.arange(271, 300).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 71 - (self.lookback - 1) if self.lookback > 1 else 71
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 0, 29,
                                       true_train_y=tty,
                                       true_test_y=ttesty,
                                       save=self.save)
        assert loader.source_is_df
        return

    def test_with_no_val_data_with_random(self):
        # we dont' want to have any validation_data
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random',
                  'val_fraction': 0.0}

        tr_examples = 70 - (self.lookback - 1) if self.lookback > 1 else 70
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 0, 30,
                                       save=self.save
                                       )

        assert loader.source_is_df
        return

    def test_with_no_test_data(self):
        # we don't want any test_data
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'test_fraction': 0.0}

        if self.output_features == ['c']:
            tty = np.arange(202, 271).reshape(-1, 1, 1)
            tvy = np.arange(271, 300).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 71 - (self.lookback - 1) if self.lookback > 1 else 71
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 29, 0,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       save=self.save
                                       )

        assert loader.source_is_df
        return

    def test_with_no_test_data_with_random(self):
        # we don't want any test_data
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random',
                  'test_fraction': 0.0,
                  'transformation': 'minmax'}

        tr_examples = 15- (self.lookback - 1) if self.lookback > 1 else 15
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 5, 0,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_dt_index(self):
        # we don't want any test_data
        #print('testing test_with_dt_index', self.lookback)
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=20, freq='D'))
        config = {'input_features': self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random',
                  'test_fraction': 0.0,
                  'transformation': 'minmax'}

        tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 15
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 5, 0,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_intervals(self):
        #print('testing test_with_intervals', self.lookback)
        examples = 35
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=35, freq='D'))

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random',
                  'transformation': 'minmax',
                  'intervals': [(0, 10), (20, 35)]
                  }

        tr_examples = 12 - (self.lookback - 1) if self.lookback > 1 else 12
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 4, 7,
                                       save=self.save
                                       )
        assert loader.source_is_df

        return

    def test_with_dt_intervals(self):
        # check whether indices of intervals can be datetime?
        examples = 35
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=35, freq='D'))

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random',
                  'transformation': 'minmax',
                  'intervals': [('20110101', '20110110'), ('20110121', '20110204')]
                  }

        tr_examples = 12 - (self.lookback - 1) if self.lookback > 1 else 12
        val_examples = 7 - (self.lookback - 2) if self.lookback > 1 else 7
        test_examples = 7 - (self.lookback - 2) if self.lookback > 1 else 7
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 4, 7,
                                       save=self.save)

        assert loader.source_is_df

        return

    def test_with_custom_train_indices(self):
        #print('testing test_with_custom_train_indices')

        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=20, freq='D'))
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': [1,2,3,4,5,6,7,8,9,10,11,12],
                  'transformation': 'minmax',
                  }

        tr_examples = 9 - (self.lookback - 2) if self.lookback > 1 else 9
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, test_examples,
                                       save=self.save)
        assert loader.source_is_df
        return

    def test_with_custom_train_indices_no_val_data(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=20, freq='D'))
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': [1,2,3,4,5,6,7,8,9,10,11,12],
                  'transformation': 'minmax',
                  'val_fraction': 0.0,
                  }

        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features, 12, 0, test_examples,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_custom_train_indices_same_val_data(self):
        #print('testing test_with_custom_train_indices_same_val_data')
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=20, freq='D'))
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback,
                  'train_data': [1,2,3,4,5,6,7,8,9,10,11,12],
                  'transformation': 'minmax',
                  'val_data': 'same',
                  }
        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features, 12, 0, test_examples,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_custom_train_and_val_indices(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=20, freq='D'))
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback,
                  'train_data': [1,2,3,4,5,6,7,8,9,10,11,12],
                  'transformation': 'minmax',
                  'val_data': [0, 12, 14, 16, 5],
                  'val_fraction': 0.0,
                  }

        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features, 12, 5, test_examples,
                                       assert_uniqueness=False,
                                       save=self.save,
                                       check_examples=False
                                       )

        assert loader.source_is_df
        return

    # def test_with_train_and_val_and_test_indices(self):
    #     # todo, does it make sense to define test_data by indices
    #     return

    def test_with_custom_train_indices_and_intervals(self):
        #print('testing test_with_custom_train_indices_and_intervals', self.lookback)
        examples = 30
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=30, freq='D'))
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': [1,2,3,4,5,6,7,8,9,10,11,12],
                  #'transformation': 'minmax',
                  'intervals': [(0, 10), (20, 30)]
                  }

        if self.output_features == ['c']:
            tty = np.array([63., 64., 65., 66., 67., 68., 69., 82.]).reshape(-1, 1, 1)
            tvy = np.arange(83, 87).reshape(-1, 1, 1)
            ttesty = np.array([62., 87., 88., 89.]).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        test_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, test_examples,
                                        true_train_y=tty,
                                        true_val_y=tvy,
                                        true_test_y=ttesty,
                                        save=self.save)
        assert loader.source_is_df
        return

    def test_with_one_feature_transformation(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback,
                  'transformation': [{'method': 'minmax', 'features': ['a']}],
                  }

        if self.output_features == ['c']:
            tty = np.arange(42, 51).reshape(-1, 1, 1)
            tvy = np.arange(51, 55).reshape(-1, 1, 1)
            ttesty = np.arange(55, 60).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       true_test_y=ttesty,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_one_feature_multi_transformation(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback,
                  'transformation': [{'method': 'minmax', 'features': ['a']}, {'method': 'zscore', 'features': ['a']}],
                  }

        if self.output_features == ['c']:
            tty = np.arange(42, 51).reshape(-1, 1, 1)
            tvy = np.arange(51, 55).reshape(-1, 1, 1)
            ttesty = np.arange(55, 60).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       true_test_y=ttesty,
                                       save=self.save)
        assert loader.source_is_df
        return

    def test_with_one_feature_multi_transformation_on_diff_features(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback,
                  'transformation': [{'method': 'minmax', 'features': ['a', 'b', 'c']}, {'method': 'zscore', 'features': ['c']}],
                  }

        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_input_transformation(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback,
                  'transformation': [{'method': 'minmax', 'features': ['a', 'b']}],
                  }

        if self.output_features == ['c']:
            tty = np.arange(42, 51).reshape(-1, 1, 1)
            tvy = np.arange(51, 55).reshape(-1, 1, 1)
            ttesty = np.arange(55, 60).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                      true_train_y=tty,
                                      true_val_y=tvy,
                                      true_test_y=ttesty,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_input_transformation_as_dict(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback,
                  'transformation': {'method': 'minmax', 'features': ['a', 'b']},
                  }

        if self.output_features == ['c']:
            tty = np.arange(42, 51).reshape(-1, 1, 1)
            tvy = np.arange(51, 55).reshape(-1, 1, 1)
            ttesty = np.arange(55, 60).reshape(-1, 1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       true_test_y=ttesty,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_output_transformation(self):
        examples = 20
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'transformation': {'method': 'minmax', 'features': ['c']},
                  }
        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_indices_and_intervals(self):
        examples = 30
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=30, freq='D'))

        config =  {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback, 'train_data': 'random',
                  'transformation': 'minmax',
                  'intervals': [(0, 10), (20, 30)]
                  }

        tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
        val_examples = 5 - (self.lookback - 1) if self.lookback > 1 else 5
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_indices_and_intervals_same_val_data(self):
        examples = 30
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=30, freq='D'))

        config =  {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'lookback': self.lookback, 'train_data': 'random', 'val_data': 'same',
                  'transformation': 'minmax',
                  'intervals': [(0, 10), (20, 30)]
                  }

        tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 13
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 5, 5,
                                       check_examples=False,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_indices_and_intervals_no_val_data(self):
        examples = 30
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=30, freq='D'))

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'lookback': self.lookback,
                  'train_data': 'random', 'val_fraction': 0.0,
                  'transformation': 'minmax',
                  'intervals': [(0, 10), (20, 30)]
                  }
        tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 13
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 0, 5,
                                       save=self.save)

        assert loader.source_is_df
        return

    def test_with_indices_and_nans(self):
        examples = 30
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=30, freq='D'))
        if self.output_features is not None:
            data['c'].iloc[10:20] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                       'train_data': 'random',
                      }
            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 4, 6,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 15
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 6, 9,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_indices_and_nans_interpolate(self):
        examples = 30
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                            index=pd.date_range('20110101', periods=30, freq='D'))

        if self.output_features is not None:
            data['b'].iloc[10:20] = np.nan

            config = {'input_features': self.input_features,
                      'output_features': self.output_features,
                      'lookback': self.lookback,
                      'nan_filler': {'method': 'KNNImputer', 'features': self.input_features},
                      'train_data': 'random',
                      }
            if self.input_features == ['a']:
                tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
                val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
                test_examples = 6
            else:
                tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 15
                val_examples = 6
                test_examples = 9

            build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, test_examples,
                                           save=self.save)

            data['c'].iloc[10:20] = np.nan
            if 'b' not in self.output_features:
                config = {'input_features': self.input_features,
                          'output_features': self.output_features,
                          'lookback': self.lookback,
                          'nan_filler': {'method': 'KNNImputer', 'features': ['b']},
                          'train_data': 'random',
                          }
                tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
                build_and_test_loader(data, config, self.output_features, tr_examples, 4, 6,
                                               save=self.save)

                config = {'input_features': self.input_features,
                          'output_features': self.output_features,
                          'lookback': self.lookback,
                          'nan_filler': {'method': 'KNNImputer', 'features': ['b'], 'imputer_args': {'n_neighbors': 4}},
                          'train_data': 'random',
                          }
                tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
                build_and_test_loader(data, config, self.output_features, tr_examples, 4, 6,
                                               save=self.save)

        return

    def test_with_indices_and_nans_at_irregular_intervals(self):
        if self.output_features is not None and len(self.output_features)>1:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[10:20] = np.nan

            config =  {'input_features':self.input_features,
                       'output_features': self.output_features,
                       'lookback': self.lookback,
                       'train_data': 'random',
                      }

            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 4, 6,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = self.allow_nan_labels
            loader = build_and_test_loader(data, config, self.output_features, 18, 8, 12,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_intervals_and_nans(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                      'intervals': [(0, 10), (20, 40)]
                      }

            tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 5,
                                           check_examples=False,  # todo
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 15
            val_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 7
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, 8,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_intervals_and_nans_at_irregular_intervals(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None and len(self.output_features) > 1:
            examples = 50
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=50, freq='D'))
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[40:50] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                      'intervals': [(0, 10), (20, 50)]
                      }
            loader = build_and_test_loader(data, config, self.output_features, 9, 4, 5,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = self.allow_nan_labels
            loader = build_and_test_loader(data, config, self.output_features, 18, 7, 11,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_intervals_and_nans_same_val_data(self):
        # if data contains nans and we also have intervals and val_data is same
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                       'val_data': 'same',
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 15
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 5, 5,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = self.allow_nan_labels
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 20
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 8, 8,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_intervals_and_nans_at_irregular_intervals_and_same_val_data(self):
        # if data contains nans and we also have intervals and val_data is same
        if self.output_features is not None and len(self.output_features) > 1:
            examples = 50
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=50, freq='D'))
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[40:50] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                       'val_data': 'same',
                      'intervals': [(0, 10), (20, 50)]
                      }
            loader = build_and_test_loader(data, config, self.output_features, 13, 5, 5,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = self.allow_nan_labels
            loader = build_and_test_loader(data, config, self.output_features, 25, 11, 11,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_intervals_and_nans_no_val_data(self):
        # if data contains nans and we also have intervals and val_data is same
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                       'val_fraction': 0.0,
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 15
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 0, 5,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = self.allow_nan_labels
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 20
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 0, 8,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_intervals_and_nans_at_irreg_intervals_and_no_val_data(self):
        # if data contains nans and we also have intervals and val_data is same
        if self.output_features is not None and len(self.output_features) > 1:
            examples = 50
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=50, freq='D'))
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[40:50] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'lookback': self.lookback,
                      'val_fraction': 0.0,
                      'intervals': [(0, 10), (20, 50)]
                      }

            loader = build_and_test_loader(data, config, self.output_features, 13, 0, 5,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = self.allow_nan_labels
            loader = build_and_test_loader(data, config, self.output_features, 25, 0, 11,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_indices_intervals_and_nans(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan

            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                       'train_data': 'random',
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 3, 5,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 15
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 5, 8, save=self.save)
            assert loader.source_is_df

        return

    def test_with_indices_intervals_and_nans_with_same_val_data(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'lookback': self.lookback,
                      'train_data': 'random',
                      'val_data': 'same',
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 13
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 5, 5,
                                           check_examples=False,
                                           save=self.save)
            assert loader.source_is_df

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 20
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 8, 8,
                                           check_examples=False,
                                           save=self.save)

            assert loader.source_is_df
        return

    def test_with_indices_intervals_and_nans_with_no_val_data(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan

            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'lookback': self.lookback,
                       'train_data': 'random',
                       'val_fraction': 0.0,
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 13
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 0, 5,
                                           save=self.save)

            assert loader.source_is_df

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 20
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 0, 8,
                                           save=self.save)
            assert loader.source_is_df
        return

    def test_with_indices_intervals_and_nans_with_no_test_data(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'lookback': self.lookback, 'train_data': 'random', 'test_fraction': 0.0,
                      'intervals': [(0, 10), (20, 40)]
                      }

            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 13
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 5, 0,
                                           save=self.save)

            assert loader.source_is_df

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 20
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, 8, 0,
                                           save=self.save)
            assert loader.source_is_df

        return

    def test_with_custom_indices_intervals_and_nans(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:
            examples = 40
            data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
            data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'lookback': self.lookback,
                      'train_data': [1,2,3,4,5,6,7,8,9,10,11,12],
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
            test_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples, test_examples,
                                           save=self.save)

            assert loader.source_is_df

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
            test_examples = 16 - (self.lookback - 1) if self.lookback > 1 else 16
            loader = build_and_test_loader(data, config, self.output_features, tr_examples, val_examples,
                                           test_examples, save=self.save)
            assert loader.source_is_df
        return


def test_with_random_with_transformation_of_features():
    examples = 100
    data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
    data = pd.DataFrame(data, columns=['a', 'b', 'c'], index=pd.date_range('20110101', periods=len(data), freq='D'))
    data['date'] = data.index
    config = {'input_features':['b'],
              'output_features': ['c'],
              'lookback': 5,
              'train_data': 'random'}

    dh = DataHandler(data, verbosity=0, **config)

    x,y = dh.training_data()

    return


def test_random_with_intervals():
    data = np.random.randint(0, 1000, (40560, 14))
    input_features = [f'input_{i}' for i in range(13)]
    output_features = ['NDX']
    data = pd.DataFrame(data, columns=input_features+output_features)

    out = data["NDX"]

    # put four chunks of missing intervals
    intervals = [(100, 200), (1000, 8000), (10000, 31000)]

    for interval in intervals:
        st, en = interval[0], interval[1]
        out[st:en] = np.nan

    data["NDX"] = out

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'lookback': 5,
        'train_data': 'random',
        'intervals': [(0, 99), (200, 999), (8000, 9999), (31000, 40560)],
    }

    build_and_test_loader(data, config, out_cols=output_features,
                          train_ex=6096, val_ex=2612, test_ex=3733,
                          assert_uniqueness=False,
                          save=False)

    return


def make_cross_validator(cv, **kwargs):

    model = Model(
        model={'randomforestregressor': {}},
        data=arg_beach(),
        cross_validator=cv,
        val_metric="mse",
        verbosity=0,
        **kwargs
    )

    return model


class TestCVs(object):

    def test_kfold(self):
        model = make_cross_validator(cv={'TimeSeriesSplit': {'n_splits': 5}})
        model.cross_val_score()
        model.dh.plot_TimeSeriesSplit_splits(show=False)
        return

    def test_loocv(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        model.cross_val_score()
        model.dh.plot_KFold_splits(show=False)
        return

    def test_tscv(self):
        model = make_cross_validator(cv={'LeaveOneOut': {}}, test_fraction=0.6)
        model.cross_val_score()
        model.dh.plot_LeaveOneOut_splits(show=False)
        return

#
# class TestDataLoader(unittest.TestCase):
#
#     def test_OndDF(self):
#         TestAllCases(
#             input_features = ['a', 'b'],
#             output_features=['c'], allow_nan_labels=2)
#         return
#
#     def test_OneDFTwoOut(self):
#         TestAllCases(input_features = ['a'],
#             output_features=['b', 'c'])
#         return
#
#     def test_MultiSources(self):
#         test_multisource_basic()
#         return
#
#     def test_MultiUnequalSources(self):
#         return

def test_AI4WaterDataSets():
    config = {'intervals': [("20000101", "20011231")],
                                      'input_features': ['precipitation_AWAP',
                                                         'evap_pan_SILO'],
                                      'output_features': ['streamflow_MLd_inclInfilled'],
                                      'dataset_args': {'stations': 1}
    }

    build_and_test_loader('CAMELS_AUS', config=config,
                          out_cols=['streamflow_MLd_inclInfilled'],
                          train_ex=358, val_ex=154, test_ex=219,
                          assert_uniqueness=False,
                          save=False)

    return


def test_multisource_basic():
    examples = 40
    data = np.arange(int(examples * 4), dtype=np.int32).reshape(-1, examples).transpose()
    df1 = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
    df2 = pd.DataFrame(np.array([5,6]).repeat(40, axis=0).reshape(40, -1), columns=['len', 'dep'],
                       index=pd.date_range('20110101', periods=40, freq='D'))

    input_features = [['a', 'b'], ['len', 'dep']]
    output_features = [['c', 'd'], []]
    lookback = 4

    config = {'input_features': input_features,
              'output_features': output_features,
              'lookback': lookback}

    build_and_test_loader(data=[df1, df2], config=config, out_cols=output_features,
                          train_ex=18, val_ex=8, test_ex=11,
                          save=True)


    #  #testing data as a dictionary
    config['input_features'] = {'cont_data': ['a', 'b'], 'static_data': ['len', 'dep']}
    config['output_features'] = {'cont_data': ['c', 'd'], 'static_data': []}
    build_and_test_loader(data={'cont_data': df1, 'static_data': df2},
                          config=config, out_cols=config['output_features'],
                          train_ex=18, val_ex=8, test_ex=11,
                          save=True)

    # #test when output_features for one data is not provided?
    config['input_features'] = {'cont_data': ['a', 'b'], 'static_data': ['len', 'dep']}
    config['output_features'] = {'cont_data': ['c', 'd']}
    build_and_test_loader(data={'cont_data': df1, 'static_data': df2},
                          config=config, out_cols=config['output_features'],
                          train_ex=18, val_ex=8, test_ex=11,
                          save=False)

    # # #testing with transformation
    config['input_features'] = {'cont_data': ['a', 'b'], 'static_data': ['len', 'dep']}
    config['transformation'] = {'cont_data': 'minmax', 'static_data': 'zscore'}
    config['output_features'] = {'cont_data': ['c', 'd'], 'static_data': []}
    build_and_test_loader(data={'cont_data': df1, 'static_data': df2},
                          config=config, out_cols=config['output_features'],
                          train_ex=18, val_ex=8, test_ex=11,
                          save=True)

    # # testing with `same` `val_data`
    config['val_data'] = 'same'
    build_and_test_loader(data={'cont_data': df1, 'static_data': df2},
                          config=config, out_cols=config['output_features'],
                          train_ex=26, val_ex=11, test_ex=11,
                          save=True)

    # # testing with random train indices
    config['val_data'] = 'same'
    config['train_data'] = random.sample(list(np.arange(37)), 25)
    config['input_features'] = {'cont_data': ['a', 'b'], 'static_data': ['len', 'dep']}
    config['output_features'] = {'cont_data': ['c', 'd'], 'static_data': []}
    build_and_test_loader(data={'cont_data': df1, 'static_data': df2},
                          config=config, out_cols=config['output_features'],
                          train_ex=25, val_ex=12, test_ex=12,
                          save=True)
    return


def test_multisource_basic2():
    examples = 40
    data = np.arange(int(examples * 4), dtype=np.int32).reshape(-1, examples).transpose()
    df1 = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
    df2 = pd.DataFrame(np.array([[5],[6], [7]]).repeat(40, axis=1).transpose(), columns=['len', 'dep', 'y'],
                       index=pd.date_range('20110101', periods=40, freq='D'))
    input_features = [['a', 'b'], ['len', 'dep']]
    output_features = [['c', 'd'], ['y']]
    lookback = 4

    config = {'input_features': input_features,
              'output_features': output_features,
              'lookback': lookback}

    build_and_test_loader(data=[df1, df2], config=config, out_cols=output_features,
                          train_ex=18, val_ex=8, test_ex=11,
                          save=True)

    config['input_features'] = {'cont_data': ['a', 'b'], 'static_data': ['len', 'dep']}
    config['output_features'] = {'cont_data': ['c', 'd'], 'static_data': ['y']}
    build_and_test_loader(data={'cont_data': df1, 'static_data': df2},
                          config=config,
                          out_cols=config['output_features'],
                          train_ex=18, val_ex=8, test_ex=11,
                          save=True)
    return


def test_multisource_basic3():
    examples = 40
    data = np.arange(int(examples * 5), dtype=np.int32).reshape(-1, examples).transpose()
    y_df = pd.DataFrame(data[:, -1], columns=['y'])
    y_df.loc[y_df.sample(frac=0.5).index] = np.nan

    cont_df = pd.DataFrame(data[:, 0:4], columns=['a', 'b', 'c', 'd'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
    static_df = pd.DataFrame(np.array([[5],[6], [7]]).repeat(40, axis=1).transpose(), columns=['len', 'dep', 'y'],
                       index=pd.date_range('20110101', periods=40, freq='D'))
    disc_df = pd.DataFrame(np.random.randint(0, 10, (40, 4)), columns=['cl', 'o', 'do', 'bod'],
                                index=pd.date_range('20110101', periods=40, freq='D'))

    cont_df['y'] =  y_df.values
    static_df['y'] = y_df.values
    disc_df['y'] = y_df.values

    input_features = [['len', 'dep'], ['a', 'b'],  ['cl', 'o', 'do', 'bod']]
    output_features = [['y'], ['c', 'y'], ['y']]
    lookback = [1, 4, 1]
    config = {'input_features': input_features,
              'output_features': output_features,
              'test_fraction': 0.3,
              'val_fraction': 0.3,
              'lookback': lookback}

    # build_and_test_loader(data=[static_df, cont_df, disc_df], config=config, out_cols=output_features, train_ex=6,
    #                   val_ex=4,
    #                   test_ex=6, save=True)

    data_handler = DataHandler(data=[static_df, cont_df, disc_df], verbosity=0, **config)
    data_handler.training_data()
    data_handler.validation_data()
    data_handler.test_data()
    return


def test_multisource_multi_loc():
    examples = 40
    data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
    training_data = pd.DataFrame(data, columns=['a', 'b', 'c'],
                        index=pd.date_range('20110101', periods=40, freq='D'))

    val_data = pd.DataFrame(data+1000.0, columns=['a', 'b', 'c'],
                        index=pd.date_range('20110101', periods=40, freq='D'))

    test_data = pd.DataFrame(data+2000, columns=['a', 'b', 'c'],
                        index=pd.date_range('20110101', periods=40, freq='D'))

    dh = MultiLocDataHandler()

    train_x, train_y = dh.training_data(data=training_data, input_features=['a', 'b'],  output_features=['c'])
    valx, val_y = dh.validation_data(data=val_data, input_features=['a', 'b'], output_features=['c'])
    test_x, test_y = dh.test_data(data=test_data, input_features=['a', 'b'], output_features=['c'])

    assert np.allclose(train_y.reshape(-1,), training_data['c'].values.reshape(-1, ))
    assert np.allclose(val_y.reshape(-1, ), val_data['c'].values.reshape(-1, ))
    assert np.allclose(test_y.reshape(-1, ), test_data['c'].values.reshape(-1, ))
    return


def test_multisource_basic4():
    examples = 40
    data = np.arange(int(examples * 4), dtype=np.int32).reshape(-1, examples).transpose()
    df1 = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'],
                                index=pd.date_range('20110101', periods=40, freq='D'))
    df2 = pd.DataFrame(np.array([5,6]).repeat(40, axis=0).reshape(40, -1), columns=['len', 'dep'],
                       index=pd.date_range('20110101', periods=40, freq='D'))

    input_features = {'cont_data':['a', 'b'], 'static_data':['len', 'dep']}
    output_features = {'cont_data': ['c', 'd']}
    lookback = {'cont_data': 4, 'static_data': 1}

    config = {'input_features': input_features,
              'output_features': output_features,
              'lookback': lookback}

    build_and_test_loader(data={'cont_data': df1, 'static_data': df2},
                          config=config, out_cols=output_features,
                          train_ex=18, val_ex=8, test_ex=11,
                          save=False)
    return


def site_distributed_basic():
    examples = 50
    data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
    df = pd.DataFrame(data, columns=['a', 'b', 'c'],
                        index=pd.date_range('20110101', periods=examples, freq='D'))
    config = {'input_features': ['a', 'b'],
              'output_features': ['c'],
              'lookback': 4,
              'val_fraction': 0.3,
              'test_fraction': 0.3,
              'verbosity': 0}
    data = {'0': df, '1': df+1000, '2': df+2000, '3': df+3000}
    configs = {'0': config, '1': config, '2': config, '3': config}

    dh = SiteDistributedDataHandler(data, configs, verbosity=0)
    train_x, train_y = dh.training_data()
    val_x, val_y = dh.validation_data()
    test_x, test_y = dh.test_data()
    assert train_x.shape == (23, len(data),  config['lookback'], len(config['input_features']))
    assert val_x.shape == (10, len(data),  config['lookback'], len(config['input_features']))
    assert test_x.shape == (14, len(data),  config['lookback'], len(config['input_features']))

    dh = SiteDistributedDataHandler(data, configs, training_sites=['0', '1'], validation_sites=['2'],
                                    test_sites=['3'], verbosity=0)
    train_x, train_y = dh.training_data()
    val_x, val_y = dh.validation_data()
    test_x, test_y = dh.test_data()

    assert train_x.shape == (len(df)-config['lookback']+1, 2,  config['lookback'], len(config['input_features']))
    assert val_x.shape == (len(df)-config['lookback']+1, 1, config['lookback'], len(config['input_features']))
    assert test_x.shape == (len(df)-config['lookback']+1, 1, config['lookback'], len(config['input_features']))


def site_distributed_diff_lens():
    examples = 50
    data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
    df = pd.DataFrame(data, columns=['a', 'b', 'c'],
                        index=pd.date_range('20110101', periods=examples, freq='D'))
    config = {'input_features': ['a', 'b'],
              'output_features': ['c'],
              'lookback': 4,
              'verbosity': 0}
    data = {'0': df,
            '1': pd.concat([df, df], axis=0)+1000,
            '2': pd.concat([df, df, df], axis=0)+2000,
            '3': df +3000
            }
    configs = {'0': config, '1': config, '2': config, '3': config}

    #dh = SiteDistributedDataHandler(data, configs) # This should raise NotImplementedError
    dh = SiteDistributedDataHandler(data, configs, allow_variable_len=True, verbosity=0)
    train_x, train_y = dh.training_data()
    val_x, val_y = dh.validation_data()
    test_x, test_y = dh.test_data()
    assert isinstance(train_x, dict)


    dh = SiteDistributedDataHandler(data, configs, training_sites=['0', '1'], validation_sites=['2'],
                                    test_sites=['3'], allow_variable_len=True, verbosity=0)
    train_x, train_y = dh.training_data()
    val_x, val_y = dh.validation_data()
    test_x, test_y = dh.test_data()
    assert isinstance(train_x, dict)


def site_distributed_multiple_srcs():
    examples = 40
    data = np.arange(int(examples * 4), dtype=np.int32).reshape(-1, examples).transpose()
    cont_df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'],
                                index=pd.date_range('20110101', periods=examples, freq='D'))
    static_df = pd.DataFrame(np.array([[5],[6], [7]]).repeat(examples, axis=1).transpose(), columns=['len', 'dep', 'width'],
                       index=pd.date_range('20110101', periods=examples, freq='D'))

    config = {'input_features': {'cont_data': ['a', 'b', 'c'], 'static_data': ['len', 'dep', 'width']},
              'output_features': {'cont_data': ['d']},
              'lookback': {'cont_data': 4, 'static_data':1},
              'verbosity': 0
              }

    data = {'cont_data': cont_df, 'static_data': static_df}
    datas = {'0': data, '1': data, '2': data, '3': data, '4': data, '5': data, '6': data}
    configs = {'0': config, '1': config, '2': config, '3': config, '4': config, '5': config, '6': config}

    dh = SiteDistributedDataHandler(datas, configs, verbosity=0)
    train_x, train_y = dh.training_data()
    val_x, val_y = dh.validation_data()
    test_x, test_y = dh.test_data()

    dh = SiteDistributedDataHandler(datas, configs, training_sites=['0', '1', '2'],
                                    validation_sites=['3', '4'], test_sites=['5', '6'], verbosity=0)
    train_x, train_y = dh.training_data()
    val_x, val_y = dh.validation_data()
    test_x, test_y = dh.test_data()


def test_with_string_index():

    data = arg_beach()
    data.index = [f"ind_{i}" for i in range(len(data))]
    config = {
        #'input_features': ['x1', 'x2', 'x3', 'x4'],
        #'output_features': ['target'],
        'input_features': ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c'],
        'output_features': ['tetx_coppml'],
        'lookback': 3,
        'train_data': 'random',
        'transformation': 'minmax'
    }

    #build_and_test_loader(data, config, out_cols=['target'], train_ex=136, val_ex=58, test_ex=84)
    build_and_test_loader(data, config, out_cols=['tetx_coppml'], train_ex=106, val_ex=46, test_ex=66)
    return

def test_with_indices_and_nans():
    # todo, check with two output columns
    data = arg_beach()
    train_idx, test_idx = train_test_split(np.arange(len(data.dropna())),
                                           test_size=0.25, random_state=332898)
    out_cols = [list(data.columns)[-1]]
    config = {
        'train_data': train_idx,
        'input_features': list(data.columns)[0:-1],
        'output_features': out_cols,
        'lookback': 14,
        'val_data': 'same'
    }

    build_and_test_loader(data,
                          config,
                          out_cols=out_cols, train_ex=163, val_ex=55, test_ex=55,
                          check_examples=False, save=False)

def test_file_formats(data):
    csv_fname = os.path.join(os.getcwd(), "results", "data.csv")
    data.to_csv(csv_fname)

    xlsx_fname = os.path.join(os.getcwd(), "results", "data.xlsx")
    data.to_excel(xlsx_fname, engine="xlsxwriter")

    parq_fname = os.path.join(os.getcwd(), "results", "data.parquet")
    data.to_parquet(parq_fname)

    feather_fname = os.path.join(os.getcwd(), "results", "data.feather")
    data.reset_index().to_feather(feather_fname)

    nc_fname = os.path.join(os.getcwd(), "results", "data.nc")
    xds = data.to_xarray()
    xds.to_netcdf(nc_fname)

    npz_fname = os.path.join(os.getcwd(), "results", "data.npz")
    np.savez(npz_fname, data.values)

    mat_fname = os.path.join(os.getcwd(), "results", "data.mat")
    scipy.io.savemat(mat_fname, {'data': data.values})

    dh = DataHandler(data, verbosity=0)
    input_features = dh.input_features
    output_features = dh.output_features
    train_x, train_y = dh.training_data()
    val_x, val_y = dh.validation_data()
    test_x, test_y = dh.test_data()
    train_x_shape, train_y_shape = train_x.shape, train_y.shape
    val_x_shape, val_y_shape = val_x.shape, val_y.shape
    test_x_shape, test_y_shape = test_x.shape, test_y.shape

    for fname in [csv_fname,
                  xlsx_fname, parq_fname,
                  feather_fname,
                  nc_fname,
                  npz_fname,
                  mat_fname
                  ]:
        #print(f'readeing {fname}')
        dh = DataHandler(fname,
                         input_features=input_features,
                         output_features=output_features,
                         verbosity=0)

        train_x, train_y = dh.training_data()
        assert train_x.shape == train_x_shape
        assert train_y.shape == train_y_shape

        val_x, val_y = dh.validation_data()
        assert val_x.shape == val_x_shape
        assert val_y.shape == val_y_shape

        test_x, test_y = dh.test_data()
        assert test_x.shape == test_x_shape
        assert test_y.shape == test_y_shape

    return

test_with_indices_and_nans()
test_with_string_index()
site_distributed_basic()
site_distributed_diff_lens()
site_distributed_multiple_srcs()
test_multisource_multi_loc()
test_with_random_with_transformation_of_features()
test_random_with_intervals()
test_AI4WaterDataSets()
test_multisource_basic()
test_multisource_basic2()
test_multisource_basic3()
# #test_multisource_basic4() todo
test_file_formats(arg_beach())

cv_tester = TestCVs()
cv_tester.test_loocv()
cv_tester.test_tscv()
cv_tester.test_kfold()


# TestAllCases(input_features = ['a', 'b'],
#             output_features=['c'], lookback=1, save=False, allow_nan_labels=2)

# # ##testing single dataframe with single output and multiple inputs
TestAllCases(input_features = ['a', 'b'],
           output_features=['c'], allow_nan_labels=2, save=False)
#
# # ## ##testing single dataframe with multiple output and sing inputs
TestAllCases(input_features = ['a'],
            output_features=['b', 'c'], allow_nan_labels=1, save=False)
# #
# #  ##testing single dataframe with all inputs and not output
TestAllCases(input_features = ['a', 'b', 'c'],
            output_features=None)

