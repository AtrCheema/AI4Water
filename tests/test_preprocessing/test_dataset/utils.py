
import numpy as np
import pandas as pd

from ai4water.preprocessing import DataSet


def make_df(examples: int, columns: list, add=0) -> pd.DataFrame:

    data = np.arange(int(examples * len(columns)),
                     dtype=np.int32).reshape(-1, examples).transpose()

    data = data + add

    dataframe = pd.DataFrame(data,
                             columns=columns,
                             index=pd.date_range('20110101', periods=examples, freq='D'))
    return dataframe


def _check_xy_equal_len(x, prev_y, y, lookback, num_ins, num_outs, num_examples,
                        data_type='training'):

    feat_dim = 1
    if lookback > 1:
        assert x.shape[1] == lookback
        feat_dim = 2

    assert x.shape[feat_dim] == num_ins, f"""
for {data_type} x's shape is {x.shape} while num_ins of dataloader are {num_ins}"""

    if y is not None:
        assert y.shape[1] == num_outs, f"""
for {data_type} y's shape is {y.shape} while num_outs of dataloader are {num_outs}"""
    else:
        assert num_outs == 0
        y = x  # just for next statement to run

    if prev_y is None:
        prev_y = x  # just for next statement to run

    assert x.shape[0] == y.shape[0] == prev_y.shape[0], f"""
    for {data_type} xshape: {x.shape}, yshape: {y.shape}, prevyshape: {prev_y.shape}"""

    if num_examples:
        assert x.shape[0] == num_examples, f"""
for {data_type} x contains {x.shape[0]} samples while expected samples are {num_examples}"""
    return



def assert_xy_equal_len(x, prev_y, y, ds, num_examples=None, data_type='training'):

    if isinstance(x, np.ndarray):
        _check_xy_equal_len(x, prev_y, y, ds.lookback, ds.num_ins, ds.num_outs, num_examples,
                            data_type=data_type)

    elif isinstance(x, list):

        while len(y)<len(x):
            y.append(None)

        for idx, i in enumerate(x):
            _check_xy_equal_len(i, prev_y[idx], y[idx], ds.lookback[idx], ds.num_ins[idx],
                                ds.num_outs[idx], num_examples, data_type=data_type
                                )

    elif isinstance(x, dict):
        for key, i in x.items():
            _check_xy_equal_len(i, prev_y.get(key, None), y.get(key, None), ds.lookback[key],
                                ds.num_ins[key], ds.num_outs[key], num_examples,
                                data_type=data_type)

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


def check_num_examples(train_x, val_x, test_x, val_ex, test_ex, ds):

    if isinstance(train_x, np.ndarray):
        _check_num_examples(train_x, val_x, test_x, val_ex, test_ex, ds.total_exs(**ds.ts_args))

    elif isinstance(train_x, list):
        for idx in range(len(train_x)):
            _check_num_examples(train_x[idx], val_x[idx], test_x[idx], val_ex, test_ex,
                                ds.total_exs(**ds.ts_args)[idx])
    return


def check_kfold_splits(data_handler):

    splits = data_handler.KFold_splits()

    for (train_x, train_y), (test_x, test_y) in splits:

        assert len(train_x)==len(train_y)
        assert len(test_x) == len(test_y)

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

    if out_cols is not None and len(val_y) != 0 and len(test_y) != 0:

        a = test_y.reshape(-1,)
        b = val_y.reshape(-1,)
        assert len(np.intersect1d(a, b)) == 0, 'test and val have overlapping values'

    return


def build_and_test_loader(data, config, out_cols,
                          train_ex=None, val_ex=None, test_ex=None,
                          save=True,
                          assert_uniqueness=True, check_examples=True,
                          true_train_y=None, true_val_y=None, true_test_y=None):

    config['teacher_forcing'] = True  # todo

    if 'val_fraction' not in config:
       config['val_fraction'] = 0.3
    if 'train_fraction' not in config:
       config['train_fraction'] = 0.7

    data_loader = DataSet(data=data, save=save, verbosity=0, **config)
    #dl = DataLoader.from_h5('data.h5')
    train_x, prev_y, train_y = data_loader.training_data(key='train')
    assert_xy_equal_len(train_x, prev_y, train_y, data_loader, train_ex)

    val_x, prev_y, val_y = data_loader.validation_data(key='val')
    if data_loader.val_fraction==0.0:
        assert len(val_x)==0
    else:
        assert_xy_equal_len(val_x, prev_y, val_y, data_loader, val_ex, data_type='validation')

    test_x, prev_y, test_y = data_loader.test_data(key='test')
    if data_loader.train_fraction < 1.0:
        assert_xy_equal_len(test_x, prev_y, test_y, data_loader, test_ex, data_type='test')
    else:
        assert len(test_x) == 0

    if check_examples:
        check_num_examples(train_x, val_x, test_x, val_ex, test_ex, data_loader)

    if isinstance(data, str):
        data = data_loader.data

    if test_ex == 0:
        assert len(test_x) == 0

    if val_ex == 0:
        assert len(val_x) == 0

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

    def test_basic(self, **kwargs):
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                   }

        tr_examples = 48 - (self.lookback - 2) if self.lookback>1 else 49
        val_examples = 22 - (self.lookback - 2) if self.lookback>1 else 21
        test_examples = 31 - (self.lookback - 2) if self.lookback>1 else 30

        if self.output_features == ['c'] and self.lookback==3:
            tty = np.arange(202, 249).reshape(-1, 1)
            tvy = np.arange(249, 270).reshape(-1, 1)
            ttesty = np.arange(270, 300).reshape(-1, 1)
        elif self.lookback == 1:
            tty = np.arange(200, 249).reshape(-1, 1)
            tvy = np.arange(249, 270).reshape(-1, 1)
            ttesty = np.arange(270, 300).reshape(-1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, test_examples,
                                       save=self.save,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       true_test_y=ttesty)
        assert isinstance(loader, DataSet)

        return

    def test_with_random(self):
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': True}

        tr_examples = 48 - (self.lookback - 2) if self.lookback>1 else 49

        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 21, 30,
                                       save=self.save)
        assert isinstance(loader, DataSet)

        return

    def test_drop_remainder(self):
        examples = 100
        data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        data = pd.DataFrame(data, columns=['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'batch_size': 8,
                  'drop_remainder': True,
                  'split_random': True}

        train_exs = 40 if self.lookback > 1 else 48
        loader = build_and_test_loader(data, config, self.output_features,
                                       train_exs, 16, 24,
                                       check_examples=False,
                                       save=self.save)
        assert isinstance(loader, DataSet)
        return

    def test_with_no_val_data(self):
        # we dont' want to have any validation_data

        data = make_df(100, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'train_fraction': 0.7,
                  'val_fraction': 0.0}

        if self.output_features == ['c'] and self.lookback == 3:
            tty = np.arange(202, 270).reshape(-1, 1)
            ttesty = np.arange(270, 300).reshape(-1, 1)
        elif self.lookback == 1:
            tty = np.arange(200, 270).reshape(-1, 1)
            ttesty = np.arange(270, 300).reshape(-1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 70 - (self.lookback - 1) if self.lookback > 1 else 70
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 0, 30,
                                       true_train_y=tty,
                                       true_test_y=ttesty,
                                       save=self.save)
        assert isinstance(loader, DataSet)
        return


    def test_with_no_val_data_with_random(self):
        # we dont' want to have any validation_data

        data = make_df(100, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': True,
                  'train_fraction': 0.7,
                  'val_fraction': 0.0}

        tr_examples = 70 - (self.lookback - 1) if self.lookback > 1 else 70
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 0, 30,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_no_test_data(self):
        # we don't want any test_data

        data = make_df(100, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'train_fraction': 1.0
                  }

        if self.output_features == ['c'] and self.lookback == 3:
            tty = np.arange(202, 270).reshape(-1, 1)
            tvy = np.arange(270, 300).reshape(-1, 1)
        elif self.lookback == 1:
            tty = np.arange(200, 270).reshape(-1, 1)
            tvy = np.arange(270, 300).reshape(-1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 70 - (self.lookback - 1) if self.lookback > 1 else 70
        loader = build_and_test_loader(data, config, self.output_features, tr_examples, 30, 0,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       save=self.save
                                       )

        assert isinstance(loader, DataSet)
        return

    def test_with_no_test_data_with_random(self):
        # we don't want any test_data

        data = make_df(20, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': True,
                  'train_fraction': 1.0}

        tr_examples = 14 - (self.lookback - 1) if self.lookback > 1 else 14
        loader = build_and_test_loader(data, config, self.output_features, tr_examples,
                                       6, 0,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_dt_index(self):
        # we don't want any test_data

        data = make_df(20, ['a', 'b', 'c'])

        config = {'input_features': self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': True,
                  'train_fraction': 1.0,
                  }

        tr_examples = 14 - (self.lookback - 1) if self.lookback > 1 else 14
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 6, 0, save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_intervals(self):

        data = make_df(35, ['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': True,
                  'intervals': [(0, 10), (20, 35)]
                  }

        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 5 if self.lookback > 1 else 6
        test_examples = 7 if self.lookback > 1 else 8

        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, test_examples,
                                       save=self.save)
        assert isinstance(loader, DataSet)

        return

    def test_with_dt_intervals(self):
        # check whether indices of intervals can be datetime?

        data = make_df(35, ['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': True,
                  'intervals': [('20110101', '20110110'), ('20110121', '20110204')]
                  }

        tr_examples = 11 - (self.lookback - 1) if self.lookback > 1 else 11
        val_examples = 6 - (self.lookback - 2) if self.lookback > 1 else 6
        test_examples = 8 - (self.lookback - 2) if self.lookback > 1 else 8

        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, test_examples,
                                       save=self.save)

        assert isinstance(loader, DataSet)

        return

    def test_with_custom_train_indices(self):

        data = make_df(20, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'indices': {'training': [1,2,3,4,5,6,7,8,9,10,11,12]},
                  'train_fraction': 0.7
                  }

        tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 8
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 4
        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, test_examples,
                                       save=self.save)
        assert isinstance(loader, DataSet)
        return

    def test_with_custom_train_indices_no_val_data(self):

        data = make_df(20, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'indices': {'training': [1,2,3,4,5,6,7,8,9,10,11,12]},
                  'val_fraction': 0.0,
                  'train_fraction': 0.7
                  }

        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features,
                                       12, 0, test_examples,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_custom_train_indices1(self):

        data = make_df(20, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'indices': {'training': [1,2,3,4,5,6,7,8,9,10,11,12]},
                  'train_fraction': 0.7,
                  }
        train_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 8
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 4
        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features,
                                       train_examples, val_examples, test_examples,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_custom_train_and_val_indices(self):

        data = make_df(20, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'ts_args': {'lookback': self.lookback},
                  'indices': {'training': [1,2,3,4,5,6,7,8,9,10,11,12],
                  'val_indices': [0, 12, 14, 16, 5]},
                  'train_fraction': 0.7,
                  }

        train_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 8
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 4
        test_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features,
                                       train_examples, val_examples, test_examples,
                                       assert_uniqueness=False,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_custom_train_indices_and_intervals(self):

        data = make_df(30, ['a', 'b', 'c'])
        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'indices': {'training': [1,2,3,4,5,6,7,8,9,10,11,12]},
                  'intervals': [(0, 10), (20, 30)],
                  'train_fraction': 0.7
                  }

        if self.output_features == ['c'] and self.lookback == 3:
            tty = np.array([63., 64., 65., 66., 67., 68., 69., 82.]).reshape(-1, 1)
            tvy = np.arange(83, 87).reshape(-1, 1)
            ttesty = np.array([62., 87., 88., 89.]).reshape(-1, 1)
        elif self.lookback == 1:
            tty = np.array(range(61, 69)).reshape(-1, 1)
            tvy = np.array([69, 80, 81, 82]).reshape(-1, 1)
            ttesty = np.array([60, 83, 84, 85, 86, 87, 88, 89]).reshape(-1, 1)
        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 8
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 4
        test_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 8
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, test_examples,
                                        true_train_y=tty,
                                        true_val_y=tvy,
                                        true_test_y=ttesty,
                                        save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_one_feature_transformation(self):

        data = make_df(20, ['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'ts_args': {'lookback': self.lookback},
                  }

        ttesty = np.arange(54, 60).reshape(-1, 1)

        if self.output_features == ['c'] and self.lookback == 3:
            tty = np.arange(42, 50).reshape(-1, 1)
            tvy = np.arange(50, 54).reshape(-1, 1)
        elif self.lookback == 1:
            tty = np.arange(40, 49).reshape(-1, 1)
            tvy = np.arange(49, 54).reshape(-1, 1)

        else:
            tty, tvy, ttesty = None, None, None

        tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 5
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, 6,
                                       true_train_y=tty,
                                       true_val_y=tvy,
                                       true_test_y=ttesty,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_indices_and_intervals(self):

        data = make_df(30, ['a', 'b', 'c'])

        config =  {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': True,
                  'intervals': [(0, 10), (20, 30)]
                  }

        tr_examples = 9 - (self.lookback - 1) if self.lookback > 1 else 9
        val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 5
        test_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, test_examples,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_random_and_intervals_no_test_data(self):

        data = make_df(30, ['a', 'b', 'c'])

        config =  {'input_features':self.input_features,
                  'output_features': self.output_features,
                   'ts_args': {'lookback': self.lookback},
                   'split_random': True,
                   'train_fraction': 1.0,
                  'intervals': [(0, 10), (20, 30)]
                  }

        tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 14
        val_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, val_examples, 0,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_indices_and_intervals_no_val_data(self):

        data = make_df(30, ['a', 'b', 'c'])

        config = {'input_features':self.input_features,
                  'output_features': self.output_features,
                  'ts_args': {'lookback': self.lookback},
                  'split_random': 'random',
                  'val_fraction': 0.0,
                  'intervals': [(0, 10), (20, 30)]
                  }
        tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 14
        test_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
        loader = build_and_test_loader(data, config, self.output_features,
                                       tr_examples, 0, test_examples,
                                       save=self.save)

        assert isinstance(loader, DataSet)
        return

    def test_with_random_and_nans(self):

        data = make_df(30, ['a', 'b', 'c'])
        if self.output_features is not None:
            data['c'].iloc[10:20] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'ts_args': {'lookback': self.lookback},
                       'split_random': True,
                      }
            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 5
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, 6,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 14
            val_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 7
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, 9,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_random_and_nans_interpolate(self):

        data = make_df(30, ['a', 'b', 'c'])

        if self.output_features is not None:
            data['b'].iloc[10:20] = np.nan

            config = {'input_features': self.input_features,
                      'output_features': self.output_features,
                      'ts_args': {'lookback': self.lookback},
                      'nan_filler': {'method': 'KNNImputer',
                                     'features': self.input_features},
                      'split_random': True,
                      }
            if self.input_features == ['a']:
                tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
                val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 6
                test_examples = 6
            else:
                tr_examples = 15 - (self.lookback - 1) if self.lookback > 1 else 14
                val_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 7
                test_examples = 9

            build_and_test_loader(data, config, self.output_features,
                                  tr_examples, val_examples, test_examples,
                                  save=self.save)

            data['c'].iloc[10:20] = np.nan
            if 'b' not in self.output_features:
                config = {'input_features': self.input_features,
                          'output_features': self.output_features,
                          'ts_args': {'lookback': self.lookback},
                          'nan_filler': {'method': 'KNNImputer', 'features': ['b']},
                          'split_random': True,
                          }
                tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
                val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 5
                build_and_test_loader(data, config, self.output_features,
                                      tr_examples, val_examples, 6,
                                      save=self.save)

                config = {'input_features': self.input_features,
                          'output_features': self.output_features,
                          'ts_args': {'lookback': self.lookback},
                          'nan_filler': {'method': 'KNNImputer',
                                         'features': ['b'],
                                         'imputer_args': {'n_neighbors': 4}},
                          'split_random': True,
                          }
                tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
                val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 5
                loader = build_and_test_loader(data, config, self.output_features,
                                      tr_examples, val_examples, 6,
                                      save=self.save)

                assert isinstance(loader, DataSet)
        return

    def test_with_random_and_nans_at_irregular_intervals(self):
        if self.output_features is not None and len(self.output_features)>1:

            data = make_df(40, ['a', 'b', 'c'])
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[10:20] = np.nan

            config =  {'input_features':self.input_features,
                       'output_features': self.output_features,
                       'ts_args': {'lookback': self.lookback},
                       'split_random': True,
                      }

            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 10
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 4, 6,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = self.allow_nan_labels
            loader = build_and_test_loader(data, config, self.output_features,
                                           18, 8, 12,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_intervals_and_nans(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:

            data = make_df(40, ['a', 'b', 'c'])
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'ts_args': {'lookback': self.lookback},
                      'intervals': [(0, 10), (20, 40)]
                      }

            tr_examples = 9 - (self.lookback - 1) if self.lookback > 1 else 9
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 5
            test_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, test_examples,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 14 - (self.lookback - 1) if self.lookback > 1 else 14
            val_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 7
            test_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, test_examples,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_intervals_and_nans_at_irregular_intervals(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None and len(self.output_features) > 1:

            data = make_df(50, ['a', 'b', 'c'])
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[40:50] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'ts_args': {'lookback': self.lookback},
                      'intervals': [(0, 10), (20, 50)]
                      }
            tr_examples = 9 - (self.lookback - 1) if self.lookback > 1 else 9
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 4, 5,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = self.allow_nan_labels
            tr_examples = 19 - (self.lookback - 1) if self.lookback > 1 else 17
            val_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 7
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, 11,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_intervals_and_nans_no_test_data(self):
        # if data contains nans and we also have intervals and no test data
        if self.output_features is not None:

            data = make_df(40, ['a', 'b', 'c'])
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'ts_args': {'lookback': self.lookback},
                       'train_fraction': 1.0,
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 14
            val_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, 0,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = self.allow_nan_labels
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 21
            val_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, 0,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_intervals_and_nans_at_irregular_intervals_and_no_test_data(self):
        # if data contains nans and we also have intervals and val_data is same
        if self.output_features is not None and len(self.output_features) > 1:

            data = make_df(50, ['a', 'b', 'c'])
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[40:50] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'ts_args': {'lookback': self.lookback},
                       'train_fraction': 1.0,
                      'intervals': [(0, 10), (20, 50)]
                      }
            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 11
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 5, 0,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = self.allow_nan_labels
            tr_examples = 27 - (self.lookback - 1) if self.lookback > 1 else 25
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 11, 0,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_intervals_and_nans_no_val_data(self):
        # if data contains nans and we also have intervals and val_data is same
        if self.output_features is not None:

            data = make_df(40, ['a', 'b', 'c'])
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'ts_args': {'lookback': self.lookback},
                       'val_fraction': 0.0,
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 14
            test_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 0, test_examples,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = self.allow_nan_labels
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 21
            test_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 0, test_examples,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_intervals_and_nans_at_irreg_intervals_and_no_val_data(self):
        # if data contains nans and we also have intervals and val_data is same
        if self.output_features is not None and len(self.output_features) > 1:

            data = make_df(50, ['a', 'b', 'c'])
            data['b'].iloc[20:30] = np.nan
            data['c'].iloc[40:50] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'ts_args': {'lookback': self.lookback},
                      'val_fraction': 0.0,
                      'intervals': [(0, 10), (20, 50)]
                      }

            loader = build_and_test_loader(data, config, self.output_features,
                                           11, 0, 5,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = self.allow_nan_labels
            loader = build_and_test_loader(data, config, self.output_features,
                                           25, 0, 11,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_indices_intervals_and_nans(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:

            data = make_df(40, ['a', 'b', 'c'])
            data['c'].iloc[20:30] = np.nan

            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'ts_args': {'lookback': self.lookback},
                       'split_random': True,
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 9- (self.lookback - 1) if self.lookback > 1 else 9
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 5
            test_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, test_examples,
                                           save=self.save)
            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 14 - (self.lookback - 1) if self.lookback > 1 else 14
            val_examples = 8 - (self.lookback - 1) if self.lookback > 1 else 7
            test_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, test_examples,
                                           save=self.save)
            assert isinstance(loader, DataSet)

        return

    def test_with_indices_intervals_and_nans_with_no_val_data(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:

            data = make_df(40, ['a', 'b', 'c'])
            data['c'].iloc[20:30] = np.nan

            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                       'ts_args': {'lookback': self.lookback},
                       'split_random': True,
                       'val_fraction': 0.0,
                      'intervals': [(0, 10), (20, 40)]
                      }
            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 14
            test_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 0, test_examples,
                                           save=self.save)

            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 21
            test_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, 0, test_examples,
                                           save=self.save)
            assert isinstance(loader, DataSet)
        return

    def test_with_indices_intervals_and_nans_with_no_test_data(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:

            data = make_df(40, ['a', 'b', 'c'])
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'ts_args': {'lookback': self.lookback},
                       'split_random': True,
                       'train_fraction': 1.0,
                      'intervals': [(0, 10), (20, 40)]
                      }

            tr_examples = 13 - (self.lookback - 1) if self.lookback > 1 else 14
            val_examples = 7 - (self.lookback - 1) if self.lookback > 1 else 6
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, 0,
                                           save=self.save)

            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 20 - (self.lookback - 1) if self.lookback > 1 else 21
            val_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 9
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, 0,
                                           save=self.save)
            assert isinstance(loader, DataSet)

        return

    def test_with_custom_indices_intervals_and_nans(self):
        # if data contains nans and we also have intervals
        if self.output_features is not None:

            data = make_df(40, ['a', 'b', 'c'])
            data['c'].iloc[20:30] = np.nan
            config =  {'input_features':self.input_features,
                      'output_features': self.output_features,
                      'ts_args': {'lookback': self.lookback},
                      'indices': {'training': [1,2,3,4,5,6,7,8,9,10,11,12]},
                      'intervals': [(0, 10), (20, 40)],
                       'train_fraction': 0.7
                      }
            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 8
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 4
            test_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 8
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples, test_examples,
                                           save=self.save)

            assert isinstance(loader, DataSet)

            config['allow_nan_labels'] = 2 if len(self.output_features) == 1 else 1
            tr_examples = 10 - (self.lookback - 1) if self.lookback > 1 else 8
            val_examples = 6 - (self.lookback - 1) if self.lookback > 1 else 4
            test_examples = 16 - (self.lookback - 1) if self.lookback > 1 else 18
            loader = build_and_test_loader(data, config, self.output_features,
                                           tr_examples, val_examples,
                                           test_examples, save=self.save)
            assert isinstance(loader, DataSet)
        return
