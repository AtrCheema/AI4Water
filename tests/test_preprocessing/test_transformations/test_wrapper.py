
import unittest

import numpy as np
import pandas as pd

from ai4water import Model
from ai4water.datasets import busan_beach
from ai4water.utils.utils import prepare_data
from ai4water.preprocessing import Transformation
from ai4water.preprocessing.transformations import Transformations
from ai4water.preprocessing.transformations.utils import SP_METHODS


data = busan_beach()


def calculate_manually(xarray_3d, **kwds):
    x1_0 = Transformation(**kwds).fit_transform(xarray_3d[:, 0].copy()).values
    x1_1 = Transformation(**kwds).fit_transform(xarray_3d[:, 1].copy())
    transformed_x = np.stack([x1_0, x1_1], axis=1)
    return transformed_x


class Test3dData(unittest.TestCase):

    def test_3d_simple_data(self):
        for method in ['minmax', 'log','log10', 'log2',
                       'robust', 'quantile', 'power', 'zscore',
                       'cumsum', 'tan', "scale", "center",
                       "box-cox", "sqrt"
                       ]:

            kwargs = {'method': method}

            if method.startswith('log'):
                kwargs['treat_negatives'] = True
            elif method == "box-cox":
                kwargs['treat_negatives'] = True
                kwargs['replace_zeros'] = True
            elif method == "sqrt":
                kwargs['treat_negatives'] = True

            x = np.arange(1, 31, dtype='float32').reshape(15, 2)
            x3d, _, _ = prepare_data(x, num_outputs=0, lookback=2)
            x1 = calculate_manually(x3d, **kwargs)

            x2_0 = Transformation(**kwargs).fit_transform(x).values
            x2, _, _ = prepare_data(x2_0, num_outputs=0, lookback=2)
            # print(x1.sum(), x2.sum())

            transformer = Transformations(feature_names=['tide_cm', 'pcp_mm'],
                                          config=[kwargs])
            x3d_ = transformer.fit_transform(x3d.copy())
            _x3d = transformer.inverse_transform(x3d_.copy())

            if method not in ['log', 'log10', 'log2', 'tan', 'cumsum']:
                # don't do inverse checking
                np.testing.assert_array_almost_equal(x3d, _x3d, decimal=5)

            # manually transforming one by one and by using Transformations class
            # should result same
            np.testing.assert_array_almost_equal(x3d_, x1,
                                                 err_msg=f"{method} failing")
        return

    def test_3d_with_log(self):
        for method in ['log']:
            # log transformations without -ves or 0s
            x = np.arange(10, 30, dtype='float32').reshape(10, 2)

            kwargs = {'method': method}
            x3d, _, _ = prepare_data(x, num_outputs=0, lookback=2)
            x1 = calculate_manually(x3d, **kwargs)

            transformer = Transformations(feature_names=['tide_cm', 'pcp_mm'],
                                          config=[kwargs])
            x3d_ = transformer.fit_transform(x3d.copy())
            _x3d = transformer.inverse_transform(x3d_)

            np.testing.assert_array_almost_equal(x3d, _x3d, decimal=5)

            np.testing.assert_array_almost_equal(x3d_, x1,
                                                 err_msg=f"{method} failing")
        return

    def test_3d_with_real_data(self):
        for method in ['minmax', 'log', 'power', 'robust', 'quantile', 'zscore']:

            kwargs = {'method': method}
            if method.startswith('log'):
                kwargs['treat_negatives'] = True
#                kwargs['replace_zeros'] = True
            elif method == "box-cox":
                kwargs['treat_negatives'] = True
                kwargs['replace_zeros'] = True
            elif method == "sqrt":
                kwargs['treat_negatives'] = True

            x = data[['tide_cm', 'pcp_mm']].values
            x3d, _, _ = prepare_data(x, num_outputs=0, lookback=2)
            x1 = calculate_manually(x3d.copy(), **kwargs)

            transformer = Transformations(feature_names=['tide_cm', 'pcp_mm'],
                                          config=[kwargs])
            x3d_ = transformer.fit_transform(x3d.copy())
            _x3d = transformer.inverse_transform(x3d_.copy())

            if method not in ['log', 'log10', 'log2', 'tan', 'cumsum']:
                np.testing.assert_array_almost_equal(x3d, _x3d, decimal=5)

            np.testing.assert_array_almost_equal(x3d_, x1,
                                                 err_msg=f"{method} failing")
        return


class TestSingleSource(unittest.TestCase):

    def test_1d(self):
        x = np.arange(100)
        transformer = Transformations(['a'])
        x_ = transformer.fit_transform(x)
        np.testing.assert_array_almost_equal(x, x_)
        return

    def test_2d(self):
        x = np.arange(100).reshape(50, 2)
        transformer = Transformations(['a', 'b'])
        x_ = transformer.fit_transform(x)
        np.testing.assert_array_almost_equal(x, x_)
        return

    def test_single_transformation(self):
        x = np.arange(50).reshape(25, 2)
        transformer = Transformations(['a', 'b'], config="minmax")
        x_ = transformer.fit_transform(x)
        _x = transformer.inverse_transform(x_)
        np.testing.assert_array_almost_equal(x, _x)
        return

    def test_single_but_complex_transformation(self):
        x = np.arange(50).reshape(25, 2)
        transformer = Transformations(['a', 'b'], config={'method': 'minmax', 'features': ['a']})
        x_ = transformer.fit_transform(x)
        np.testing.assert_array_almost_equal(x[:, 1], x_[:, 1])
        _x = transformer.inverse_transform(x_)
        np.testing.assert_array_almost_equal(x, _x)
        return

    def test_multiple_transformations(self):
        x = np.arange(50).reshape(25, 2)
        transformer = Transformations(['a', 'b'], config=['minmax', 'zscore'])
        x_ = transformer.fit_transform(x)
        _x = transformer.inverse_transform(x_)
        np.testing.assert_array_almost_equal(x, _x)
        return

    def test_multiple_transformations_but_selected_features(self):
        x = np.arange(50).reshape(25, 2)
        transformer = Transformations(['a', 'b'], config=[
            {'method': 'minmax', 'features': ['a']},
            {'method': 'zscore', 'features': ['a']}
        ])
        x_ = transformer.fit_transform(x)
        np.testing.assert_array_almost_equal(x[:, 1], x_[:, 1])
        _x = transformer.inverse_transform(x_)
        np.testing.assert_array_almost_equal(x, _x)
        return


class TestMultipleSources(unittest.TestCase):

    def test_list_2d(self):
        transformer = Transformations([['a', 'b'], ['a', 'b']],
                                      config=['minmax', 'zscore'])
        x1 = np.arange(50).reshape(25, 2)
        x2 = np.arange(50, 100).reshape(25, 2)
        x1_ = transformer.fit_transform([x1, x2])
        _x1 = transformer.inverse_transform(x1_)

        for i, j in zip([x1, x2], _x1):
            np.testing.assert_array_almost_equal(i, j)
        return

    def test_list_3d(self):
        # list with 2 3d arrays
        transformer = Transformations([['a', 'b'], ['a', 'b']],
                                      config=['minmax', 'zscore'])
        x1 = np.arange(100).reshape(10, 5, 2)
        x2 = np.arange(100, 200).reshape(10, 5, 2)
        x1_ = transformer.fit_transform([x1, x2])
        _x1 = transformer.inverse_transform(x1_)

        for i, j in zip([x1, x2], _x1):
            np.testing.assert_array_almost_equal(i, j)

        return

    def test_list_2d_3d(self):
        transformer = Transformations([['a', 'b'], ['a', 'b']],
                                      config='minmax')
        x1 = np.arange(50).reshape(25, 2)
        x2 = np.arange(100).reshape(10, 5, 2)
        x1_ = transformer.fit_transform([x1, x2])
        _x1 = transformer.inverse_transform(x1_)

        for i, j in zip([x1, x2], _x1):
            np.testing.assert_array_almost_equal(i, j)
        return

    def test_dict_2d(self):
        transformer = Transformations({'x1': ['a', 'b'], 'x2': ['a', 'b']},
                                      config={'x1': ['minmax', 'zscore'],
                                              'x2': [{'method': 'log', 'features': ['a', 'b']},
                                                     {'method': 'robust', 'features': ['a', 'b']}]
                                              })
        x1 = np.arange(20).reshape(10, 2)
        x2 = np.arange(100, 120).reshape(10, 2)
        x = {'x1': x1, 'x2': x2}
        x_ = transformer.fit_transform(x)
        _x = transformer.inverse_transform(x_)

        for i, j in zip(x.values(), _x.values()):
            np.testing.assert_array_almost_equal(i, j)
        return

    def test_dict_3d(self):
        transformer = Transformations({'x1': ['a', 'b'], 'x2': ['a', 'b']},
                                      config={'x1': 'minmax', 'x2': 'zscore'})
        x1 = np.arange(100).reshape(10, 5, 2)
        x2 = np.arange(100, 200).reshape(10, 5, 2)
        x = {'x1': x1, 'x2': x2}
        x_ = transformer.fit_transform(x)
        _x = transformer.inverse_transform(x_)

        for i, j in zip(x.values(), _x.values()):
            np.testing.assert_array_almost_equal(i, j)
        return

    def test_dict_2d_3d(self):
        transformer = Transformations({'x1': ['a', 'b'], 'x2': ['a', 'b']},
                                      config='minmax')
        x1 = np.arange(20).reshape(10, 2)
        x2 = np.arange(100, 200).reshape(10, 5, 2)
        x = {'x1': x1, 'x2': x2}
        x_ = transformer.fit_transform(x)
        _x = transformer.inverse_transform(x_)

        for i, j in zip(x.values(), _x.values()):
            np.testing.assert_array_almost_equal(i, j)
        return

    def test_from_config(self):
        for method in ["minmax", "log", "sqrt", "box-cox", "scale", "center", "robust"]:

            config = {'method': method, 'treat_negatives': True, 'replace_zeros': True}

            x = np.random.randint(2, 32, (10, 3))
            t = Transformations(['a', 'b', 'c'], config)
            x_ = t.fit_transform(x.copy())
            conf = t.config()

            t2 = Transformations.from_config(conf)
            _x = t2.inverse_transform(x_.copy())
            np.testing.assert_array_almost_equal(x, _x)
        return

    def test_model_predict(self):
        # first use transformation and inverse_transformation from fit/predict and
        # then compare with manually doing the same
        methods = ["minmax", "box-cox", "robust", "power",
                   "center", "zscore", "scale", "quantile"]

        data1 = np.random.random((10, 3))
        df = pd.DataFrame(data1, columns=['a', 'b', 'c'])
        for x_trans in methods:

            for y_trans in methods:

                model = Model(model="RandomForestRegressor",
                              x_transformation = x_trans,
                              y_transformation = y_trans,
                              verbosity=0,
                              val_fraction=0.0, train_fraction=1.0)

                model.fit(data=df)
                t,p = model.predict(data='training',
                                    return_true=True,
                                    process_results=False)

                x = data1[:, 0:2]
                y = data1[:, -1]
                minmax_t = Transformation(x_trans)
                x_ = minmax_t.fit_transform(x)
                log_t = Transformation(y_trans)
                y_ = log_t.fit_transform(y)
                data_ = np.column_stack((x_, y_))
                df2 = pd.DataFrame(data_, columns=['a', 'b', 'c'])

                model2 = Model(model="RandomForestRegressor",
                               verbosity=0,
                               val_fraction=0.0, train_fraction=1.0,)
                model2.fit(data=df2)
                t2, p2 = model2.predict(data='training', return_true=True, process_results=False)
                _t2 = log_t.inverse_transform(t2)
                _p2 = log_t.inverse_transform(p2)
                np.testing.assert_array_almost_equal(t, _t2)
                np.testing.assert_array_almost_equal(p, _p2)
        return


class TestWithoutFit(unittest.TestCase):

    def make_config(self, method, to):

        if to=="dict":
            return {'method': method, 'features': ['a'],
             'treat_negatives': True, 'replace_zeros': True}
        elif to=="list":
            return [{'method': method, 'features': ['a'],
                                         'treat_negatives': True, 'replace_zeros': True}]
        else:
            raise ValueError

    def test_without_fit(self):
        for method in SP_METHODS:
            y = np.random.random(100)
            tr = Transformations(['a'], method)
            y_ = tr.fit_transform(y)
            _y = tr.inverse_transform(y_)

            tr1 = Transformations(['a'], method)
            _y1 = tr1.inverse_transform_without_fit(y_)

            assert np.allclose(_y, _y1)
        return

    def test_without_fit_dict(self):
        for method in SP_METHODS:
            y = np.random.randint(-2, 10, 100)
            tr = Transformations(['a'], self.make_config(method, "dict"))
            y_ = tr.fit_transform(y)
            _y = tr.inverse_transform(y_)

            tr1 = Transformations(['a'], self.make_config(method, "dict"))
            _y1 = tr1.inverse_transform_without_fit(y_)

            print(np.allclose(_y, _y1), method)
        return

    def test_without_fit_list(self):
        for method in SP_METHODS:
            y = np.random.randint(-2, 10, 100)
            tr = Transformations(['a'], self.make_config(method, "list"))
            y_ = tr.fit_transform(y)
            _y = tr.inverse_transform(y_)

            tr1 = Transformations(['a'], self.make_config(method, "list"))
            _y1 = tr1.inverse_transform_without_fit(y_)

            print(np.allclose(_y, _y1), method)
        return


if __name__ == "__main__":
    unittest.main()