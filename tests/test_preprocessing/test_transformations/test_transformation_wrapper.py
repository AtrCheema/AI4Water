
import unittest

import numpy as np

from ai4water.preprocessing import Transformation
from ai4water.datasets import arg_beach
from ai4water.utils.utils import prepare_data
from ai4water.preprocessing.transformations import Transformations


data = arg_beach()


def calculate_manually(xarray_3d, **kwds):
    x1_0 = Transformation(xarray_3d[:, 0].copy(), **kwds).transform().values
    x1_1 = Transformation(xarray_3d[:, 1].copy(), **kwds).transform()
    transformed_x = np.stack([x1_0, x1_1], axis=1)
    return transformed_x


class Test3dData(unittest.TestCase):

    def test_3d_simple_data(self):
        for method in ['minmax', 'log','log10', 'log2',
                       'robust', 'quantile', 'power', 'zscore',
                        'cumsum', 'tan']:

            kwargs = {'method': method}

            if method.startswith('log'):
                kwargs['treat_negatives'] = True
                kwargs['replace_zeros'] = True
                kwargs['replace_zeros_with'] = 1.0

            x = np.arange(-1, 29, dtype='float32').reshape(15, 2)
            x3d, _, _ = prepare_data(x, num_outputs=0, lookback_steps=2)
            x1 = calculate_manually(x3d, **kwargs)

            x2_0 = Transformation(x, **kwargs).transform().values
            x2, _, _ = prepare_data(x2_0, num_outputs=0, lookback_steps=2)
            # print(x1.sum(), x2.sum())

            transformer = Transformations(feature_names=['tide_cm', 'pcp_mm'],
                                          config=[kwargs])
            x3d_ = transformer.fit_transform(x3d.copy())
            _x3d = transformer.inverse_transform(x3d_)

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
            x3d, _, _ = prepare_data(x, num_outputs=0, lookback_steps=2)
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
                kwargs['replace_zeros'] = True
                kwargs['replace_zeros_with'] = 1.0

            x = data[['tide_cm', 'pcp_mm']].values
            x3d, _, _ = prepare_data(x, num_outputs=0, lookback_steps=2)
            x1 = calculate_manually(x3d, **kwargs)

            transformer = Transformations(feature_names=['tide_cm', 'pcp_mm'],
                                          config=[kwargs])
            x3d_ = transformer.fit_transform(x3d.copy())
            _x3d = transformer.inverse_transform(x3d_)

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

if __name__ == "__main__":
    unittest.main()