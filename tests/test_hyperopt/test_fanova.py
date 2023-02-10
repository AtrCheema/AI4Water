
import unittest

import numpy as np
import pandas as pd

from ai4water.hyperopt import fANOVA

# These tests were compared against values calculated from
# optuna 2.6.0, numpy:1.19.2, sklearn:1.0.2


class Test2features(unittest.TestCase):

    def test_nuemrical(self):
        x = np.arange(20).reshape(10, 2).astype(float)
        y = np.linspace(1, 30, 10).astype(float)

        f = fANOVA(X=x, Y=y,
                     bounds=[(-2, 20), (-5, 50)],
                     dtypes=["numerical", "numerical"],
                     random_state=313, max_depth=3, n_estimators=1
                     )

        imp = f.feature_importance()
        desired = np.array([0.9597029570966623, 0.04029704290333764])
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired=desired)

        return

    def test_categorical(self):
        x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3'], columns=['a'])
        x['b'] = ['3', '3', '1', '3', '1', '2', '4', '4', '3', '3', '4']
        y = np.linspace(-1., 1.0, 11)
        f = fANOVA(X=x, Y=y, bounds=[None, None], dtypes=['categorical', 'categorical'],
                     random_state=313, max_depth=3, n_estimators=1)

        imp = f.feature_importance()
        desired = [0.8644067796610171, 0.135593220338983]
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired)
        return

    def test_categorical_5trees(self):
        x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3'], columns=['a'])
        x['b'] = ['3', '3', '1', '3', '1', '2', '4', '4', '3', '3', '4']
        y = np.linspace(-1., 2.0, 11)
        f = fANOVA(X=x, Y=y, bounds=[None, None], dtypes=['categorical', 'categorical'],
                     random_state=313, max_depth=5, n_estimators=5)

        imp = f.feature_importance()
        desired = [0.5001132499753341, 0.4998867500246659]
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired)
        return

    def test_mixture(self):

        x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3'], columns=['a'])
        x['b'] = np.arange(100, 100+len(x))
        y = np.linspace(-1., 2.0, 11)
        f = fANOVA(X=x, Y=y, bounds=[None, (10, 150)], dtypes=['categorical', 'numerical'],
                     random_state=313, max_depth=5, n_estimators=5)

        imp = f.feature_importance()
        desired = [0.9646987509852436, 0.03530124901475634]
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired)
        return

    def test_2trees(self):
        x = np.arange(20).reshape(10, 2).astype(float)
        y = np.linspace(1, 30, 10).astype(float)

        f = fANOVA(X=x, Y=y,
                     bounds=[(-2, 20), (-5, 50)],
                     dtypes=["numerical", "numerical"],
                     n_estimators=2,
                     random_state=313, max_depth=3
                     )

        imp = f.feature_importance()
        desired = np.array([0.5226278632146351, 0.477372136785365])
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired=desired)
        return


class Test3features(unittest.TestCase):

    def test_nuemrical(self):
        x = np.arange(30).reshape(10, 3).astype(float)
        y = np.linspace(1, 30, 10).astype(float)

        f = fANOVA(X=x, Y=y,
                     bounds=[(-2.0, 20.0), (-5.0, 50.0), (0.0, 30.0)],
                     dtypes=["numerical", "numerical", "numerical"],
                     random_state=313, max_depth=3, n_estimators=1
                     )

        imp = f.feature_importance()
        desired = np.array([0.835710446690425, 0.16428955330957506, 0.0])
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired=desired)
        return

    def test_categorical(self):
        x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3'], columns=['a'])
        x['b'] = ['3', '3', '1', '3', '1', '2', '4', '4', '3', '3', '4']
        x['c'] = list("ababababbaa")
        y = np.linspace(-1., 2.0, 11)
        f = fANOVA(X=x, Y=y, bounds=[None, None, None],
                     dtypes=['categorical', 'categorical', 'categorical'],
                     random_state=313, max_depth=5, n_estimators=5)

        imp = f.feature_importance()
        desired = [0.5079631976022259, 0.4885331227835112, 0.003503679614262952]
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired=desired)
        return

    def test_mix(self):
        x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3'], columns=['a'])
        x['b'] = list("ababababbaa")
        x['c'] = np.arange(100, 100+len(x))
        y = np.linspace(-1., 3.0, 11)
        f = fANOVA(X=x, Y=y, bounds=[None, None, (99, 120)],
                     dtypes=['categorical', 'categorical', 'numerical'],
                     random_state=313, max_depth=5, n_estimators=5)
        imp = f.feature_importance()
        desired = [0.9073139859509122, 0.06904326302331217, 0.02364275102577564]
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired=desired)
        return


class Test10features(unittest.TestCase):

    def test_nuemrical(self):
        x = np.arange(60).reshape(10, 6).astype(float)
        y = np.linspace(1, 30, 10).astype(float)

        f = fANOVA(X=x, Y=y,
                     bounds=[(-2.0, 20.0), (-5.0, 50.0), (0.0, 30.0),
                             (-2.0, 20.0), (-5.0, 50.0), (0.0, 30.0),
                             #(-2.0, 20.0), (-5.0, 50.0), (0.0, 30.0),
                             #(-2.0, 20.0)
                             ],
                     dtypes=["numerical" for i in range(6)],
                     random_state=313, max_depth=64, n_estimators=1
                     )

        self.assertRaises(AssertionError, f.feature_importance)

        return


    def test_mix(self):
        x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3', '1', '2'], columns=['a'])
        x['b'] = np.linspace(0, 10, len(x))
        x['c'] = np.linspace(10, 11, len(x))
        x['d'] = np.linspace(100, 101, len(x))
        x['e'] = list("ababababbaaab")
        x['f'] = list("ababababbacab")
        x['g'] = list("abababdbbacab")
        x['h'] = np.linspace(0, 0.5, len(x))

        y = np.linspace(1000.0, 2000.0, len(x))
        f = fANOVA(X=x, Y=y,
                     bounds=[None, (-1, 15), (5, 20), (90, 110), None, None, None, (-1, 1.0)],
                     dtypes=['categorical', 'numerical', 'numerical', 'numerical',
                             'categorical', 'categorical', 'categorical', 'numerical'],
                     random_state=313, max_depth=64, n_estimators=64)

        imp = f.feature_importance()
        desired = [0.33392323604064855, 0.24324300895296924, 0.2140487927063801,
                   0.19457767969807768, 0.00641277223726712, 0.00503955923737171,
                   0.0019451190410500819, 0.0008098320862354798]
        np.testing.assert_almost_equal(np.array(list(imp.values())), desired)
        return


if __name__ == "__main__":
    unittest.main()