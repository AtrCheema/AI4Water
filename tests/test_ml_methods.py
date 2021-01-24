import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.datasets import load_diabetes, load_breast_cancer
import unittest

import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq import Model

seed = 313
np.random.seed(seed)

data_reg = load_diabetes()
cols = data_reg['feature_names'] + ['target']
df_reg = pd.DataFrame(np.concatenate([data_reg['data'], data_reg['target'].reshape(-1,1)], axis=1), columns=cols)

data_class = load_breast_cancer()
cols = data_class['feature_names'].tolist() + ['target']
df_class = pd.DataFrame(np.concatenate([data_class['data'], data_class['target'].reshape(-1,1)], axis=1), columns=cols)


def run_class_test(method):

    problem = "classification" if method.lower().startswith("class") else "regression"

    if method not in ["STACKINGREGRESSOR", "VOTINGREGRESSOR",  "LOGISTICREGRESSIONCV", # has convergence issues
                      "RIDGE_REGRESSION", "MULTIOUTPUTREGRESSOR", "REGRESSORCHAIN", "REGRESSORMIXIN",
                      # classifications methods
                      "STACKINGCLASSIFIER", "VOTINGCLASSIFIER", "CLASSIFIERCHAIN",
                      "CLASSIFIERMIXIN", "MULTIOUTPUTCLASSIFIER", "CHECK_CLASSIFICATION_TARGETS", "IS_CLASSIFIER"
                      ]:

        print(f"testing {method}")

        model = Model(
            inputs=data_reg['feature_names'] if problem=="regression" else data_class['feature_names'],
            outputs=['target'],
            val_fraction=0.0,
            problem=problem,
            transformation=None,
            data=df_reg if problem=="regression" else data_class,
            model={method: {'iterations': 2} if "CATBOOST" in method else {}},
            verbosity=0)

        return model.fit()

class TestMLMethods(unittest.TestCase):

    def test_XGBOOSTRFCLASSIFIER(self):
        run_class_test("XGBOOSTRFCLASSIFIER")
        return

    def test_XGBOOSTRFREGRESSOR(self):
        run_class_test("XGBOOSTRFREGRESSOR")
        return

    def test_XGBOOSTCLASSIFIER(self):
        run_class_test("XGBOOSTCLASSIFIER")
        return

    def test_XGBOOSTREGRESSOR(self):
        run_class_test("XGBOOSTREGRESSOR")
        return

    def test_KernelRidge(self):
        run_class_test('KernelRidge')
        return

    def test_LassoLars(self):
        run_class_test("LassoLars")
        return

    def test_Lars(self):
        run_class_test("Lars")
        return

    def test_LarsCV(self):
        run_class_test("LarsCV")
        return

    def test_DummyRegressor(self):
        run_class_test("DummyRegressor")
        return

    def test_GaussianProcessRegressor(self):
        run_class_test("GaussianProcessRegressor")
        return

    def test_OrthogonalMatchingPursuit(self):
        run_class_test("OrthogonalMatchingPursuit")
        return

    def test_ElasticNet(self):
        run_class_test("ElasticNet")
        return

    def test_Lasso(self):
        run_class_test("Lasso")
        return

    def test_NuSVR(self):
        run_class_test("NuSVR")
        return

    def test_SVR(self):
        run_class_test("SVR")
        return

    def test_LinearSVR(self):
        run_class_test("LinearSVR")
        return

    def test_OrthogonalMatchingPursuitCV(self):
        run_class_test("OrthogonalMatchingPursuitCV")
        return

    def test_ElasticNetCV(self):
        run_class_test("ElasticNetCV")
        return

    def test_LassoCV(self):
        run_class_test("LassoCV")
        return

    def test_LassoLarsCV(self):
        run_class_test("LassoLarsCV")
        return

    def test_RidgeCV(self):
        run_class_test("RidgeCV")
        return

    def test_EXTRATREEREGRESSOR(self):
        run_class_test("EXTRATREEREGRESSOR")
        return

    def test_EXTRATREECLASSIFIER(self):
        run_class_test("EXTRATREECLASSIFIER")
        return

    def test_DECISIONTREEREGRESSOR(self):
        run_class_test("DECISIONTREEREGRESSOR")
        return

    def test_DECISIONTREECLASSIFIER(self):
        run_class_test("DECISIONTREECLASSIFIER")
        return

    def test_ONECLASSSVM(self):
        run_class_test("ONECLASSSVM")
        return

    def test_MLPREGRESSOR(self):
        run_class_test("MLPREGRESSOR")
        return

    def test_MLPCLASSIFIER(self):
        run_class_test("MLPCLASSIFIER")
        return

    def test_RADIUSNEIGHBORSREGRESSOR(self):
        run_class_test("RADIUSNEIGHBORSREGRESSOR")
        return

    def test_RADIUSNEIGHBORSCLASSIFIER(self):
        run_class_test("RADIUSNEIGHBORSCLASSIFIER")
        return

    def test_KNEIGHBORSREGRESSOR(self):
        run_class_test("KNEIGHBORSREGRESSOR")
        return

    def test_KNEIGHBORSCLASSIFIER(self):
        run_class_test("KNEIGHBORSCLASSIFIER")
        return

    def test_TWEEDIEREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("TWEEDIEREGRESSOR")
        return

    def test_THEILSENREGRESSOR(self):
        run_class_test("THEILSENREGRESSOR")
        return

    def test_SGDREGRESSOR(self):
        run_class_test("SGDREGRESSOR")
        return

    def test_SGDCLASSIFIER(self):
        run_class_test("SGDCLASSIFIER")
        return

    def test_RIDGECLASSIFIERCV(self):
        run_class_test("RIDGECLASSIFIERCV")
        return

    def test_RIDGECLASSIFIER(self):
        run_class_test("RIDGECLASSIFIER")
        return

    def test_RANSACREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("RANSACREGRESSOR")
        return

    def test_POISSONREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("POISSONREGRESSOR")
        return

    def test_PASSIVEAGGRESSIVEREGRESSOR(self):
        run_class_test("PASSIVEAGGRESSIVEREGRESSOR")
        return

    def test_PASSIVEAGGRESSIVECLASSIFIER(self):
        run_class_test("PASSIVEAGGRESSIVECLASSIFIER")
        return

    def test_LOGISTICREGRESSION(self):
        run_class_test("LOGISTICREGRESSION")
        return

    def test_LINEARREGRESSION(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("LINEARREGRESSION")
        return

    def test_HUBERREGRESSOR(self):
        run_class_test("HUBERREGRESSOR")
        return

    def test_GAMMAREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("GAMMAREGRESSOR")
        return

    def test_ARDREGRESSION(self):
        run_class_test("ARDREGRESSION")
        return

    def test_RANDOMFORESTREGRESSOR(self):
        run_class_test("RANDOMFORESTREGRESSOR")
        return

    def test_RANDOMFORESTCLASSIFIER(self):
        run_class_test("RANDOMFORESTCLASSIFIER")
        return

    def test_GRADIENTBOOSTINGREGRESSOR(self):
        run_class_test("GRADIENTBOOSTINGREGRESSOR")
        return

    def test_GRADIENTBOOSTINGCLASSIFIER(self):
        run_class_test("GRADIENTBOOSTINGCLASSIFIER")
        return

    def test_EXTRATREESREGRESSOR(self):
        run_class_test("EXTRATREESREGRESSOR")
        return

    def test_EXTRATREESCLASSIFIER(self):
        run_class_test("EXTRATREESCLASSIFIER")
        return

    def test_BAGGINGREGRESSOR(self):
        run_class_test("BAGGINGREGRESSOR")
        return

    def test_BAGGINGCLASSIFIER(self):
        run_class_test("BAGGINGCLASSIFIER")
        return

    def test_ADABOOSTREGRESSOR(self):
        run_class_test("ADABOOSTREGRESSOR")
        return

    def test_ADABOOSTCLASSIFIER(self):
        run_class_test("ADABOOSTCLASSIFIER")
        return

    def test_LGBMCLASSIFIER(self):
        run_class_test("LGBMCLASSIFIER")
        return

    def test_CATBOOSTCLASSIFIER(self):
        run_class_test("CATBOOSTCLASSIFIER")
        return

    def test_LGBMREGRESSOR(self):
        run_class_test("LGBMREGRESSOR")
        return

    def test_CATBOOSTREGRESSOR(self):
        run_class_test("CATBOOSTREGRESSOR")
        return

    def test_HISTGRADIENTBOOSTINGCLASSIFIER(self):
        run_class_test("HISTGRADIENTBOOSTINGCLASSIFIER")
        return

    def test_HISTGRADIENTBOOSTINGREGRESSOR(self):
        run_class_test("HISTGRADIENTBOOSTINGREGRESSOR")
        return

    def test_ml_random_indices(self):

        model = Model(
            inputs=data_reg['feature_names'],
            outputs=["target"],
            lookback=1,
            batches="2d",
            val_fraction=0.0,
            val_data="same",
            test_fraction=0.3,
            category="ML",
            problem="regression",
            model={"xgboostregressor": {}},
            transformation=None,
            data=df_reg,
            verbosity=0)

        model.fit(indices="random")
        trtt, trp = model.predict(indices=model.train_indices, pref='train')
        t, p = model.predict(indices=model.test_indices, pref='test')
        self.assertGreater(len(t), 1)
        self.assertGreater(len(trtt), 1)
        return


if __name__ == "__main__":
    unittest.main()