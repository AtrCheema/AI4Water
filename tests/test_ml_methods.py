import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_diabetes, load_breast_cancer
import unittest

import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq.utils import make_model
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
        config = make_model(
            inputs=data_reg['feature_names'] if problem=="regression" else data_class['feature_names'],
            outputs=['target'],
            lookback=1,
            batches="2d",
            val_fraction=0.0,
            ml_model=method,
            problem=problem,
            transformation=None
        )


        model = Model(config,
                      data=df_reg if problem=="regression" else data_class,
                      verbosity=0)

        return model.train()

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
        run_class_test("RANSACREGRESSOR")
        return

    def test_POISSONREGRESSOR(self):
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
        run_class_test("LINEARREGRESSION")
        return

    def test_HUBERREGRESSOR(self):
        run_class_test("HUBERREGRESSOR")
        return

    def test_GAMMAREGRESSOR(self):
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
        config = make_model(
            inputs=data_reg['feature_names'],
            outputs=["target"],
            lookback=1,
            batches="2d",
            val_fraction=0.0,
            val_data="same",
            test_fraction=0.3,
            category="ML",
            problem="regression",
            ml_model="xgboostregressor",
            # ml_model_args= {"max_dep":1000},
            transformation=None
        )
        model = Model(config,
                      data=df_reg,
                      verbosity=0)

        model.train(indices="random")
        trtt, trp = model.predict(indices=model.train_indices, pref='train')
        t, p = model.predict(indices=model.test_indices, pref='test')
        self.assertGreater(len(t), 1)
        self.assertGreater(len(trtt), 1)
        return


if __name__ == "__main__":
    unittest.main()