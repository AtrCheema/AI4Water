import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


from ai4water.tf_attributes import tf

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes, load_breast_cancer

seed = 313
np.random.seed(seed)

data_reg = load_diabetes()
cols = data_reg['feature_names'] + ['target']
df_reg = pd.DataFrame(np.concatenate([data_reg['data'], data_reg['target'].reshape(-1,1)], axis=1), columns=cols)

data_class = load_breast_cancer()
cols = data_class['feature_names'].tolist() + ['target']
df_class = pd.DataFrame(np.concatenate([data_class['data'], data_class['target'].reshape(-1,1)], axis=1), columns=cols)


def run_class_test(method):

    mode = "classification" if "class" in method else "regression"

    if method not in ["STACKINGREGRESSOR", "VOTINGREGRESSOR",  "LOGISTICREGRESSIONCV", # has convergence issues
                      "RIDGE_REGRESSION", "MULTIOUTPUTREGRESSOR", "REGRESSORCHAIN", "REGRESSORMIXIN",
                      # classifications methods
                      "STACKINGCLASSIFIER", "VOTINGCLASSIFIER", "CLASSIFIERCHAIN",
                      "CLASSIFIERMIXIN", "MULTIOUTPUTCLASSIFIER", "CHECK_CLASSIFICATION_TARGETS", "IS_CLASSIFIER"
                      ]:

        kwargs = {}
        if "CatBoost" in method:
            kwargs = {'iterations': 2}

        model = Model(
            input_features=data_reg['feature_names'] if mode=="regression" else data_class['feature_names'].tolist(),
            output_features=['target'],
            val_fraction=0.2,
            mode=mode,
            model={method: kwargs},
            verbosity=0)

        return model.fit(data=df_reg if mode=="regression" else df_class)


class TestMLMethods(unittest.TestCase):

    def test_XGBRFCLASSIFIER(self):
        run_class_test("XGBRFClassifier")
        return

    def test_XGBRFREGRESSOR(self):
        run_class_test("XGBRFRegressor")
        return

    def test_XGBCLASSIFIER(self):
        run_class_test("XGBRFClassifier")
        return

    def test_XGBREGRESSOR(self):
        run_class_test("XGBRegressor")
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

    def test_BayesianRidge(self):
        run_class_test("BayesianRidge")
        return

    def test_LassoLarsIC(self):
        run_class_test("LassoLarsIC")
        return

    # def test_TransformedTargetRegressor(self):
    #     run_class_test("TransformedTargetRegressor")
    #     return

    def test_EXTRATREEREGRESSOR(self):
        run_class_test("ExtraTreeRegressor")
        return

    def test_EXTRATREECLASSIFIER(self):
        run_class_test("ExtraTreeClassifier")
        return

    def test_DECISIONTREEREGRESSOR(self):
        run_class_test("DecisionTreeRegressor")
        return

    def test_DECISIONTREECLASSIFIER(self):
        run_class_test("DecisionTreeClassifier")
        return

    def test_ONECLASSSVM(self):
        run_class_test("OneClassSVM")
        return

    def test_MLPREGRESSOR(self):
        run_class_test("MLPRegressor")
        return

    def test_MLPCLASSIFIER(self):
        run_class_test("MLPClassifier")
        return

    def test_RADIUSNEIGHBORSREGRESSOR(self):
        run_class_test("RadiusNeighborsRegressor")
        return

    def test_RADIUSNEIGHBORSCLASSIFIER(self):
        run_class_test("RadiusNeighborsClassifier")
        return

    def test_KNEIGHBORSREGRESSOR(self):
        run_class_test("KNeighborsRegressor")
        return

    def test_KNEIGHBORSCLASSIFIER(self):
        run_class_test("KNeighborsClassifier")
        return

    def test_TWEEDIEREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("TweedieRegressor")
        return

    def test_THEILSENREGRESSOR(self):
        run_class_test("TheilSenRegressor")
        return

    def test_SGDREGRESSOR(self):
        run_class_test("SGDRegressor")
        return

    def test_SGDCLASSIFIER(self):
        run_class_test("SGDClassifier")
        return

    def test_RIDGECLASSIFIERCV(self):
        run_class_test("RidgeClassifierCV")
        return

    def test_RIDGECLASSIFIER(self):
        run_class_test("RidgeClassifier")
        return

    def test_RANSACREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("RANSACRegressor")
        return

    def test_POISSONREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("PoissonRegressor")
        return

    def test_PASSIVEAGGRESSIVEREGRESSOR(self):
        run_class_test("PassiveAggressiveRegressor")
        return

    def test_PASSIVEAGGRESSIVECLASSIFIER(self):
        run_class_test("PassiveAggressiveClassifier")
        return

    def test_LOGISTICREGRESSION(self):
        run_class_test("LogisticRegression")
        return

    def test_LINEARREGRESSION(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("LinearRegression")
        return

    def test_HUBERREGRESSOR(self):
        run_class_test("HuberRegressor")
        return

    def test_GAMMAREGRESSOR(self):
        if int(sklearn.__version__.split('.')[1]) < 23:
            self.assertRaises(ValueError)
        else:
            run_class_test("GammaRegressor")
        return

    def test_ARDREGRESSION(self):
        run_class_test("ARDRegression")
        return

    def test_RANDOMFORESTREGRESSOR(self):
        run_class_test("RandomForestRegressor")
        return

    def test_RANDOMFORESTCLASSIFIER(self):
        run_class_test("RandomForestClassifier")
        return

    def test_GRADIENTBOOSTINGREGRESSOR(self):
        run_class_test("GradientBoostingRegressor")
        return

    def test_GRADIENTBOOSTINGCLASSIFIER(self):
        run_class_test("GradientBoostingClassifier")
        return

    def test_EXTRATREESREGRESSOR(self):
        run_class_test("ExtraTreesRegressor")
        return

    def test_EXTRATREESCLASSIFIER(self):
        run_class_test("ExtraTreesClassifier")
        return

    def test_BAGGINGREGRESSOR(self):
        run_class_test("BaggingRegressor")
        return

    def test_BAGGINGCLASSIFIER(self):
        run_class_test("BaggingClassifier")
        return

    def test_ADABOOSTREGRESSOR(self):
        run_class_test("AdaBoostRegressor")
        return

    def test_ADABOOSTCLASSIFIER(self):
        run_class_test("AdaBoostClassifier")
        return

    def test_LGBMCLASSIFIER(self):
        run_class_test("LGBMClassifier")
        return

    def test_CATBOOSTCLASSIFIER(self):
        run_class_test("CatBoostClassifier")
        return

    def test_LGBMREGRESSOR(self):
        run_class_test("LGBMRegressor")
        return

    def test_CATBOOSTREGRESSOR(self):
        run_class_test("CatBoostRegressor")
        return

    def test_HISTGRADIENTBOOSTINGCLASSIFIER(self):
        run_class_test("HistGradientBoostingClassifier")
        return

    def test_HISTGRADIENTBOOSTINGREGRESSOR(self):
        run_class_test("HistGradientBoostingRegressor")
        return

    def test_ml_random_indices(self):

        model = Model(
            input_features=data_reg['feature_names'],
            output_features=["target"],
            ts_args={'lookback':1},
            batches="2d",
            val_fraction=0.0,
            train_fraction=0.7,
            category="ML",
            mode="regression",
            model={"XGBRegressor": {}},
            split_random=True,
            verbosity=0)

        model.fit(data=df_reg)
        trtt, trp = model.predict(data='training', return_true=True)
        t, p = model.predict(return_true=True)
        self.assertGreater(len(t), 1)
        self.assertGreater(len(trtt), 1)
        return


if __name__ == "__main__":
    unittest.main()