import time
import unittest
import os

import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(ai4_dir)


import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.functional import Model as FModel
from ai4water.datasets import busan_beach
from ai4water.datasets import MtropicsLaos


laos = MtropicsLaos()
cls_data = laos.make_classification(lookback_steps=1)


def test_evaluation(model, _data):

    model.evaluate_on_training_data(data=_data)
    model.training_data(data=_data)

    model.evaluate_on_validation_data(data=_data)
    val_data = model.validation_data(data=_data)

    model.evaluate_on_test_data(data=_data)
    test_data = model.test_data(data=_data)
    if not isinstance(test_data, tf.data.Dataset):
        test_x, test_y = test_data

    if not isinstance(val_data, tf.data.Dataset):
        val_x, y = val_data
        assert test_x[0].shape == val_x[0].shape

    return


def test_prediction(model, df):

    t, p =model.predict_on_training_data(data=df, return_true=True, metrics="all")
    assert t.size == p.size
    t,p = model.predict_on_validation_data(data=df, return_true=True, metrics="all")
    assert t.size == p.size
    t,p = model.predict_on_test_data(data=df, return_true=True, metrics="all")
    assert t.size == p.size

    return


def make_dl_model(n_classes, activation='softmax'):

    return {'layers': {
            'Dense_0': 10,
            'Flatten': {},
            'Dense_1': n_classes,
            'Activation': activation}}


def build_and_run_class_problem(n_classes,
                                loss,
                                model=None,
                                is_multilabel=False,
                                ):

    input_features = [f'input_{n}' for n in range(10)]

    if is_multilabel:
        outputs = [f'target_{n}' for n in range(n_classes)]
        X, y = make_multilabel_classification(n_samples=100,
                                              n_features=len(input_features),
                                              n_classes=n_classes,
                                              n_labels=2, random_state=0)
        y = y.reshape(-1, n_classes)

    else:
        outputs = ['target']
        X, y = make_classification(n_samples=100, n_features=len(input_features),
                                   n_informative=n_classes, n_classes=n_classes,
                               random_state=1)
        y = y.reshape(-1, 1)

    df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=input_features + outputs)

    model = Model(
        model=model,
        input_features=input_features,
        loss=loss,
        output_features=outputs,
        verbosity=0,
    )
    model.fit(data=df)
    test_evaluation(model, df)

    test_prediction(model, df)

    assert model.mode == 'classification'
    assert len(model.classes_) == n_classes, len(model.classes_)
    assert model.num_classes_ == n_classes
    return model


class TestClassifications(unittest.TestCase):

    def test_ml_cls_model(self):
        # FModel because tensorflow sucks
        model = FModel(model="RandomForestClassifier", verbosity=0)
        model.fit(data=cls_data)
        proba = model.predict_proba()
        log_proba = model.predict_log_proba()
        assert proba.shape[1] == 2
        assert log_proba.shape[1] == 2
        return

    def test_binary_classification(self):

        model = build_and_run_class_problem(
            2,
            'binary_crossentropy',
            model=make_dl_model(2, "sigmoid"))

        assert model.is_binary_
        assert not model.is_multiclass_
        assert not model.is_multilabel_

        return

    def test_binary_classification_softmax(self):

        model = build_and_run_class_problem(2,
                                            'binary_crossentropy',
                                            model=make_dl_model(2))

        assert model.is_binary_
        assert not model.is_multiclass_
        assert not model.is_multilabel_

        return

    def test_binary_cls_ml(self):

        for algo in ["RandomForestClassifier",
                      "XGBClassifier",
                      "CatBoostClassifier",
                      "LGBMClassifier"]:

            model = build_and_run_class_problem(
                2,
                'binary_crossentropy',
                model=algo)
            assert model.is_binary_
            assert not model.is_multiclass_
            assert not model.is_multilabel_

        return

    def test_multicls_cls_ml(self):

        for algo in ["RandomForestClassifier",
                     "XGBClassifier",
                     "CatBoostClassifier",
                     "LGBMClassifier"]:
            model = build_and_run_class_problem(5,
                                                'binary_crossentropy',
                                                model=algo)
            assert model.is_multiclass_
            assert not model.is_binary_
            assert not model.is_multilabel_

        return

    def test_multiclass_classification(self):
        time.sleep(1)
        model = build_and_run_class_problem(3,
                                            'binary_crossentropy',
                                            model=make_dl_model(3))

        assert not model.is_binary_
        assert model.is_multiclass_
        assert not model.is_multilabel_

        return

    def test_multilabel_classification(self):

        model = build_and_run_class_problem(5,
                                            'binary_crossentropy',
                                            is_multilabel=True,
                                            model=make_dl_model(5)
                                            )

        assert not model.is_binary_
        assert not model.is_multiclass_
        assert model.is_multilabel_

        return

    def test_multilabel_classification_with_categorical(self):

        model = build_and_run_class_problem(5,
                                            'categorical_crossentropy',
                                            is_multilabel=True,
                                            model=make_dl_model(5)
                                            )

        assert not model.is_binary_
        assert not model.is_multiclass_
        assert model.is_multilabel_

        return

    def test_multilabel_classification_with_binary_sigmoid(self):

        model = build_and_run_class_problem(5,
                                            'binary_crossentropy',
                                            model=make_dl_model(5, "sigmoid"),
                                            is_multilabel=True)

        assert not model.is_binary_
        assert not model.is_multiclass_
        assert model.is_multilabel_

        return

    def test_multilabel_classification_with_categorical_sigmoid(self):

        model = build_and_run_class_problem(5,
                                            'categorical_crossentropy',
                                            make_dl_model(5, "sigmoid"),
                                            is_multilabel=True)

        assert not model.is_binary_
        assert not model.is_multiclass_
        assert model.is_multilabel_

        return

    def test_basic_multi_output(self):
        time.sleep(1)
        model = Model(
            model= {'layers': {'LSTM': {'units': 32},
                               'Dense': {'units': 2},
                               }},
            ts_args={'lookback':5},
            input_features=busan_beach().columns.tolist()[0:-1],
            output_features = ['blaTEM_coppml', 'tetx_coppml'],
            verbosity=0,
            train_fraction=0.8,
            shuffle=False
        )

        data = busan_beach(target=['blaTEM_coppml', 'tetx_coppml'])
        model.fit(data=data)
        t,p = model.predict_on_test_data(data=data, return_true=True)

        assert np.allclose(t[3:5, 1].reshape(-1,).tolist(), [14976057.52, 3279413.328])

        for out in model.output_features:
            assert out in os.listdir(model.path)
        return


if __name__ == "__main__":
    unittest.main()