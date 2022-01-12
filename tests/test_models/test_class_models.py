import time
import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
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


from ai4water.datasets import busan_beach


def test_evaluation(model):

    model.evaluate(data='training')
    train_x, train_y = model.training_data()

    model.evaluate(data='validation')
    val_data = model.validation_data()

    model.evaluate(data='test')
    test_data = model.test_data()
    if not isinstance(test_data, tf.data.Dataset):
        test_x, test_y = test_data

    if model.config['val_data'] == 'same' and not isinstance(val_data, tf.data.Dataset):
        val_x, y = val_data
        assert test_x[0].shape == val_x[0].shape

    return


def build_and_run_class_problem(n_classes, loss, is_multilabel=False, activation='softmax'):

    input_features = [f'input_{n}' for n in range(10)]

    if is_multilabel:
        outputs = [f'target_{n}' for n in range(n_classes)]
        X, y = make_multilabel_classification(n_samples=100, n_features=len(input_features), n_classes=n_classes,
                                              n_labels=2, random_state=0)
        y = y.reshape(-1, n_classes)

    else:
        outputs = ['target']
        X, y = make_classification(n_samples=100, n_features=len(input_features), n_informative=n_classes, n_classes=n_classes,
                               random_state=1)
        y = y.reshape(-1, 1)

    df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=input_features + outputs)

    model = Model(
        model={'layers': {
            'Dense_0': 10,
            'Flatten': {},
            'Dense_1': n_classes,
            'Activation': activation}},
        input_features=input_features,
        loss=loss,
        output_features=outputs,
        verbosity=0,
    )
    model.fit(data=df)
    test_evaluation(model)

    assert model.mode == 'classification'
    assert len(model.classes) == n_classes
    assert model.num_classes == n_classes
    return model


class TestClassifications(unittest.TestCase):

    def test_binary_classification(self):

        model = build_and_run_class_problem(2, 'binary_crossentropy')

        assert model.is_binary
        assert not model.is_multiclass
        assert not model.is_multilabel

        return

    def test_multiclass_classification(self):

        model = build_and_run_class_problem(3, 'binary_crossentropy')

        assert not model.is_binary
        assert model.is_multiclass
        assert not model.is_multilabel

        return

    def test_multilabel_classification(self):

        model = build_and_run_class_problem(5, 'binary_crossentropy', is_multilabel=True)

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

    def test_multilabel_classification_with_categorical(self):

        model = build_and_run_class_problem(5, 'categorical_crossentropy', is_multilabel=True)

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

    def test_multilabel_classification_with_binary_sigmoid(self):

        model = build_and_run_class_problem(5, 'binary_crossentropy', is_multilabel=True, activation='sigmoid')

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

    def test_multilabel_classification_with_categorical_sigmoid(self):

        model = build_and_run_class_problem(5, 'categorical_crossentropy', is_multilabel=True, activation='sigmoid')

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

    def test_basic_multi_output(self):
        time.sleep(1)
        model = Model(
            model= {'layers': {'LSTM': {'units': 32},
                               'Dense': {'units': 2},
                               }},
            lookback=5,
            input_features=busan_beach().columns.tolist()[0:-1],
            output_features = ['blaTEM_coppml', 'tetx_coppml'],
            verbosity=0
        )

        model.fit(data=busan_beach(target=['blaTEM_coppml', 'tetx_coppml']))
        t,p = model.predict(data='test', return_true=True)

        assert np.allclose(t[0:2, 1].reshape(-1,).tolist(), [14976057.52, 3279413.328])

        for out in model.output_features:
            assert out in os.listdir(model.path)


if __name__ == "__main__":
    unittest.main()