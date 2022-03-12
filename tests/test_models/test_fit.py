
import unittest
import time

import tensorflow as tf


if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.functional import Model as FModel
from ai4water.preprocessing import DataSet
from ai4water.datasets import busan_beach, MtropicsLaos

data = busan_beach()
dh = DataSet(data=data, verbosity=0)
x_reg, y_reg = dh.training_data()

laos = MtropicsLaos()
data_cls = laos.make_classification(lookback_steps=2)
dh_cls = DataSet(data=data_cls, verbosity=0)
x_cls, y_cls = dh_cls.training_data()


def _test_fit(_model, model_name, x, y):
    model = _model(model=model_name, verbosity=0)
    model.fit(x, y)

    model.fit(x, y=y)

    model.fit(x=x, y=y)
    return


class TestFit(unittest.TestCase):

    def test_fit(self):
        _test_fit(Model, "RandomForestRegressor", x_reg, y_reg)
        _test_fit(Model, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_fit_functional(self):
        _test_fit(FModel, "RandomForestRegressor", x_reg, y_reg)
        _test_fit(FModel, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_fill_on_all_data(self):
        model = Model(model="RandomForestRegressor", verbosity=0)
        model.fit_on_all_training_data(data=data)
        return

    def test_fit_as_native(self):
        time.sleep(1)
        model = FModel(
            model={"layers": {"Dense": 1}},
            ts_args={'lookback':1},
            input_features=data.columns.tolist()[0:-1],
            output_features=data.columns.tolist()[-1:],
            verbosity=0,
        )

        model.fit(data=data,
                  batch_size=30,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)] )
        assert model.config['batch_size'] == 30

    def test_tf_data(self):
        """when x is tf.data.Dataset"""
        time.sleep(1)
        model = Model(model={"layers": {"Dense": 1}},
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      verbosity=0
                      )
        ds = DataSet(data=data, verbosity=0)
        x,y = ds.training_data()
        tr_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=32)
        tr_ds = tr_ds.repeat(50)
        val_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=32)
        val_ds = val_ds.repeat(50)
        model.fit(tr_ds, epochs=5)
        model.fit(x=tr_ds, epochs=5)
        model.fit(x=tr_ds, validation_data=val_ds, epochs=5)
        model.fit(x=tr_ds, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)], epochs=5)
        return


if __name__ == "__main__":
    unittest.main()