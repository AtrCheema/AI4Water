
import unittest

import numpy as np


from ai4water import Model
from ai4water.datasets import busan_beach
from ai4water.preprocessing import DataSet


beach_data = busan_beach()
inputs = beach_data.columns.tolist()[0:-1]
outputs = beach_data.columns.tolist()[-1:]


class TestEvaluate(unittest.TestCase):
    model = Model(
        model={"layers": {"Dense": 1}},
        input_features=inputs,
        output_features=outputs,
        ts_args={'lookback':1},
        monitor="nse",
        verbosity=0,
    )
    def test_basic(self):
        eval_scores = self.model.evaluate(data=beach_data)
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_basic_on_training_data(self):
        self. model.fit(data=beach_data, epochs=1)
        eval_scores = self.model.evaluate_on_training_data(data=beach_data)
        assert isinstance(eval_scores, list) and len(eval_scores) == 2

        eval_scores = self.model.evaluate_on_training_data(data=beach_data)
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_basic_on_validation_data(self):
        self.model.fit(data=beach_data, epochs=1)
        eval_scores = self.model.evaluate_on_validation_data(data=beach_data)
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        eval_scores = self.model.evaluate_on_validation_data(data=beach_data)
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_basic_on_all(self):
        self.model.fit(data=beach_data, epochs=1)
        eval_scores = self.model.evaluate_on_all_data(data=beach_data)
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_basic_with_metrics(self):
        # basic example with metrics
        eval_scores = self.model.evaluate_on_test_data(data=beach_data, metrics="kge")
        assert isinstance(eval_scores, float)

    def test_basic_with_metric_groups(self):
        # basic example with metrics
        eval_scores = self.model.evaluate_on_test_data(data=beach_data, metrics="hydro_metrics")
        assert isinstance(eval_scores, dict)
        return

    def test_custom_xy(self):
        eval_scores = self.model.evaluate(np.random.random((10, 13)), np.random.random((10, 1)))
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_custom_xy0(self):
        # only y as keyword
        eval_scores = self.model.evaluate(np.random.random((10, 13)), y=np.random.random((10, 1)))
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_custom_xy_as_keyword_args(self):
        eval_scores = self.model.evaluate(x=np.random.random((10, 13)), y=np.random.random((10, 1)))
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_custom_xy_with_metrics(self):
        # custom data with metrics
        pbias = self.model.evaluate(x=np.random.random((10, 13)), y=np.random.random((10, 1)),
                                    metrics='pbias')
        assert isinstance(pbias, float)
        return

    def test_custom_xy_with_metric_groups(self):
        # custom data with group of metrics
        hydro = self.model.evaluate(x=np.random.random((10, 13)), y=np.random.random((10, 1)),
                                    metrics='hydro_metrics')
        assert isinstance(hydro, dict)
        return

    def test_tf_data(self):
        """when x is tf.data.Dataset"""
        import tensorflow as tf
        x,y = DataSet(data=beach_data, verbosity=0).training_data()
        tr_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=32)
        self.model.evaluate(tr_ds)
        self.model.evaluate(x=tr_ds)
        return

    def test_2d_data(self):
        """when x is 2D"""
        import tensorflow as tf
        tf.keras.backend.clear_session()
        _model = Model(model={"layers": {
            "Input": {"shape": (10, 10)},
            "Flatten": {},
            "Dense": 12}},
                      verbosity=0
                      )
        x = np.random.random((100, 10, 10))
        y = np.random.random((100, 12))
        _model.evaluate(x, y)
        _model.evaluate(x=x, y=y)
        #_model.evaluate(x=x, y=y, metrics='r2')
        tf.keras.backend.clear_session()
        return


class TestMinimal(unittest.TestCase):

    model = Model(model="RandomForestRegressor",
                  input_features=inputs,
                  output_features = outputs,
                  verbosity=0,
                  )

    def test_basic(self):
        y = np.random.random(10).reshape(-1,1)
        self.model.fit(np.random.random((10, 13)), y)
        self.model.evaluate(np.random.random((10, 13)), y)
        return


if __name__ == "__main__":

    unittest.main()