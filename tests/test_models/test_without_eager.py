import os
import unittest
import site  # so that ai4water directory is in path
import sys
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from ai4water.datasets import busan_beach, load_nasdaq
from ai4water import InputAttentionModel, DualAttentionModel

arg_busan = busan_beach()
arg_input_features = arg_busan.columns.tolist()[0:-1]
arg_output_features = arg_busan.columns.tolist()[-1:]

nasdaq= load_nasdaq(inputs=['AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK'])
nasdaq_input_features = nasdaq.columns.tolist()[0:-1]
nasdaq_output_features = nasdaq.columns.tolist()[-1:]


def make_and_run(input_model, data, _layers=None, lookback=12,
                 batch_size=64, epochs=3, **kwargs):

    model = input_model(
        verbosity=0,
        batch_size=batch_size,
        ts_args = {'lookback': lookback},
        lr=0.001,
        epochs=epochs,
        split_random=True,
        **kwargs
    )

    _ = model.fit(data=data)

    _ = model.predict(data='training')
    _ = model.predict(data='validation')
    _ = model.evaluate(data='training')
    pred_y = model.predict()

    # user defined data
    x,y = model.training_data(data=data)
    model.fit(x=x,y=y, epochs=1)
    model.fit_on_all_training_data(data=data, epochs=1)
    _ = model.predict(x=x,y=y)
    model.predict_on_validation_data(data=data)
    model.predict_on_all_data(data=data)

    return pred_y

class TestModels(unittest.TestCase):

    # InputAttention based model does not conform reproducibility
    # so just testing that it runs.

    def test_InputAttentionModel(self):

        prediction = make_and_run(InputAttentionModel,
                                  data=arg_busan,
                                  input_features=arg_input_features,
                                  output_features=arg_output_features)
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return

    def test_IA_predict_without_fit(self):

        model = InputAttentionModel(
            input_features=arg_input_features,
            output_features=arg_output_features,
            ts_args={"lookback": 15},
            verbosity=0,
        )
        model.predict(data=arg_busan)
        return

    def test_IA_with_transformation(self):

        prediction = make_and_run(InputAttentionModel,
                                  data=arg_busan,
                                  input_features=arg_input_features,
                                  x_transformation='minmax',
                                  y_transformation='zscore',
                                  output_features=arg_output_features)
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return

    def test_DA_with_transformation(self):

        prediction = make_and_run(DualAttentionModel,
                                  data=arg_busan,
                                  teacher_forcing=False,
                                  batch_size=8,
                                  drop_remainder=True,
                                  input_features=arg_input_features,
                                  x_transformation='minmax',
                                  y_transformation='zscore',
                                  output_features=arg_output_features)
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return

    # def test_InputAttentionModel_with_drop_remainder(self):
    #
    #     prediction = make_and_run(InputAttentionModel, drop_remainder=True)
    #     self.assertGreater(float(prediction[0].sum()), 0.0)

    def test_DualAttentionModel(self):
        # DualAttentionModel based model

        prediction = make_and_run(
            DualAttentionModel,
            data=nasdaq,
            input_features=nasdaq_input_features,
            output_features=nasdaq_output_features
        )

        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return

    def test_da_without_prev_y(self):
        prediction = make_and_run(
            DualAttentionModel,
            data=arg_busan,
            teacher_forcing=False,
            batch_size=8,
            drop_remainder=True,
            input_features=arg_input_features,
            output_features=arg_output_features
        )
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return


if __name__ == "__main__":
    unittest.main()