
import unittest

from ai4water.functional import Model
from ai4water.models import TFT
from ai4water.postprocessing import Interpret


def tft_model(**kwargs):
    num_encoder_steps = 30

    model = TFT(input_shape=(num_encoder_steps, 7),
                hidden_units=8)
    model = Model(model=model,
                  input_features=['et_morton_point_SILO',
                                  'precipitation_AWAP',
                                  'tmax_AWAP',
                                  'tmin_AWAP',
                                  'vprp_AWAP',
                                  'rh_tmax_SILO',
                                  'rh_tmin_SILO'
                                  ],
                  output_features=['streamflow_MLd_inclInfilled'],
                  dataset_args={'st': '19700101', 'en': '20141231',
                                'stations': '224206'},
                  ts_args={'lookback':num_encoder_steps},
                  epochs=2,
                  verbosity=0)
    return model


class TestTFT(unittest.TestCase):

    def test_tft(self):

        model = tft_model()
        model.fit(data='CAMELS_AUS')

        i = Interpret(model, show=False)

        i.interpret_example_tft(0, data='CAMELS_AUS')

        i.interpret_tft(data='CAMELS_AUS')
        return


if __name__ == "__main__":
    unittest.main()