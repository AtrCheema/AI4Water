import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np

from ai4water import Model

from ai4water.datasets import arg_beach

data = arg_beach()


class TestFrontPage(unittest.TestCase):

    def test_example1(self):
        model = Model(
                model = {'layers': {"LSTM": 64,
                                    "Dense": 1}},
                data = data,
                input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm'],   # columns in csv file to be used as input
                output_features = ['tetx_coppml'],     # columns in csv file to be used as output
                lookback = 12,
            verbosity=0
        )

        model.fit()

        model.predict()

        import tensorflow as tf
        assert isinstance(model, tf.keras.Model)  # True
        return

    def test_example2(self):
        batch_size = 16
        lookback = 15
        inputs = ['dummy1', 'dummy2', 'dummy3', 'dumm4', 'dummy5']  # just dummy names for plotting and saving results.
        outputs=['DummyTarget']

        model = Model(
                    model = {'layers': {"LSTM": 64,
                                        'Dense': 1}},
                    batch_size=batch_size,
                    lookback=lookback,
                    input_features=inputs,
                    output_features=outputs,
                    lr=0.001,
            verbosity=0
                      )
        x = np.random.random((batch_size*10, lookback, len(inputs)))
        y = np.random.random((batch_size*10, len(outputs)))

        model.fit(x=x,y=y)
        return


    def test_example3(self):

        model = Model(
            input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm'],
            # columns in csv file to be used as input
            output_features=['tetx_coppml'],
            lookback=1,
            val_fraction=0.0,
            #  any regressor from https://scikit-learn.org/stable/modules/classes.html
            model={"randomforestregressor": {"n_estimators": 1000}},
            # set any of regressor's parameters. e.g. for RandomForestRegressor above used,
            # some of the paramters are https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
            data=data,
            verbosity=0
        )

        model.fit()

        model.predict()
        return

if __name__ == "__main__":

    unittest.main()
