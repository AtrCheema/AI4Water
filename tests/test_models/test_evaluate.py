
import unittest

import numpy as np

from ai4water import Model
from ai4water.datasets import busan_beach


beach_data = busan_beach()
inputs = beach_data.columns.tolist()[0:-1]
outputs = beach_data.columns.tolist()[-1:]



class TestEvaluateML(unittest.TestCase):

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