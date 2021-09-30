import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import numpy as np
import pandas as pd

from ai4water import Model
from ai4water.post_processing import Interpret


default_model = {
    'layers': {
        "Dense_0": {'units': 64, 'activation': 'relu'},
        "Flatten": {},
        "Dense_3": 1,
        "Reshape": {"target_shape": (1, 1)}
    }
}


examples = 2000
ins = 5
outs = 1
in_cols = ['input_'+str(i) for i in range(5)]
out_cols = ['output']
cols=  in_cols + out_cols
data1 = pd.DataFrame(np.arange(int(examples*len(cols))).reshape(-1,examples).transpose(),
                    columns=cols,
                    index=pd.date_range("20110101", periods=examples, freq="D"))


def build_model(**kwargs):

    model = Model(
        data=data1.astype(np.float32),
        verbosity=0,
        batch_size=16,
        lookback=7,
        transformation=None,  # todo, test with transformation
        epochs=1,
        **kwargs
    )

    return model

class TestInterpret(unittest.TestCase):

    def test_plot_feature_importance(self):

        model = build_model(input_features=in_cols,
                            output_features=out_cols,
                            model=default_model)
        Interpret(model).plot_feature_importance(np.random.randint(1, 10, 5))




if __name__ == "__main__":
    unittest.main()