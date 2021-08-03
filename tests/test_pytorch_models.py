import os
import pandas as pd
from inspect import getsourcefile
from os.path import abspath
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from ai4water.pytorch_models import HARHNModel
from ai4water.utils.datasets import load_nasdaq

lookback = 10
epochs = 50
df = load_nasdaq()


model = HARHNModel(batch_size=16,
                    lookback=lookback,
                    lr=0.001,
                    use_predicted_output=True,
                    intervals=((0, 146,),
                              (145, 386,),
                              (385, 628,),
                              (625, 821,),
                              (821, 1110),
                              (1110, 1447)),
                    epochs=epochs,
                     data=df,
                     )

h = model.train(st=0, en=1000)

x, y = model.predict(st=0, en=1000)
#model.plot_activations()
