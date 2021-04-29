import os
import pandas as pd
from inspect import getsourcefile
from os.path import abspath
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from AI4Water.pytorch_models import HARHNModel
from AI4Water.utils import make_model

lookback = 10
epochs = 50
file_path = abspath(getsourcefile(lambda:0))
dpath = os.path.join(os.path.join(os.path.dirname(os.path.dirname(file_path)), "dl4seq"), "data")
fname = os.path.join(dpath, "nasdaq100_padding.csv")
df = pd.read_csv(fname)


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
