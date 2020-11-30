import os
import pandas as pd
from inspect import getsourcefile
from os.path import abspath
import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq.pytorch_models import IMVLSTMModel
from dl4seq.utils import make_model

lookback = 10
epochs = 50
file_path = abspath(getsourcefile(lambda:0))
dpath = os.path.join(os.path.join(os.path.dirname(os.path.dirname(file_path)), "dl4seq"), "data")
fname = os.path.join(dpath, "nasdaq100_padding.csv")
df = pd.read_csv(fname)

data_config, nn_config = make_model(batch_size=16,
                                    lookback=lookback,
                                    lr=0.001,
                                    use_predicted_output=True,
                                    epochs=epochs)


model = IMVLSTMModel(data_config=data_config,
                     nn_config=nn_config,
                     data=df,
                     intervals=data_config['intervals']
                     )

model.build()
h = model.train(st=0, en=1000)

x, y = model.predict(st=0, en=1000)
model.plot_activations()
