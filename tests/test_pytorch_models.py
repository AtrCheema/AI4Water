from models import IMVLSTMModel
from utils import make_model

import pandas as pd

lookback = 10
epochs = 50
df = pd.read_csv("../data/nasdaq100_padding.csv")

data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                     lookback=lookback,
                                                     lr=0.001,
                                                     use_predicted_output=True,
                                                     epochs=epochs)


model = IMVLSTMModel(data_config=data_config, nn_config=nn_config, data = df)

model.build_nn()
h = model.train_nn(st=0, en=1000)

x, y = model.predict(st=0, en=1000)
model.plot_activations()
