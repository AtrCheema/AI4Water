import pandas as pd
import numpy as np
import AI4Water
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from AI4Water import Model
from AI4Water.utils.utils import plot
from AI4Water.utils.torch_utils import to_torch_dataset

from AI4Water.utils.datasets import arg_beach


#examples = 1000
#data = np.arange(int(examples*9)).reshape(-1,examples).transpose()
#df = pd.DataFrame(data, columns=[f'in_{i}' for i in range(data.shape[1])])
df = arg_beach(inputs=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm',
       'pcp6_mm', 'pcp12_mm'])

default_model = {'layers':{
    "Linear_0": {"in_features": 8, "out_features": 64},
    "ReLU_0": {},
    "Dropout_0": 0.3,
    "Linear_1": {"in_features": 64, "out_features": 32},
    "ReLU_1": {},
    "Dropout_1": 0.3,
    "Linear_2": {"in_features": 32, "out_features": 16},
    "Linear_3": {"in_features": 16, "out_features": 1},
}}


model = Model(
    model=default_model,
    data=df,
    lookback=1,
    epochs=5,
    transformation='minmax',
    lr=0.0001,
    batch_size=4,
    val_data="same",
)

assert model.trainable_parameters() == 3201
model.fit()
model.predict()




