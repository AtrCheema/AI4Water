import os
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from ai4water.pytorch_models import HARHNModel, IMVModel
from ai4water.utils.datasets import arg_beach, load_u1

lookback = 10
epochs = 50
df = arg_beach()


model = HARHNModel(data=load_u1(),
                   teacher_forcing=True,
                   epochs=3,
                   model={'layers': {'n_conv_lyrs': 3, 'enc_units': 64, 'dec_units': 64}}
                   )
model.fit()
t,p = model.predict()
s = model.evaluate('training')


model = IMVModel(data=arg_beach(),
                 val_data="same",
                 val_fraction=0.0,
                 epochs=2,
                 lr=0.0001,
                 batch_size=4,
                 train_data='random',
                 transformation=[
                     {'method': 'minmax', 'features': list(arg_beach().columns)[0:-1]},
                     {'method': 'log2', 'features': ['tetx_coppml'], 'replace_zeros': True, 'replace_nans': True}
                 ],
                 model={'layers': {'hidden_units': 64}}
                 )

model.fit()
model.predict()
model.evaluate('training')
model.interpret()