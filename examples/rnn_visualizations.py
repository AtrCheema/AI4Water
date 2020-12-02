import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

from dl4seq import Model
from dl4seq.utils import make_model

lookback = 10
batch_size = 32
examples = batch_size*10
epochs = 1
ins = 16
units = 64
outs = 1
batch_shape = (batch_size, lookback,  ins)
layers = {
    "Input": {"config": {"shape": (lookback, ins)}},
    "LSTM": {"config": {"units": units, "return_sequences": True}},
    "Flatten": {"config": {}},
    "Dense": {"config": {"units": outs}},
    "Reshape": {"config": {"target_shape": (outs,1)}}
}

config = make_model(
    layers = layers,
    lookback=lookback,
    epochs=epochs,
    batch_size=batch_size,
)

model = Model(config, data=None)

x = np.random.random((examples, lookback, ins))
y = np.random.random((examples, outs, 1))
model.train(data=(x,y))


model.plot_layer_outputs(data=(x,y))
model.plot_act_grads(data=(x,y))
model.plot_weights()
model.plot_weight_grads(data=(x,y))