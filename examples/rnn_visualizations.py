import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from AI4Water import Model

lookback = 10
batch_size = 32
examples = batch_size*10
epochs = 1
ins = 16
units = 16
outs = 1
batch_shape = (batch_size, lookback,  ins)
layers = {
    "Input": {"config": {"shape": (lookback, ins)}},
    "LSTM": {"config": {"units": units, "return_sequences": True}},
    "Flatten": {"config": {}},
    "Dense": {"config": {"units": outs}},
}

model = Model(
    model={'layers':layers},
    lookback=lookback,
    epochs=epochs,
    batch_size=batch_size,
    data=None)

x = np.random.random((examples, lookback, ins))
y = np.random.random((examples, outs, 1))
model.fit(data=(x,y))


model.plot_layer_outputs(data=(x,y))
model.plot_act_grads(data=(x,y))
model.plot_weights()
model.plot_weight_grads(data=(x,y))