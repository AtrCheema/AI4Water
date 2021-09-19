import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from ai4water import Model

lookback = 10
batch_size = 32
examples = batch_size*10
epochs = 1
ins = 16
units = 16
outs = 1
batch_shape = (batch_size, lookback,  ins)
layers = {
    "Input": {"shape": (lookback, ins)},
    "LSTM": {"units": units, "return_sequences": True},
    "Flatten": {},
    "Dense": outs,
}

model = Model(
    model={'layers': layers},
    lookback=lookback,
    epochs=epochs,
    batch_size=batch_size,
    input_features=[f'in_{i}' for i in range(ins)],
    output_features=['out'],
    data=None)

x = np.random.random((examples, lookback, ins))
y = np.random.random((examples, outs))
model.fit(x=x, y=y)


model.plot_layer_outputs(x=x)
model.plot_act_grads(x=x, y=y)
model.plot_weights()
model.plot_weight_grads(x=x, y=y)
