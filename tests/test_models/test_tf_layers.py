
import unittest

from ai4water.functional import Model
from ai4water.models.utils import gen_cat_vocab
from ai4water.utils.utils import TrainTestSplit
from tensorflow.keras.layers import Dense, Input
from ai4water.datasets import mg_photodegradation
from tensorflow.keras.models import Model as KModel
from ai4water.models import FTTransformer, TabTransformer
from ai4water.models._tensorflow.private_layers import Transformer, TransformerBlocks

import numpy as np


# bring the data as DataFrame
data, _, _ = mg_photodegradation()
# Define categorical and numerical features and label
NUMERIC_FEATURES = data.columns.tolist()[0:9]
CAT_FEATURES = ["Catalyst_type", "Anions"]
LABEL = "Efficiency (%)"
# create vocabulary of unique values of categorical features
cat_vocab = gen_cat_vocab(data)
# make sure the data types are correct
data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype(float)
data[CAT_FEATURES] = data[CAT_FEATURES].astype(str)
data[LABEL] = data[LABEL].astype(float)
# split the data into training and test set
splitter = TrainTestSplit(seed=313)
train_data, test_data, _, _ = splitter.split_by_random(data)

# make a list of input arrays for training data
train_x = [train_data[NUMERIC_FEATURES].values,
           train_data[CAT_FEATURES].values]

test_x = [test_data[NUMERIC_FEATURES].values,
          test_data[CAT_FEATURES].values]


class TestLayers(unittest.TestCase):

    def test_transformer(self):
        # test as layer
        inp = Input(shape=(10, 32))
        out, _ = Transformer(4, 32)(inp)
        out = Dense(1)(out)
        model = KModel(inputs=inp, outputs=out)
        model.compile(optimizer="Adam", loss="mse")
        x = np.random.random((100, 10, 32))
        model.fit(x,y = np.random.random(100), verbose=0)

    def test_transformersblock(self):
        inp = Input(shape=(10, 32))
        out, _ = TransformerBlocks(4, 4, 32)(inp)
        out = Dense(1)(out)
        model = KModel(inputs=inp, outputs=out)
        model.compile(optimizer="Adam", loss="mse")
        x = np.random.random((100, 10, 32))
        model.fit(x,np.random.random(100), verbose=0)

    def test_fttransformer(self):
        # build the model
        model = Model(model=FTTransformer(
            len(NUMERIC_FEATURES),cat_vocab, hidden_units=16,
            num_heads=8),
                      verbosity=0)

        assert model.count_params() == 37610, model.count_params()
        model.fit(x=train_x, y=train_data[LABEL].values,
                      validation_data=(test_x, test_data[LABEL].values),
                      epochs=1)
        return

    def test_tabtransformer(self):
        # build the model
        model = Model(model=TabTransformer(
            num_numeric_features=len(NUMERIC_FEATURES),
            cat_vocabulary=cat_vocab,
            hidden_units=16, final_mlp_units=[84, 42]), verbosity=0)

        assert model.count_params() == 26347

        model.fit(x=train_x, y= train_data[LABEL].values,
                      validation_data=(test_x, test_data[LABEL].values), epochs=1)
        return

    def test_ealstm(self):
        from ai4water.models._tensorflow import EALSTM
        import tensorflow as tf
        batch_size, lookback, num_dyn_inputs, num_static_inputs, units = 10, 5, 3, 2, 8
        inputs = tf.range(batch_size * lookback * num_dyn_inputs,
                          dtype=tf.float32)
        inputs = tf.reshape(inputs, (batch_size,
                                     lookback, num_dyn_inputs))
        stat_inputs = tf.range(batch_size * num_static_inputs,
                               dtype=tf.float32)
        stat_inputs = tf.reshape(stat_inputs,
                                 (batch_size, num_static_inputs))
        lstm = EALSTM(units, num_static_inputs)
        h_n = lstm(inputs, stat_inputs)  # -> (batch_size, units)

        # with return sequences
        lstm = EALSTM(units, num_static_inputs, return_sequences=True)
        h_n = lstm(inputs, stat_inputs)  # -> (batch, lookback, units)

        # with return sequences and return_state
        lstm = EALSTM(units, num_static_inputs,
                      return_sequences=True, return_state=True)
        # -> (batch, lookback, units), [(), ()]
        h_n, [c_n, y_hat] = lstm(inputs, stat_inputs)

        # end to end Keras model
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        import numpy as np
        inp_dyn = Input(batch_shape=(batch_size, lookback, num_dyn_inputs))
        inp_static = Input(batch_shape=(batch_size, num_static_inputs))
        lstm = EALSTM(units, num_static_inputs)(inp_dyn, inp_static)
        out = Dense(1)(lstm)
        model = Model(inputs=[inp_dyn, inp_static], outputs=out)
        model.compile(loss='mse')
        print(model.summary())
        # generate hypothetical data and train it
        dyn_x = np.random.random((100, lookback, num_dyn_inputs))
        static_x = np.random.random((100, num_static_inputs))
        y = np.random.random((100, 1))
        h = model.fit(x=[dyn_x, static_x], y=y, batch_size=batch_size)
        return

    def test_mclstm(self):
        from tensorflow.keras.models import Model
        from ai4water.models._tensorflow import MCLSTM
        import tensorflow as tf
        inputs = tf.range(150, dtype=tf.float32)
        inputs = tf.reshape(inputs, (10, 5, 3))
        mc = MCLSTM(1, 2, 8, 1)
        h = mc(inputs)  # (batch, units)

        mc = MCLSTM(1, 2, 8, 1, return_sequences=True)
        h = mc(inputs)  # (batch, lookback, units)

        mc = MCLSTM(1, 2, 8, 1, return_state=True)
        _h, _o, _c = mc(inputs)  # (batch, lookback, units)

        mc = MCLSTM(1, 2, 8, 1, return_state=True, return_sequences=True)
        _h, _o, _c = mc(inputs)  # (batch, lookback, units)

        # with time_major as True
        inputs = tf.range(150, dtype=tf.float32)
        inputs = tf.reshape(inputs, (5, 10, 3))
        mc = MCLSTM(1, 2, 8, 1, time_major=True)
        _h = mc(inputs)  # (batch, units)

        mc = MCLSTM(1, 2, 8, 1, time_major=True, return_sequences=True)
        _h = mc(inputs)  # (lookback, batch, units)

        mc = MCLSTM(1, 2, 8, 1, time_major=True, return_state=True)
        _h, _o, _c = mc(inputs)  # (batch, units), ..., (lookback, batch, units)

        # end to end keras Model
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Model
        import numpy as np

        inp = Input(batch_shape=(32, 10, 3))
        lstm = MCLSTM(1, 2, 8)(inp)
        out = Dense(1)(lstm)

        model = Model(inputs=inp, outputs=out)
        model.compile(loss='mse')

        x = np.random.random((320, 10, 3))
        y = np.random.random((320, 1))
        y = model.fit(x=x, y=y)
        return


if __name__ == "__main__":
    unittest.main()