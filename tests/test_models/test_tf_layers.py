
import unittest

from ai4water import Model
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
            len(NUMERIC_FEATURES),cat_vocab, hidden_units=16, num_heads=8),
                      verbosity=0)

        assert model.count_params() == 37610, model.count_params()
        model.fit(x=train_x, y=train_data[LABEL].values,
                      validation_data=(test_x, test_data[LABEL].values),
                      epochs=1)
        return

    def test_tabtransformer(self):
        # build the model
        model = Model(model=TabTransformer(
            num_numeric_features=len(NUMERIC_FEATURES),cat_vocabulary=cat_vocab,
            hidden_units=16, final_mlp_units=[84, 42]), verbosity=0)

        assert model.count_params() == 26347

        model.fit(x=train_x, y= train_data[LABEL].values,
                      validation_data=(test_x, test_data[LABEL].values), epochs=1)
        return

if __name__ == "__main__":
    unittest.main()