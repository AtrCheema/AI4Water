
import unittest

from ai4water.datasets import mg_photodegradation
from ai4water.datasets.utils import LabelEncoder, OneHotEncoder


class TestEncoders(unittest.TestCase):

    def test_labelencoder(self):
        data, _, _ = mg_photodegradation()
        cat_enc1 = LabelEncoder()
        cat_ = cat_enc1.fit_transform(data['Catalyst_type'].values)
        _cat = cat_enc1.inverse_transform(cat_)
        all([a == b for a, b in zip(data['Catalyst_type'].values, _cat)])
        return

    def test_ohe(self):
        data, _, _ = mg_photodegradation()
        cat_enc1 = OneHotEncoder()
        cat_ = cat_enc1.fit_transform(data['Catalyst_type'].values)
        _cat = cat_enc1.inverse_transform(cat_)
        all([a==b for a,b in zip(data['Catalyst_type'].values, _cat)])
        return


if __name__ == "__main__":
    unittest.main()