
import unittest

from ai4water.experiments.utils import regression_space, regression_models
from ai4water.experiments.utils import classification_space, classification_models


class TestUtils(unittest.TestCase):

    def test_regression_space(self):
        space = regression_space(2)
        assert isinstance(space, dict)
        assert len(space)>20
        return

    def test_classification_space(self):
        space = classification_space(2)
        assert isinstance(space, dict)
        assert len(space)>20
        return

    def test_regression_models(self):
        models = regression_models()
        assert isinstance(models, list)
        assert len(models)>20
        return

    def test_classification_models(self):
        models = classification_models()
        assert isinstance(models, list)
        assert len(models)>20
        return


if __name__ == "__main__":
    unittest.main()