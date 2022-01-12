
import unittest

from ai4water.preprocessing.transformations.utils import InvalidTransformation
from ai4water.preprocessing.transformations import Transformation


class Test(unittest.TestCase):

    def test_InvalidTransformationError(self):

        self.assertRaises(InvalidTransformation,
                          Transformation, method='Minmax')
        return


if __name__ == "__main__":
    unittest.main()
