
import unittest

from ai4water.datasets import RC4USCoast


ds = RC4USCoast()


class TestRC4USCoast(unittest.TestCase):

    def test_parameters(self):
        assert isinstance(ds.parameters, list)
        return


if __name__ == "__main__":
    unittest.main()