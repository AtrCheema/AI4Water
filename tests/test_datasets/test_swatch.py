
import unittest

from ai4water.datasets import Swatch


ds = Swatch(path=r'F:\data\Swatch')
df = ds.fetch()

class TestSwatch(unittest.TestCase):

    def test_sites(self):
        sites = ds.sites
        assert isinstance(sites, list)
        assert len(sites) == 26322
        return

    def test_fetch(self):
        assert df.shape == (3901296, 6)

        st_name = "Jordan Lake"
        df1 = df[df['location'] == st_name]
        assert df1.shape == (4, 6)

        return


if __name__ == "__main__":
    unittest.main()
