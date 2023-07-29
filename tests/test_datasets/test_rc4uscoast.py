
import unittest

import numpy as np
import xarray as xr

from ai4water.datasets import RC4USCoast


ds = RC4USCoast(path=r'F:\data\RC4USCoast')


class TestRC4USCoast(unittest.TestCase):

    def test_parameters(self):
        params = ds.parameters
        assert isinstance(params, list)
        assert len(params) == 27
        return

    def test_stations(self):
        stns = ds.stations
        assert isinstance(stns, np.ndarray), type(stns)
        assert len(stns) == 140
        return

    def test_fetch_q(self):
        # get data of all stations as DataFrame
        q = ds.fetch_q("all")
        assert q.shape == (852, 140)

        # get data of only two stations
        q = ds.fetch_q([1, 10])
        assert q.shape == (852, 2)
        # get data as xarray Dataset
        q = ds.fetch_q("all", as_dataframe=False)
        assert isinstance(q, xr.Dataset)

        data = ds.fetch_q("all", as_dataframe=True, st="20000101", en="20181230")
        assert data.shape == (228, 140), data.shape
        return

    def test_fetch_chem(self):

        data = ds.fetch_chem(['temp', 'do'])
        assert isinstance(data, xr.Dataset)
        data = ds.fetch_chem(['temp', 'do'], as_dataframe=True)
        assert data.shape == (119280, 4)

        data = ds.fetch_chem(['temp', 'do'], st="19800101", en="20181230",
                             as_dataframe=True)
        assert data.shape == (65520, 4)
        return


if __name__ == "__main__":
    unittest.main()