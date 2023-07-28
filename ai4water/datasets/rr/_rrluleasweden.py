
from typing import Union

from ai4water.backend import pd, os

from .._datasets import Datasets
from ..utils import check_st_en


class RRLuleaSweden(Datasets):
    """
    Rainfall runoff data for an urban catchment from 2016-2019 following the work
    of `Broekhuizen et al., 2020 <https://doi.org/10.5194/hess-24-869-2020>`_ .
    """
    url = "https://zenodo.org/record/3931582"
    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

    def fetch(
            self,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None
    ):
        """fetches rainfall runoff data

        Parameters
        ----------
            st : optional
                start of data to be fetched. By default the data starts from
                2016-06-16 20:50:00
            en : optional
                end of data to be fetched. By default the end is 2019-09-15 18:41
        """

        flow = self.fetch_flow(st,en)
        pcp = self.fetch_pcp(st, en)
        return flow, pcp

    def fetch_flow(
            self,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None
    )->pd.DataFrame:
        """fetches flow data

        Parameters
        ----------
            st : optional
                start of data to be fetched. By default the data starts from
                2016-06-16 20:50:00
            en : optional
                end of data to be fetched. By default the end is 2019-09-15 18:35:00

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (37_618, 3) where the columns are velocity,
            level and flow rate

        Examples
        --------
            >>> from ai4water.datasets import RRLuleaSweden
            >>> dataset = RRLuleaSweden()
            >>> flow = dataset.fetch_flow()
            >>> flow.shape
            (37618, 3)
        """
        fname = os.path.join(self.ds_dir, "flow_2016_2019.csv")
        df = pd.read_csv(fname, sep=";")
        df.index = pd.to_datetime(df.pop("time"))
        return check_st_en(df, st, en)

    def fetch_pcp(
            self,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None
    )->pd.DataFrame:
        """fetches precipitation data

        Parameters
        ----------
            st : optional
                start of data to be fetched. By default the data starts from
                2016-06-16 19:48:00
            en : optional
                end of data to be fetched. By default the end is 2019-10-26 23:59:00

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (967_080, 1)

        Examples
        --------
            >>> from ai4water.datasets import RRLuleaSweden
            >>> dataset = RRLuleaSweden()
            >>> pcp = dataset.fetch_pcp()
            >>> pcp.shape
            (967080, 1)

        """

        fname = os.path.join(self.ds_dir, "prec_2016_2019.csv")
        df = pd.read_csv(fname, sep=";")
        df.index = pd.to_datetime(df.pop("time"))
        return check_st_en(df, st, en)

