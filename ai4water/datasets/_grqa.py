
__all__ = ["GRQA"]

from typing import Union, List

from ai4water.backend import pd, os

from ._datasets import Datasets
from .utils import check_st_en


class GRQA(Datasets):
    """
    Global River Water Quality Archive following the work of Virro et al., 2021 [21]_.

    .. [21] https://essd.copernicus.org/articles/13/5483/2021/
    """

    url = 'https://zenodo.org/record/7056647#.YzBzDHZByUk'


    def __init__(
            self,
            download_source:bool = False,
            **kwargs):
        """
        parameters
        ----------
        download_source : bool
            whether to download source data or not
        """
        super().__init__(**kwargs)

        files = ['GRQA_data_v1.3.zip', 'GRQA_meta.zip']
        if download_source:
            files += ['GRQA_source_data.zip']
        self._download(include=files)


    @property
    def files(self):
        return os.listdir(os.path.join(self.ds_dir, "GRQA_data_v1.3", "GRQA_data_v1.3"))

    @property
    def parameters(self):
        return [f.split('_')[0] for f in self.files]

    def fetch_parameter(
            self,
            parameter: str = "COD",
            site_name: Union[List[str], str] = None,
            country: Union[List[str], str] = None,
            st:Union[int, str, pd.DatetimeIndex] = None,
            en:Union[int, str, pd.DatetimeIndex] = None,
    )->pd.DataFrame:
        """
        parameters
        ----------
        parameter : str, optional
            name of parameter
        site_name : str/list, optional
            location for which data is to be fetched.
        country : str/list optional (default=None)
        st : str
            starting date date or index
        en : str
            end date or index

        Returns
        -------
        pd.DataFrame
            a pandas dataframe

        Example
        --------
        >>> from ai4water.datasets import GRQA
        >>> dataset = GRQA()
        >>> df = dataset.fetch_parameter()
        fetch data for only one country
        >>> cod_pak = dataset.fetch_parameter("COD", country="Pakistan")
        fetch data for only one site
        >>> cod_kotri = dataset.fetch_parameter("COD", site_name="Indus River - at Kotri")
        we can find out the number of data points and sites available for a specific country as below
        >>> for para in dataset.parameters:
        >>>     data = dataset.fetch_parameter(para, country="Germany")
        >>>     if len(data)>0:
        >>>         print(f"{para}, {df.shape}, {len(df['site_name'].unique())}")

        """

        assert isinstance(parameter, str)
        assert parameter in self.parameters

        if isinstance(site_name, str):
            site_name = [site_name]

        if isinstance(country, str):
            country = [country]

        df = self._load_df(parameter)

        if site_name is not None:
            assert isinstance(site_name, list)
            df = df[df['site_name'].isin(site_name)]
        if country is not None:
            assert isinstance(country, list)
            df = df[df['site_country'].isin(country)]

        df.index = pd.to_datetime(df.pop("obs_date") + " " + df.pop("obs_time"))

        return check_st_en(df, st, en)

    def _load_df(self, parameter):
        if hasattr(self, f"_load_{parameter}"):
            return getattr(self, f"_load_{parameter}")()

        fname = os.path.join(self.ds_dir, "GRQA_data_v1.3", "GRQA_data_v1.3", f"{parameter}_GRQA.csv")
        return pd.read_csv(fname, sep=";")

    def _load_DO(self):
        # read_csv is causing mysterious errors

        f = os.path.join(self.ds_dir, "GRQA_data_v1.3",
                         "GRQA_data_v1.3", f"DO_GRQA.csv")
        lines = []
        with open(f, 'r', encoding='utf-8') as fp:
            for idx, line in enumerate(fp):
                lines.append(line.split(';'))

        return pd.DataFrame(lines[1:], columns=lines[0])
