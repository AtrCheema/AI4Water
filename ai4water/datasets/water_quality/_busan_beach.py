
from typing import Union
from ai4water.backend import pd, os


def busan_beach(
        inputs: list = None,
        target: Union[list, str] = 'tetx_coppml'
) -> pd.DataFrame:
    """
    Loads the Antibiotic resitance genes (ARG) data from a recreational beach
    in Busan, South Korea along with environment variables.

    The data is in the form of
    mutlivariate time series and was collected over the period of 2 years during
    several precipitation events. The frequency of environmental data is 30 mins
    while that of ARG is discontinuous. The data and its pre-processing is described
    in detail in `Jang et al., 2021 <https://doi.org/10.1016/j.watres.2021.117001>`_

    Arguments
    ---------
        inputs :
            features to use as input. By default all environmental data
            is used which consists of following parameters

            - tide_cm
            - wat_temp_c
            - sal_psu
            - air_temp_c
            - pcp_mm
            - pcp3_mm
            - pcp6_mm
            - pcp12_mm
            - wind_dir_deg
            - wind_speed_mps
            - air_p_hpa
            - mslp_hpa
            - rel_hum

        target :
            feature/features to use as target/output. By default
            `tetx_coppml` is used as target.
            Logically one or more from following can be considered as target

            - ecoli
            - 16s
            - inti1
            - Total_args
            - tetx_coppml
            - sul1_coppml
            - blaTEM_coppml
            - aac_coppml
            - Total_otus
            - otu_5575
            - otu_273
            - otu_94

    Returns
    -------
    pd.DataFrame
        a pandas dataframe with inputs and target and indexed
        with pandas.DateTimeIndex

    Examples
    --------
        >>> from ai4water.datasets import busan_beach
        >>> dataframe = busan_beach()
        >>> dataframe.shape
        (1446, 14)
        >>> dataframe = busan_beach(target=['tetx_coppml', 'sul1_coppml'])
        >>> dataframe.shape
        (1446, 15)

    """
    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arg_busan.csv")
    df = pd.read_csv(fpath, index_col="index")
    df.index = pd.to_datetime(df.index)

    default_inputs = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'pcp6_mm',
                      'pcp12_mm', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'
                      ]
    default_targets = [col for col in df.columns if col not in default_inputs]

    if inputs is None:
        inputs = default_inputs

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    df = df[inputs + target]

    return df

