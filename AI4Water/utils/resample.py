import re

import pandas as pd
import numpy as np


class Resampler(object):
    """Resamples time-series data from one frequency to another frequency.
    """
    min_in_freqs = {
        'MIN': 1,
        'MINUTE': 1,
        'DAILY': 1440,
        'D': 1440,
        'HOURLY': 60,
        'HOUR': 60,
        'H': 60,
        'MONTHLY': 43200,
        'M': 43200,
        'YEARLY': 525600
        }

    def __init__(self, data, freq, how='mean', verbosity=1):
        """
        Arguments:
            data : data to use
            freq : frequency at which to transform/resample
            how : string or dictionary mapping to columns in data defining how to resample the data.
        """
        data = pd.DataFrame(data)
        self.orig_df = data.copy()
        self.target_freq = self.freq_in_mins_from_string(freq)
        self.how = self.check_how(how)
        self.verbosity = verbosity


    def __call__(self, *args, **kwargs):
        if self.target_freq > self.orig_freq:
            # we want to calculate at higher/larger time-step
            return self.downsample()

        else:
            # we want to calculate at smaller time-step
            return self.upsamle()

    @property
    def orig_freq(self):
        return self.freq_in_mins_from_string(pd.infer_freq(self.orig_df.index))

    @property
    def allowed_freqs(self):
        return self.min_in_freqs.keys()

    def check_how(self, how):
        if not isinstance(how, str):
            assert isinstance(how, dict)
            assert len(how) == len(self.orig_df.columns)
        else:
            assert isinstance(how, str)
            how = {col:how for col in self.orig_df.columns}
        return how

    def downsample(self):
        df = pd.DataFrame()
        for col in self.orig_df:
            _df = downsample_df(self.orig_df[col], how=self.how[col], target_freq=self.target_freq)
            df = pd.concat([df, _df], axis=1)

        return df

    def upsamle(self, drop_nan=True):
        df = pd.DataFrame()
        for col in self.orig_df:
            _df = upsample_df(self.orig_df[col], how=self.how[col], target_freq=self.target_freq)
            df = pd.concat([df, _df], axis=1)

        # concatenation of dataframes where one sample was upsampled with linear and the other with same, will result
        # in different length and thus concatenation will add NaNs to the smaller column.
        if drop_nan:
            df = df.dropna()
        return df

    def str_to_mins(self, input_string: str) -> int:

        return self.min_in_freqs[input_string]

    def freq_in_mins_from_string(self, input_string: str) -> int:

        if has_numbers(input_string):
            in_minutes = split_freq(input_string)
        elif input_string.upper() in ['D', 'H', 'M', 'DAILY', 'HOURLY', 'MONTHLY', 'YEARLY', 'MIN', 'MINUTE']:
            in_minutes = self.str_to_mins(input_string.upper())
        else:
            raise TypeError("invalid input string", input_string)

        return int(in_minutes)


def downsample_df(df, how, target_freq):

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    assert how in ['mean', 'sum']
    # from low timestep to high timestep i.e from 1 hour to 24 hour
    # For quantities like temprature, relative humidity, Q, wind speed
    if how == 'mean':
        return df.resample(f'{target_freq}min').mean()
    # For quantities like 'rain', solar radiation', evapotranspiration'
    elif how == 'sum':
        return df.resample(f'{target_freq}min').sum()

def upsample_df(df,  how:str, target_freq:int):
    """drop_nan: if how='linear', we may """
    # from larger timestep to smaller timestep, such as from daily to hourly
    out_freq = str(target_freq) + 'min'

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    col_name  = df.columns[0]
    nan_idx = df.isna()  # preserving indices with nan values
    assert df.shape[1] <=1

    nan_idx_r = nan_idx.resample(out_freq).ffill()
    nan_idx_r = nan_idx_r.fillna(False)  # the first value was being filled with NaN, idk y?
    data_frame = df.copy()

    # For quantities like temprature, relative humidity, Q, wind speed, we would like to do an interpolation
    if how == 'linear':
        data_frame = data_frame.resample(out_freq).interpolate(method='linear')
        # filling those interpolated values with NaNs which were NaN before interpolation
        data_frame[nan_idx_r] = np.nan

    # For quantities like 'rain', solar radiation', evapotranspiration', we would like to distribute them equally
    # at smaller time-steps.
    elif how == 'same':
        # distribute rainfall equally to smaller time steps. like hourly 17.4 will be 1.74 at 6 min resolution
        idx = data_frame.index[-1] + get_offset(data_frame.index.freqstr)
        data_frame = data_frame.append(data_frame.iloc[[-1]].rename({data_frame.index[-1]: idx}))
        data_frame = add_freq(data_frame)
        df1 = data_frame.resample(out_freq).ffill().iloc[:-1]
        df1[col_name ] /= df1.resample(data_frame.index.freqstr)[col_name ].transform('size')
        data_frame = df1.copy()
        # filling those interpolated values with NaNs which were NaN before interpolation
        data_frame[nan_idx_r] = np.nan

    else:
        raise ValueError(f"unoknown method to transform '{how}'")

    return data_frame


def add_freq(df, assert_feq=False, freq=None, method=None):

    idx = df.index.copy()
    if idx.freq is None:
        _freq = pd.infer_freq(idx)
        idx.freq = _freq

        if idx.freq is None:
            if assert_feq:
                df = force_freq(df, freq, method=method)
            else:

                raise AttributeError('no discernible frequency found.  Specify'
                                     ' a frequency string with `freq`.'.format())
        else:
            df.index = idx
    return df


def force_freq(data_frame, freq_to_force, method=None):

    old_nan_counts = data_frame.isna().sum()
    old_shape = data_frame.shape
    dr = pd.date_range(data_frame.index[0], data_frame.index[-1], freq=freq_to_force)

    df_unique = data_frame[~data_frame.index.duplicated(keep='first')]  # first remove duplicate indices if present
    if method:
        df_idx_sorted = df_unique.sort_index()
        df_reindexed = df_idx_sorted.reindex(dr, method='nearest')
    else:
        df_reindexed = df_unique.reindex(dr, fill_value=np.nan)

    df_reindexed.index.freq = pd.infer_freq(df_reindexed.index)
    new_nan_counts = df_reindexed.isna().sum()
    print('Frequency {} is forced to dataframe, NaN counts changed from {} to {}, shape changed from {} to {}'
          .format(df_reindexed.index.freq, old_nan_counts.values, new_nan_counts.values,
                  old_shape, df_reindexed.shape))
    return df_reindexed


def split_freq(freq_str: str) -> int:
    match = re.match(r"([0-9]+)([a-z]+)", freq_str, re.I)
    if match:
        minutes, freq = match.groups()
        if freq.upper() in ['H', 'HOURLY', 'HOURS', 'HOUR']:
            minutes = int(minutes) * 60
        elif freq.upper() in ['D', 'DAILY', 'DAY', 'DAYS']:
            minutes = int(minutes) * 1440
        return int(minutes)
    else:
        raise NotImplementedError

TIME_STEP = {'D': 'Day', 'H': 'Hour', 'M': 'MonthEnd'}

def get_offset(freqstr: str) -> str:
    offset_step = 1
    if freqstr in TIME_STEP:
        freqstr = TIME_STEP[freqstr]
    elif has_numbers(freqstr):
        in_minutes = split_freq(freqstr)
        freqstr = 'Minute'
        offset_step = int(in_minutes)

    offset = getattr(pd.offsets, freqstr)(offset_step)

    return offset

def has_numbers(input_string: str) -> bool:
    return bool(re.search(r'\d', input_string))