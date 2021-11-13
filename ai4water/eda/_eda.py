import os
import math
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from .utils import pac_yw, auto_corr, plot_autocorr
from ai4water.utils.utils import _missing_vals
from ai4water.utils.visualizations import Plot
from ai4water.utils.plotting_tools import bar_chart
from ai4water.utils.utils import find_tot_plots
from ai4water.preprocessing import Transformations
from ai4water.utils.utils import  dict_to_file, dateandtime_now, ts_features


# qq plot
# decompose into trend/seasonality and noise


class EDA(Plot):
    """Performns a comprehensive exploratory data analysis on a tabular/structured
    data. It is meant to be a one stop shop for eda.

    Methods
    ---------
    - heatmap
    - box_plot
    - plot_missing
    - plot_histograms
    - plot_index
    - plot_data
    - plot_pcs
    - grouped_scatter
    - correlation
    - stats
    - autocorrelation
    - partial_autocorrelation
    - probability_plots
    - lag_plot
    - plot_ecdf

    Example:
        >>> from ai4water.datasets import arg_beach
        >>> eda = EDA(data=arg_beach())
        >>> eda()
    """

    def __init__(
            self,
            data:Union[pd.DataFrame, List[pd.DataFrame], Dict, np.ndarray],
            in_cols=None,
            out_cols=None,
            path=None,
            dpi=300,
            save=True,
            show=True,
    ):
        """

        Arguments:
            data : either a dataframe, or list of dataframes or a dictionary whose
                values are dataframes or a numpy array
            in_cols :
            out_cols :
            path : the path where to save the figures
            save : whether to save the plots or not
            show : whether to show the plots or not
            dpi :  the resolution with which to save the image
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        self.data = data
        self.in_cols = in_cols
        self.out_cols = out_cols
        self.show = show

        super().__init__(path, save=save, dpi=dpi)

    @property
    def in_cols(self):
        return self._in_cols

    @in_cols.setter
    def in_cols(self, x):
        if x is None:
            if isinstance(self.data, pd.DataFrame):
                x = self.data.columns.to_list()
            else:
                raise ValueError(f"unsupported type of {self.data.__class__.__name__}")
        self._in_cols = x

    @property
    def out_cols(self):
        return self._out_cols

    @out_cols.setter
    def out_cols(self, x):
        if x is None:
            if isinstance(self.data, pd.DataFrame):
                x = []
            else:
                raise ValueError
        self._out_cols = x

    def _save_or_show(self, fname, dpi=None):

        return self.save_or_show(where='data', fname=fname, show=self.show, dpi=dpi, close=False)

    def __call__(self,
                 methods: Union[str, list] = 'all',
                 cols=None,
                 ):
        """Shortcut to draw maximum possible plots.

        Arguments:
            methods :
            cols :
        """
        all_methods = [
            'heatmap', 'plot_missing', 'plot_histograms', 'plot_data',
            'plot_index', 'stats', 'box_plot',
            'autocorrelation', 'partial_autocorrelation',
            'lag_plot', 'plot_ecdf'
        ]

        if isinstance(self.data, pd.DataFrame) and self.data.shape[-1] > 1:
            all_methods = all_methods + [# 'plot_pcs',
                                         'grouped_scatter',
                                         'correlation']

        if isinstance(methods, str):
            if methods == 'all':
                methods = all_methods
            else:
                methods = [methods]
        else:
            assert isinstance(methods, list)

        assert all([m in all_methods for m in methods])

        for m in methods:
            if m in ["plot_index", "stats", "plot_pcs"]:
                getattr(self, m)()
            else:
                getattr(self, m)(cols=cols)

        return

    def heatmap(self, cols=None, st=None, en=None, **kwargs):
        """
        Plots a heatmap which depicts missing values.

        Arguments:
            cols : columns to use to draw heatmap
            kwargs :
            st : Starting index of data to be used
            en : end index of data to be used

        Return:
            None

        Example:
            >>> from ai4water.datasets import arg_beach
            >>> data = arg_beach()
            >>> vis = EDA(data)
            >>> vis.heatmap()
        """
        return self._call_method('_heatmap_df', cols=cols, st=st, en=en, **kwargs)

    def _heatmap_df(
            self,
            data: pd.DataFrame,
            cols=None,
            st=None,
            en=None,
            spine_color: str = "#EEEEEE",
            title=None,
            title_fs=16,
            fname="",
            **kwargs
    ):
        """
        Plots a heat map of a dataframe. Helpful to show where missing values are
        located in a dataframe.

        Arguments:
            data : pd.DataFrame,
            cols : list, columns from data to be used.
            st :
            en :
            spine_color
            title: str, title of the plot
            title_fs: int, font size of title
            fname: str, name of saved file, only valid if save is True.
            kwargs: following kwargs are allowed:
            xtick_labels_fs, 12
            ytick_labels_fs, 20
            figsize: tuple
            any additional keyword argument will be passed to sns.heatmap

        Return:

        """
        if cols is None:
            cols = data.columns

        data = _preprocess_df(data, st, en)

        _kwargs = {
            "xtick_labels_fs": 12,
            "ytick_labels_fs": 20,
            "figsize": (5 + len(cols)*0.25, 10 + len(cols)*0.1),
        }
        for k in _kwargs.keys():
            if k in kwargs:
                _kwargs[k] = kwargs.pop(k)

        show_time_on_yaxis = False
        if isinstance(data.index, pd.DatetimeIndex):
            show_time_on_yaxis = True

        _, axis = plt.subplots(figsize=_kwargs['figsize'])
        # ax2 - Heatmap
        sns.heatmap(data[cols].isna(), cbar=False, cmap="binary", ax=axis, **kwargs)

        axis.set_yticks(axis.get_yticks()[0::5].astype('int'))

        if show_time_on_yaxis:
            index = pd.date_range(data.index[0], data.index[-1], periods=len(axis.get_yticks()))
            # formatting y-ticklabels
            index = [d.strftime('%Y-%m-%d') for d in index]
            axis.set_yticklabels(index, fontsize="18")
        else:
            axis.set_yticklabels(axis.get_yticks(),
                                 fontsize=_kwargs['ytick_labels_fs'])
        axis.set_xticklabels(
            axis.get_xticklabels(),
            horizontalalignment="center",
            fontweight="light",
            fontsize=_kwargs['xtick_labels_fs'],
        )
        axis.tick_params(length=1, colors="#111111")
        axis.set_ylabel("Examples", fontsize="24")
        for _, spine in axis.spines.items():
            spine.set_visible(True)
            spine.set_color(spine_color)
        if title is not None:
            axis.set_title(title, fontsize=title_fs)

        self._save_or_show(fname=fname + '_heat_map', dpi=500)
        return axis

    def plot_missing(self, st=None, en=None, cols=None, **kwargs):
        """
        plot data to indicate missingness in data

        Arguments:
            cols : columns to be used.
            st :
            en :

        Example:
            >>> from ai4water.datasets import arg_beach
            >>> data = arg_beach()
            >>>vis = EDA(data)
            >>>vis.plot_missing()
        """
        return self._call_method('_plot_missing_df', cols=cols, st=st, en=en, **kwargs)

    def _plot_missing_df(self,
                         data: pd.DataFrame,
                         cols=None,
                         st=None,
                         en=None,
                         fname: str = '',
                         **kwargs):
        """
        kwargs:
            xtick_labels_fs
            ytick_labels_fs
            figsize
            any other keyword argument will be passed to bar_chart()
        """
        ax1 = None

        if cols is None:
            cols = data.columns
        data = data[cols]

        data = _preprocess_df(data, st, en)

        # Identify missing values
        mv_total, _, mv_cols, _, mv_cols_ratio = _missing_vals(data).values()

        _kwargs = {
            "xtick_labels_fs": 12,
            "ytick_labels_fs": 20,
            "figsize": (5 + len(cols)*0.25, 10 + len(cols)*0.1),
        }
        for k in _kwargs.keys():
            if k in kwargs:
                _kwargs[k] = kwargs.pop(k)
        if mv_total < 6:
            print("No missing values found in the dataset.")
        else:
            # Create figure and axes
            plt.close('all')
            fig = plt.figure(figsize=_kwargs['figsize'])
            gs = fig.add_gridspec(nrows=1, ncols=1, left=0.1, wspace=0.05)
            ax1 = fig.add_subplot(gs[:1, :5])

            # ax1 - Barplot
            ax1 = bar_chart(labels=list(data.columns), values=np.round(mv_cols_ratio * 100, 2),
                            orient='v', axis=ax1)

            ax1.set(frame_on=True, xlim=(-0.5, len(mv_cols) - 0.5))
            ax1.set_ylim(0, np.max(mv_cols_ratio) * 100)
            ax1.grid(linestyle=":", linewidth=1)

            ax1.set_yticklabels(ax1.get_yticks(), fontsize="18")
            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax1.set_ylabel("Missing Percentage", fontsize=_kwargs['ytick_labels_fs'])

            ax1.set_xticklabels(
                ax1.get_xticklabels(),
                horizontalalignment="center",
                fontweight="light",
                rotation=90,
                fontsize=_kwargs['xtick_labels_fs'],
            )
            ax1.tick_params(axis="y", colors="#111111", length=1)

            # annotate missing values on top of the bars
            for rect, label in zip(ax1.patches, mv_cols):
                height = rect.get_height()
                ax1.text(
                    0.1 + rect.get_x() + rect.get_width() / 2,
                    height + height*0.02,
                    label,
                    ha="center",
                    va="bottom",
                    rotation="90",
                    alpha=0.5,
                    fontsize="11",
                )
            self._save_or_show(fname=fname+'_missing_vals', dpi=500)
        return ax1

    def plot_data(
            self,
            st=None,
            en=None,
            freq: str = None,
            cols=None,
            max_subplots: int = 10,
            **kwargs
    ):
        """
        Plots the data.

        Arguments:
            st :
            en :
            cols : columns in self.data to plot
            max_subplots : number of subplots within one plot. Each feature will
                be shown in a separate subplot.
            freq : one of 'daily', 'weekly', 'monthly', 'yearly', determines
                interval of plot of data. It is valid for only time-series data.
            kwargs : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html

        Rreturn:

        Example:
            >>> from ai4water.datasets import arg_beach
            >>> eda = EDA(arg_beach())
            >>> eda.plot_data(subplots=True, figsize=(12, 14), sharex=True)
            >>> eda.plot_data(freq='monthly', subplots=True, figsize=(12, 14), sharex=True)
        ```
        """
        # TODO, this method should be available from `model` as well
        return self._call_method("_plot_df", st=st, en=en, cols=cols, freq=freq, max_subplots=max_subplots,
                                 **kwargs)

    def _plot_df(self,
                 df,
                 st=None,
                 en=None,
                 cols=None,
                 freq=None, max_subplots=10,
                 prefix='',
                 leg_kws=None,
                 label_kws=None,
                 tick_kws=None,
                 **kwargs):
        """Plots each columns of dataframe and saves it if `save` is True.

         max_subplots: determines how many sub_plots are to be plotted within
            one plot. If dataframe contains columns
         greater than max_subplots, a separate plot will be generated for remaining columns.
         """
        assert isinstance(df, pd.DataFrame)
        plt.close('all')

        if leg_kws is None:
            leg_kws = {'fontsize': 14}
        if label_kws is None:
            label_kws = {'fontsize': 14}
        if tick_kws is None:
            tick_kws = {'axis': "both", 'which': 'major', 'labelsize': 12}

        df = _preprocess_df(df, st, en, cols)

        if df.shape[1] <= max_subplots:

            if freq is None:
                kwargs = plot_style(df, **kwargs)
                axis = df.plot(**kwargs)
                if isinstance(axis, np.ndarray):
                    for ax in axis:
                        set_axis_paras(ax, leg_kws, label_kws, tick_kws)
                else:
                    set_axis_paras(axis, leg_kws, label_kws, tick_kws)

                self._save_or_show(fname=f"input_{prefix}")
            else:
                self._plot_df_with_freq(df, freq, **kwargs)
        else:
            tot_plots = find_tot_plots(df.shape[1], max_subplots)

            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = df.iloc[:, st:en]

                if freq is None:
                    kwargs = plot_style(sub_df, **kwargs)
                    axis = sub_df.plot(**kwargs)
                    if kwargs.get('subplots', False):
                        for ax in axis:
                            ax.legend(**leg_kws)
                            ax.set_ylabel(ax.get_ylabel(), **label_kws)
                            ax.set_xlabel(ax.get_xlabel(), **label_kws)
                            ax.tick_params(**tick_kws)
                    else:
                        axis.legend(**leg_kws)
                        axis.set_ylabel(axis.get_ylabel(), **label_kws)
                        axis.set_xlabel(axis.get_xlabel(), **label_kws)
                        axis.tick_params(**tick_kws)

                    self._save_or_show(fname=f'input_{prefix}_{st}_{en}')
                else:
                    self._plot_df_with_freq(sub_df, freq, prefix=f'{prefix}_{st}_{en}', **kwargs)
        return

    def _plot_df_with_freq(self,
                           df: pd.DataFrame,
                           freq: str,
                           prefix: str = '',
                           **kwargs):
        """Plots a dataframe which has data as time-series and its index is pd.DatetimeIndex"""
        validate_freq(df, freq)

        st_year = df.index[0].year
        en_year = df.index[-1].year

        assert isinstance(df.index, pd.DatetimeIndex)

        for yr in range(st_year, en_year + 1):

            _df = df[df.index.year == yr]

            if freq == 'yearly':
                kwargs = plot_style(_df, **kwargs)
                _df.plot(**kwargs)
                self._save_or_show(fname=f'input_{prefix}_{str(yr)}')

            elif freq == 'monthly':
                st_mon = _df.index[0].month
                en_mon = _df.index[-1].month

                for mon in range(st_mon, en_mon+1):

                    __df = _df[_df.index.month == mon]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self._save_or_show(fname=f'input_{prefix}_{str(yr)} _{str(mon)}')

            elif freq == 'weekly':
                st_week = _df.index[0].isocalendar()[1]
                en_week = _df.index[-1].isocalendar()[1]

                for week in range(st_week, en_week+1):
                    __df = _df[_df.index.week == week]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self._save_or_show(fname=f'input_{prefix}_{str(yr)} _{str(week)}')
        return

    def correlation(
            self,
            cols=None,
            remove_targets=False,
            st=None,
            en=None,
            **kwargs
    ):
        """
        Plots correlation between features.

        Arguments:
            cols : columns to use
            remove_targets :
            st :
            en :
            kwargs :

        Example:
            >>> from ai4water.eda import EDA
            >>> from ai4water.datasets import arg_beach
            >>> vis = EDA(arg_beach())
            >>> vis.correlation()
        """
        # todo, by default it is using corr_coeff, added other possible correlation methods such as Spearman rank correlation etc
        if cols is None:
            if remove_targets:
                cols = self.in_cols
            else:
                cols = self.in_cols + self.out_cols
            if isinstance(cols, dict):
                cols = None

        return self._call_method("_feature_feature_corr_df", cols=cols, st=st, en=en, **kwargs)

    def _feature_feature_corr_df(self,
                                 data,
                                 cols=None,
                                 st=None,
                                 en=None,
                                 prefix='',
                                 split=None,
                                 threshold=0,
                                 method='pearson',
                                 **kwargs
                                 ):
        """
        split : Optional[str], optional
        Type of split to be performed {None, "pos", "neg", "high", "low"}, by default None
        method : str, optional
                 {"pearson", "spearman", "kendall"}, by default "pearson"

        kwargs
            * vmax: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 or vmin <= vmax <= 1, limits the range of the cbar.
            * vmin: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 <= vmin <= 1 or vmax, limits the range of the cbar.
        To plot positive correlation only:
        feature_feature_corr_df(model.data, list(model.data.columns), split="pos")
        To plot negative correlation only
        feature_feature_corr_df(model.data, list(model.data.columns), split="neg")
        """
        plt.close('all')

        if cols is None:
            cols = data.columns.to_list()

        data = _preprocess_df(data, st, en)

        corr = data[cols].corr(method=method)

        if split == "pos":
            corr = corr.where((corr >= threshold) & (corr > 0))
        elif split == "neg":
            corr = corr.where((corr <= threshold) & (corr < 0))

        mask = np.zeros_like(corr, dtype=np.bool)

        vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
        vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)
        # width x height
        _, ax = plt.subplots(figsize=kwargs.get('figsize', (5 + len(cols)*0.25, 9 + len(cols)*0.1)))

        _kwargs = dict()
        _kwargs['annot'] = kwargs.get('annot', True if len(cols) <= 20 else False)
        _kwargs['cmap'] = kwargs.get('cmap', "BrBG")
        _kwargs['vmax'] = kwargs.get('vmax', vmax)
        _kwargs['vmin'] = kwargs.get('vmin', vmin)
        _kwargs['linewidths'] = kwargs.get('linewidths', 0.5)
        _kwargs['annot_kws'] = kwargs.get('annot_kws', {"size": 10})
        _kwargs['cbar_kws'] = kwargs.get('cbar_kws', {"shrink": 0.95, "aspect": 30})

        ax = sns.heatmap(corr, center=0, fmt=".2f", ax=ax, **_kwargs)
        ax.set(frame_on=True)

        self._save_or_show(fname=f"{split if split else ''}_feature_corr_{prefix}")

        return ax

    def plot_pcs(self, num_pcs=None, st=None, en=None, save_as_csv=False,
                 figsize=(12, 8), **kwargs):
        """Plots principle components.
        Arguments:
            num_pcs :
            st :
            en :
            save_as_csv :
            figsize :
            kwargs :will go to sns.pairplot.
        """

        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self._plot_pcs(data[self.in_cols],
                               num_pcs,
                               st=st, en=en,
                               prefix=str(idx), save_as_csv=save_as_csv,
                               hue=self.out_cols[idx], figsize=figsize, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                self._plot_pcs(data[self.in_cols], num_pcs,
                               st=st, en=en,
                               prefix=data_name, save_as_csv=save_as_csv,
                               hue=self.out_cols,
                               figsize=figsize, **kwargs)
        else:
            self._plot_pcs(self.data[self.in_cols],
                           num_pcs,
                           st=st, en=en,
                           save_as_csv=save_as_csv, hue=self.out_cols,
                           figsize=figsize, **kwargs)
        return

    def _plot_pcs(self,
                  data,
                  num_pcs,
                  st=None,
                  en=None,
                  prefix='',
                  save_as_csv=False,
                  hue=None,
                  figsize=(12, 8), **kwargs):

        data = _preprocess_df(data, st, en)

        if num_pcs is None:
            _num_pcs = int(data.shape[1]/2)
            if _num_pcs > 5 and num_pcs is None:
                num_pcs = 5
            else:
                num_pcs = _num_pcs

        if num_pcs < 1:
            print(f'{num_pcs} pcs can not be plotted because data has shape {data.shape}')
            return
        # df_pca = data[self.in_cols]
        # pca = PCA(n_components=num_pcs).fit(df_pca)
        # df_pca = pd.DataFrame(pca.transform(df_pca))

        transformer = Transformations(data=data, method='pca', n_components=num_pcs,
                                      replace_nans=True)
        df_pca = transformer.transform()

        pcs = ['pc' + str(i + 1) for i in range(num_pcs)]
        df_pca.columns = pcs

        if hue is not None and len(self.out_cols) > 0:
            if isinstance(hue, list):
                if len(hue) == 1:
                    hue = hue[0]
                else:
                    hue = None
            if hue in data:
                df_pca[hue] = data[hue]

                if df_pca[hue].isna().sum() > 0: # output columns contains nans, so don't use it as hue.
                    hue = None

        if isinstance(hue, list) and len(hue) == 0:
            hue = None

        if save_as_csv:
            df_pca.to_csv(os.path.join(self.path, f"data\\first_{num_pcs}_pcs_{prefix}"))

        plt.close('all')
        plt.figure(figsize=figsize)
        sns.pairplot(data=df_pca, vars=pcs, hue=hue, **kwargs)
        self._save_or_show(fname=f"first_{num_pcs}_pcs_{prefix}")
        return

    def grouped_scatter(
            self,
            inputs: bool = True,
            outputs: bool = True,
            cols=None,
            st=None,
            en=None,
            max_subplots: int = 8,
            **kwargs
    ):
        """Makes scatter plot for each of feature in data.

        Arguments:
            inputs :
            outputs :
            cols :
            st :
            en :
            max_subplots :
            kwargs :
        """
        fname = "scatter_plot_"

        if cols is None:
            cols = []

            if inputs:
                cols += self.in_cols
                fname += "inputs_"
            if outputs:
                cols += self.out_cols
                fname += "outptuts_"
        else:
            assert isinstance(cols, list)

        if isinstance(self.data, pd.DataFrame):
            self._grouped_scatter_plot_df(self.data[cols], max_subplots, st=st, en=en,
                                          **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    _cols = cols + [self.out_cols[idx]] if outputs else cols
                    self._grouped_scatter_plot_df(data[_cols], max_subplots,
                                                  st=st, en=en,
                                                  prefix=str(idx), **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self._grouped_scatter_plot_df(data, max_subplots,
                                                  st=st, en=en,
                                                  prefix=data_name,
                                                  **kwargs)
        return

    def _grouped_scatter_plot_df(self,
                                 data: pd.DataFrame,
                                 max_subplots: int = 10,
                                 st=None, en=None,
                                 prefix='', **kwargs):
        """
        max_subplots: int, it can be set to large number to show all the scatter
        plots on one axis.
        """
        data = data.copy()

        data = _preprocess_df(data, st, en)

        if data.shape[1] <= max_subplots:
            self._grouped_scatter_plot(data, name=f'grouped_scatter_{prefix}',  **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_subplots)
            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, st:en]
                self._grouped_scatter_plot(sub_df, name=f'grouped_scatter_{prefix}_{st}_{en}', **kwargs)
        return

    def _grouped_scatter_plot(self, df, name='grouped_scatter', **kwargs):
        plt.close('all')
        sns.set()
        sns.pairplot(df, size=2.5, **kwargs)
        self._save_or_show(fname=name)
        return

    def plot_histograms(
            self,
            st=None,
            en=None,
            cols=None,
            **kwargs
    ):
        """Plots distribution of data as histogram.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html

        Arguments:
            st : starting index of data to use
            en : end index of data to use
            cols : columns to use
            kwargs : anykeyword argument for pandas.DataFrame.hist function
        """
        return self._call_method("_plot_his_df", st=st, en=en, cols=cols, **kwargs)

    def _plot_his_df(self,
                     data: pd.DataFrame,
                     cols=None,
                     st=None,
                     en=None,
                     prefix='',
                     bins=100,
                     figsize=(20, 14),
                     **kwargs
                     ):
        """Plots histogram of one dataframe"""

        data = _preprocess_df(data, st, en, cols)

        axis = data.hist(bins=bins, figsize=figsize, **kwargs)

        self._save_or_show(fname=f"hist_{prefix}")

        return axis

    def plot_index(self, st=None, en=None, **kwargs):
        """plots the datetime index of dataframe
        """
        return self._call_method("_plot_index", st=st, en=en, **kwargs)

    def _plot_index(self,
                    index,
                    st=None,
                    en=None,
                    fname="index",
                    figsize=(10, 5),
                    label_fs=18,
                    title_fs=20,
                    leg_fs=14,
                    leg_ms=4,
                    color='r',
                    ):
        """
        Plots the index of a datafram.
        index: can be pandas dataframe or index itself. if dataframe, its index
        will be used for plotting
        """
        plt.close('all')
        if isinstance(index, pd.DataFrame):
            index = index.index

        idx = pd.DataFrame(np.ones(len(index)), index=index, columns=['Observations'])
        axis = idx.plot(linestyle='', marker='.', color=color, figsize=figsize)
        axis.legend(fontsize=leg_fs, markerscale=leg_ms)
        axis.set_xlabel(axis.get_xlabel(), fontdict={'fontsize': label_fs})
        axis.set_title("Temporal distribution of Observations", fontsize=title_fs)
        axis.get_yaxis().set_visible(False)
        self._save_or_show(fname=fname)

        return axis

    def stats(self,
              precision=3,
              inputs=True,
              outputs=True,
              st=None,
              en=None,
              out_fmt="csv",
              ):
        """Finds the stats of inputs and outputs and puts them in a json file.
        inputs: bool
        fpath: str, path like
        out_fmt: str, in which format to save. csv or json"""
        cols = []
        fname = "data_description_"
        if inputs:
            cols += self.in_cols
            fname += "inputs_"
        if outputs:
            cols += self.out_cols
            fname += "outputs_"

        fname += str(dateandtime_now())

        def save_stats(_description, _fpath):
            if self.save:
                if out_fmt == "csv":
                    pd.DataFrame.from_dict(_description).to_csv(_fpath + ".csv")
                else:
                    dict_to_file(others=_description, path=_fpath + ".json")

        description = {}
        if isinstance(self.data, pd.DataFrame):
            description = {}
            for col in cols:
                if col in self.data:
                    description[col] = ts_features(_preprocess_df(self.data[col], st, en),
                                                   precision=precision, name=col)

            save_stats(description, self.path)

        elif isinstance(self.data, list):
            description = {}

            for idx, data in enumerate(self.data):
                _description = {}

                if isinstance(data, pd.DataFrame):

                    for col in cols:
                        if col in data:
                            _description[col] = ts_features(_preprocess_df(data[col], st, en),
                                                            precision=precision, name=col)

                description['data' + str(idx)] = _description
                _fpath = os.path.join(self.path, fname + f'_{idx}')
                save_stats(_description, _fpath)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                _description = {}
                if isinstance(data, pd.DataFrame):
                    for col in data.columns:
                        _description[col] = ts_features(_preprocess_df(data[col], st, en),
                                                        precision=precision, name=col)

                description[f'data_{data_name}'] = _description
                _fpath = os.path.join(self.path, fname + f'_{data_name}')
                save_stats(_description, _fpath)
        else:
            print(f"description can not be found for data type of {self.data.__class__.__name__}")

        return description

    def box_plot(
            self,
            inputs: bool = True,
            outputs: bool = True,
            st=None,
            en=None,
            violen=False,
            normalize=True,
            cols=None,
            figsize=(12, 8),
            max_features=8,
            show_datapoints=False,
            freq=None,
            **kwargs
    ):
        """
        Plots box whister or violen plot of data.

        Arguments:
            inputs :
            outputs :
            st :
            en :
            normalize : If True, then each feature/column is rescaled between 0 and 1.
            figsize :
            freq : str, one of 'weekly', 'monthly', 'yearly'. If given, box plot
                will be plotted for these intervals.
            max_features : int, maximum number of features to appear in one plot.
            violen : bool, if True, then violen plot will be plotted else box_whisker plot
            cols : list, the name of columns from data to be plotted.
            kwargs : any args for seaborn.boxplot/seaborn.violenplot or seaborn.swarmplot.
            show_datapoints : if True, sns.swarmplot() will be plotted. Will be time
                consuming for bigger data.
        """

        axis = None
        data = self.data
        fname = "violen_plot_" if violen else "box_plot_"

        if cols is None:
            cols = []

            if inputs:
                cols += self.in_cols
                fname += "inputs_"
            if outputs:
                cols += self.out_cols
                fname += "outptuts_"
        else:
            assert isinstance(cols, list)

        if isinstance(data, list):
            for idx, d in enumerate(data):
                if isinstance(self.in_cols, dict):
                    cols_ = [item for sublist in list(self.in_cols.values()) for item in sublist]
                    _cols = []
                    for c in cols_:
                        if c in d:
                            cols_.append(c)
                else:
                    _cols = cols + [self.out_cols[idx]] if outputs else cols

                self._box_plot(d, _cols, st, en, normalize, figsize, max_features, show_datapoints, freq,
                               violen=violen,
                               prefix=str(idx),
                               **kwargs)

        elif isinstance(data, dict):
            for data_name, _data in data.items():
                self._box_plot(_data,
                               list(_data.columns),
                               st=st,
                               en=en,
                               normalize=normalize,
                               figsize=figsize,
                               max_features=max_features,
                               show_datapoints=show_datapoints,
                               freq=freq,
                               violen=violen,
                               prefix=data_name,
                               **kwargs)
        else:
            cols = cols + self.out_cols if outputs else cols
            axis = self._box_plot(data, cols, st=st, en=en,
                           normalize=normalize, figsize=figsize,
                           max_features=max_features,
                           show_datapoints=show_datapoints, freq=freq,
                           violen=violen,
                           **kwargs)
        return axis

    def _box_plot(self,
                  data,
                  cols,
                  st=None,
                  en=None,
                  normalize=True,
                  figsize=(12, 8),
                  max_features=8,
                  show_datapoints=False,
                  freq=None,
                  violen=False,
                  prefix='',
                  **kwargs):
        data = data[cols]
        axis = None

        if data.shape[1] <= max_features:
            axis = self._box_plot_df(data,
                              st=st,
                              en=en,
                              normalize=normalize, show_datapoints=show_datapoints,
                              violen=violen,
                              freq=freq,
                              prefix=f"{'violen' if violen else 'box'}_{prefix}",
                              figsize=figsize, **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_features)
            for i in range(len(tot_plots) - 1):
                _st, _en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, _st:_en]
                self._box_plot_df(sub_df,
                                  st=st,
                                  en=en,
                                  normalize=normalize, show_datapoints=show_datapoints,
                                  violen=violen,
                                  figsize=figsize,
                                  freq=freq,
                                  prefix=f"{'violen' if violen else 'box'}_{prefix}_{_st}_{_en}",
                                  **kwargs)
        return axis

    def _box_plot_df(self,
                     data,
                     st=None,
                     en=None,
                     normalize=True,
                     show_datapoints=False,
                     violen=False,
                     figsize=(12, 8),
                     prefix="box_plot",
                     freq=None,
                     **kwargs
                     ):

        data = data.copy()

        data = _preprocess_df(data, st, en)

        # if data contains duplicated columns, transformation will not work
        data = data.loc[:, ~data.columns.duplicated()]
        if normalize:
            transformer = Transformations(data=data)
            data = transformer.transform()

        if freq is not None:
            return self._box_plot_with_freq(data,
                                            freq=freq,
                                            show_datapoints=show_datapoints,
                                            figsize=figsize,
                                            violen=violen,
                                            prefix=prefix,
                                            **kwargs
                                            )

        return self.__box_plot_df(data=data,
                                  name=prefix,
                                  violen=violen,
                                  figsize=figsize,
                                  show_datapoints=show_datapoints,
                                  **kwargs)

    def __box_plot_df(self,
                      data,
                      name,
                      violen=False,
                      figsize=(12, 8),
                      show_datapoints=False,
                      **kwargs):

        plt.close('all')
        plt.figure(figsize=figsize)

        if violen:
            axis = sns.violinplot(data=data, **kwargs)
        else:
            axis = sns.boxplot(data=data, **kwargs)
            axis.set_xticklabels(list(data.columns), fontdict={'rotation': 70})

        if show_datapoints:
            sns.swarmplot(data=data)

        self._save_or_show(fname=name)

        return axis

    def _box_plot_with_freq(self,
                            data,
                            freq,
                            violen=False,
                            show_datapoints=False,
                            figsize=(12,8),
                            name='bw',
                            prefix='',
                            **kwargs
                            ):

        validate_freq(data, freq)

        st_year = data.index[0].year
        en_year = data.index[-1].year

        for yr in range(st_year, en_year + 1):

            _df = data[data.index.year == yr]

            if freq == 'yearly':
                self._box_plot_df(_df,
                                  name=f'{name}_input_{prefix}_{str(yr)}',
                                  figsize=figsize,
                                  violen=violen,
                                  show_datapoints=show_datapoints,
                                  **kwargs)

            elif freq == 'monthly':
                st_mon = _df.index[0].month
                en_mon = _df.index[-1].month

                for mon in range(st_mon, en_mon+1):

                    __df = _df[_df.index.month == mon]

                    self._box_plot_df(__df,
                                      name=f'{prefix}_{str(yr)} _{str(mon)}',
                                      where='data/monthly',
                                      figsize=figsize,
                                      violen=violen,
                                      show_datapoints=show_datapoints,
                                      **kwargs)

            elif freq == 'weekly':
                st_week = _df.index[0].isocalendar()[1]
                en_week = _df.index[-1].isocalendar()[1]

                for week in range(st_week, en_week+1):
                    __df = _df[_df.index.week == week]

                    self._box_plot_df(__df,
                                      name=f'{prefix}_{str(yr)} _{str(week)}',
                                      where='data/weely',
                                      violen=violen,
                                      figsize=figsize,
                                      show_datapoints=show_datapoints,
                                      **kwargs)
        return

    def autocorrelation(
            self,
            nlags:int=10,
            cols:Union[list]=None,
    ):
        """autocorrelation of individual features of data
        Arguments:
            nlags : number of lag steps to consider
            cols : columns to use. If not defined then all the columns are used
        """
        return self._call_method("_autocorr_df", partial=False, nlags=nlags, cols=cols)

    def partial_autocorrelation(
            self,
            nlags:int=10,
            cols:Union[list]=None,
    ):
        """Partial autocorrelation of individual features of data
        Arguments:
            nlags : number of lag steps to consider
            cols : columns to use. If not defined then all the columns are used
        """
        return self._call_method("_autocorr_df", partial=True, nlags=nlags, cols=cols)

    def _autocorr_df(
            self,
            data:pd.DataFrame,
            nlags:int,
            partial:bool=False,
            cols=None,
            fname='',
    ):
        """autocorrelation on a dataframe."""
        prefix = 'Partial' if partial else ''

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
            assert isinstance(cols, list)
            data = data[cols]

        non_nan = data.isna().sum()
        num_subplots = max(math.ceil(len(non_nan[non_nan==0])/2)*2, 1)

        if num_subplots==1:
            nrows, ncols = 1,1
        elif num_subplots == 2:
            nrows, ncols = 1, 2
        else:
            nrows, ncols = int(num_subplots/2), 2

        fig, axis = plt.subplots(nrows, ncols, figsize=(9, nrows*2),
                               sharex="all", sharey="all")
        axis = np.array(axis)  # if it is a single axis then axis.flat will not work
        for col, ax in zip(data.columns, axis.flat):

            x = data[col].values
            if np.isnan(x).sum() == 0:

                if partial:
                    _ac = pac_yw(x, nlags)
                else:
                    _ac = auto_corr(x, nlags)

                plot_autocorr(_ac, axis=ax, legend=col, show=False, legend_fs=nrows*2)
            else:
                print(f"cannot plot autocorrelation for {col} feature")

        plt.suptitle(f"{prefix} Autocorrelation",
                     fontsize=nrows*3)

        fname = f"{prefix} autocorr_{fname}"
        self._save_or_show(fname=fname)

        return axis

    def _call_method(self, method_name, *args, **kwargs):
        """calls the method with the data and args + kwargs"""

        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                getattr(self, method_name)(data, fname=str(idx), *args, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                getattr(self, method_name)(data, fname=data_name, *args, **kwargs)

        else:
            return getattr(self, method_name)(self.data, *args, **kwargs)

    def probability_plots(
            self,
            cols:Union[str, list]=None
    ):
        """
        draws prbability plot using scipy.stats.probplot
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

        """
        return self._call_method("_plot_prob_df", cols=cols)

    def _plot_prob_df(
            self,
            data:pd.DataFrame,
            cols:Union[str, list] = None,
            fname=None,
    ):
        """probability plots for one dataframe"""
        assert isinstance(data, pd.DataFrame)

        if cols is not None:
            if isinstance(cols, str):
                cols = [cols]
        else:
            cols = data.columns.to_list()

        assert isinstance(cols, list)
        data = data[cols]

        for col in data.columns:
            series = data[col]

            self._prob_plot_series(series, fname=fname)

        return

    def _prob_plot_series(
            self,
            data:Union[pd.DataFrame, pd.Series],
            fname:str = None
    ):
        """probability plots for one series."""
        if not isinstance(data, pd.Series):
            assert isinstance(data, pd.DataFrame) and data.shape[1] == 1
            data = pd.Series(data)

        if data.isna().sum() > 0:
            print(f"removing nan values from {data.name}")
            data = data.dropna()

        array = data.values

        cont_distros = {
            "norm": stats.norm(),
            "uniform": stats.uniform(),
            "semicircular": stats.semicircular(),
            "cauchy": stats.cauchy(),
            "expon": stats.expon(),
            "rayleight": stats.rayleigh(),
            "moyal": stats.moyal(),
            "arcsine": stats.arcsine(),
            "anglit": stats.anglit(),
            "gumbel_l": stats.gumbel_l(),
            "gilbrat": stats.gilbrat(),
            "levy": stats.levy(),
            "laplace": stats.laplace(),
            "bradford": stats.bradford(0.5),
            "kappa3":   stats.kappa3(1),
            "pareto":   stats.pareto(2.62)
        }

        fig, axis = plt.subplots(4,4, figsize=(10, 10))

        for (idx, rv), ax in zip(enumerate(cont_distros.values()), axis.flat):

            if isinstance(rv, str):
                _name = rv
            else:
                _name = rv.dist.name

            (osm, osr), (slope, intercept, r) = stats.probplot(array, dist=rv, plot=ax)

            h = ax.plot(osm, osr, label="bo")

            if idx%4==0:
                ax.set_ylabel("Ordered Values", fontsize=12)
            else:
                ax.set_ylabel("")

            if idx>11:
                ax.set_xlabel("Theoretical Quantiles", fontsize=12)
            else:
                ax.set_xlabel("")
            ax.set_title("")
            text = f"{_name}"

            ax.legend(h, [text], loc="best", fontsize=12,
                       fancybox=True, framealpha=0.7,
                       handlelength=0, handletextpad=0)

        plt.suptitle(data.name, fontsize=18)
        self._save_or_show(f"probplot_{data.name}_{fname}")

        return fig

    def _lag_plot_series(self, series:pd.Series, n_lags:int, figsize=None, **kwargs):

        if hasattr(n_lags, '__len__'):
            lags = np.array(n_lags)
            n_lags = len(lags)
        else:
            lags = range(1, n_lags+1)

        figsize = figsize or (5, 5 + n_lags*0.2)

        n_rows, n_cols = 1,1
        if n_lags>1:
            n_rows = (math.ceil(n_lags/2) * 2) // 2
            n_cols = 2

        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, sharex="all")

        if n_lags == 1:
            axis = np.array([axis])

        for n, ax in zip(lags, axis.flat):
            lag_plot(series, n, ax, **kwargs)

        plt.suptitle(series.name)
        self._save_or_show(fname=f"lagplot_{series.name}")

        return axis

    def _lag_plot_df(self, data:pd.DataFrame, n_lags:int, cols=None, **kwargs):

        data = _preprocess_df(data, cols=cols)

        axes = []

        for col in data.columns:
            axes.append(self._lag_plot_series(data[col], n_lags, **kwargs))

        return axes

    def lag_plot(self, n_lags:Union[int, list]=1, cols=None, figsize=None, **kwargs):
        """lag plot between series and its lags

        Arguments:
            n_lags : lag step against which to plot the data, it can be integer
                or a list of integers
            cols : columns to use
            figsize : figsize
            kwargs : any keyword arguments for axis.scatter
        """
        return self._call_method("_lag_plot_df", n_lags=n_lags, cols=cols, figsize=figsize, **kwargs)

    def plot_ecdf(
            self,
            cols,
            figsize=None,
            **kwargs
    ):
        """plots empirical cummulative distribution function

        Arguments:
            cols : columns to use
            figsize :
            kwargs : any keyword argument for axis.plot
        """
        return self._call_method("_plot_ecdf_df", cols=cols, figsize=figsize, **kwargs)

    def _plot_ecdf_df(self, data:pd.DataFrame, cols=None, figsize=None, fname=None, **kwargs):

        data = _preprocess_df(data, cols=cols)

        ncols = data.shape[1]
        n_rows, n_cols = 1, 1
        if ncols > 1:
            n_rows = (math.ceil(ncols / 2) * 2) // 2
            n_cols = 2

        figsize = figsize or (6, 5 + ncols * 0.2)

        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

        if ncols == 1:
            axis = np.array([axis])

        for col, ax in zip(data.columns, axis.flat):
            plot_ecdf(data[col], ax=ax, **kwargs)

        self._save_or_show(fname=f"ecdf_{fname}")

        return axis


def plot_ecdf(x:Union[pd.Series, np.ndarray], ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if isinstance(x, pd.Series):
        _name = x.name
        x = x.values
    else:
        assert isinstance(x, np.ndarray)
        _name = "ecdf"

    x,y = ecdf(x)
    ax.plot(x, y, label=_name, **kwargs)
    ax.legend()

    return ax


def ecdf(x:np.ndarray):
    # https://stackoverflow.com/a/37660583/5982232
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))

    return xs, ys


def lag_plot(series: pd.Series, lag: int, ax, **kwargs):

    data = series.values
    y1 = data[:-lag]
    y2 = data[lag:]

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel("y(t)")
    ax.set_ylabel(f"y(t + {lag})")
    ax.scatter(y1, y2, **kwargs)

    return ax


def set_axis_paras(axis, leg_kws, label_kws, tick_kws):
    axis.legend(**leg_kws)
    axis.set_ylabel(axis.get_ylabel(), **label_kws)
    axis.set_xlabel(axis.get_xlabel(), **label_kws)
    axis.tick_params(**tick_kws)
    return


def plot_style(df:pd.DataFrame, **kwargs):
    if 'style' not in kwargs and df.isna().sum().sum() > 0:
        kwargs['style'] = ['.' for _ in range(df.shape[1])]
    return kwargs


def validate_freq(df, freq):
    assert isinstance(df.index, pd.DatetimeIndex), "index of dataframe must be pandas DatetimeIndex"
    assert freq in ["weekly", "monthly",
                    "yearly"], f"freq must be one of {'weekly', 'monthly', 'yearly'} but it is {freq}"
    return


def _preprocess_df(df, st=None, en=None, cols=None):

    if cols is not None:
        if isinstance(cols, str):
            cols = [cols]
        df = df[cols]

    if st is None:
        st = df.index[0]
    if en is None:
        en = df.index[-1]

    if isinstance(st, int):
        df = df.iloc[st:en]
    else:
        df = df.loc[st:en]

    return df



