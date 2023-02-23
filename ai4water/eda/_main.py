
import math
import warnings
from typing import Union, List, Dict

import scipy.stats as stats

from .utils import _missing_vals
from ai4water.backend import easy_mpl as ep
from .utils import pac_yw, auto_corr, plot_autocorr
from ai4water.utils.visualizations import Plot
from ai4water.preprocessing import Transformation
from ai4water.backend import np, pd, os, plt, sns, mpl
from ai4water.utils.utils import find_tot_plots, get_nrows_ncols
from ai4water.utils.utils import dict_to_file, dateandtime_now, ts_features

ticker = mpl.ticker
create_subplots = ep.utils.create_subplots

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
    - normality_test
    - parallel_coordinates
    - show_unique_vals

    Example:
        >>> from ai4water.datasets import busan_beach
        >>> eda = EDA(data=busan_beach())
        >>> eda()  # to plot all available plots with single line
    """

    def __init__(
            self,
            data: Union[pd.DataFrame, List[pd.DataFrame], Dict, np.ndarray],
            in_cols=None,
            out_cols=None,
            path=None,
            dpi=300,
            save=True,
            show=True,
    ):
        """

        Arguments
        ---------
            data : DataFrame, array, dict, list
                either a dataframe, or list of dataframes or a dictionary whose
                values are dataframes or a numpy arrays
            in_cols : str, list, optional
                columns to consider as input features
            out_cols : str, optional
                columns to consider as output features
            path : str, optional
                the path where to save the figures. If not given, plots will be
                saved in 'data' folder in current working directory.
            save : bool, optional
                whether to save the plots or not
            show : bool, optional
                whether to show the plots or not
            dpi : int, optional
                the resolution with which to save the image
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif isinstance(data, pd.Series):
            data = pd.DataFrame(data, columns=[data.name], index=data.index)

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
            elif isinstance(self.data, pd.Series):
                x = self.data.name
            else:
                raise ValueError(f"unsupported type of {self.data.__class__.__name__}")
        self._in_cols = x

    @property
    def out_cols(self):
        return self._out_cols

    @out_cols.setter
    def out_cols(self, x):
        if x is None:
            if isinstance(self.data, pd.DataFrame) or isinstance(self.data, pd.Series):
                x = []
            else:
                raise ValueError
        self._out_cols = x

    def _save_or_show(self, fname, dpi=None):

        return self.save_or_show(where='data', fname=fname, show=self.show, dpi=dpi,
                                 close=False)

    def __call__(self,
                 methods: Union[str, list] = 'all',
                 cols=None,
                 ):
        """Shortcut to draw maximum possible plots.

        Arguments
        ---------
            methods : str, list, optional
                the methods to call. If 'all', all available methods will be called.
            cols : str, list, optional
                columns to use for plotting. If None, all columns will be used.
        """
        all_methods = [
            'heatmap', 'plot_missing', 'plot_histograms', 'plot_data',
            'plot_index', 'stats', 'box_plot',
            'autocorrelation', 'partial_autocorrelation',
            'lag_plot', 'plot_ecdf'
        ]

        if isinstance(self.data, pd.DataFrame) and self.data.shape[-1] > 1:
            all_methods = all_methods + [  # 'plot_pcs',
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

    def heatmap(self,
                st=None,
                en=None,
                cols=None,
                figsize: tuple = None,
                **kwargs):
        """
        Plots data as heatmap which depicts missing values.

        Arguments
        ---------
            st : int, str, optional
                starting row/index in data to be used for plotting
            en : int, str, optional
                end row/index in data to be used for plotting
            cols : str, list
                columns to use to draw heatmap
            figsize : tuple, optional
                figure size
            **kwargs :
                Keyword arguments for sns.heatmap
        Return
        ------
            None

        Example
        -------
            >>> from ai4water.datasets import busan_beach
            >>> data = busan_beach()
            >>> vis = EDA(data)
            >>> vis.heatmap()
        """
        if sns is None:
            raise SeabornNotFound()

        return self._call_method('_heatmap_df',
                                 cols=cols,
                                 st=st,
                                 en=en,
                                 figsize=figsize,
                                 **kwargs)

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
            figsize= None,
            **kwargs
    ):
        """
        Plots a heat map of a dataframe. Helpful to show where missing values are
        located in a dataframe.

        Arguments:
            data : pd.DataFrame,
            cols : list, columns from data to be used.
            st : starting row/index in data to be used for plotting
            en : end row/index in data to be used for plotting
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
            "ytick_labels_fs": 20
        }
        for k in _kwargs.keys():
            if k in kwargs:
                _kwargs[k] = kwargs.pop(k)

        show_time_on_yaxis = False
        if isinstance(data.index, pd.DatetimeIndex):
            show_time_on_yaxis = True

        _, axis = plt.subplots(figsize=figsize or (5 + len(cols)*0.25, 10 + len(cols)*0.1))
        # ax2 - Heatmap
        sns.heatmap(data[cols].isna(), cbar=False, cmap="binary", ax=axis, **kwargs)

        axis.set_yticks(axis.get_yticks()[0::5].astype('int'))

        if show_time_on_yaxis:
            index = pd.date_range(data.index[0], data.index[-1],
                                  periods=len(axis.get_yticks()))
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

        Arguments
        ---------
            cols : list, str, optional
                columns to be used.
            st : int, str, optional
                starting row/index in data to be used for plotting
            en : int, str, optional
                end row/index in data to be used for plotting
            **kwargs :
                Keyword Args such as figsize

        Example
        -------
            >>> from ai4water.datasets import busan_beach
            >>> data = busan_beach()
            >>> vis = EDA(data)
            >>> vis.plot_missing()
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
            ax1 = ep.bar_chart(labels=list(data.columns),
                            values=np.round(mv_cols_ratio * 100, 2),
                            orient='v',
                            show=False,
                            ax=ax1)

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
                    rotation="horizontal",
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
            max_cols_in_plot: int = 10,
            ignore_datetime_index=False,
            **kwargs
    ):
        """
        Plots the data.

        Arguments
        ---------
            st : int, str, optional
                starting row/index in data to be used for plotting
            en : int, str, optional
                end row/index in data to be used for plotting
            cols : str, list, optional
                columns in data to consider for plotting
            max_cols_in_plot : int, optional
                Maximum number of columns in one plot. Maximum number of plots
                depends upon this value and number of columns
                in data.
            freq : str, optional
                one of 'daily', 'weekly', 'monthly', 'yearly', determines
                interval of plot of data. It is valid for only time-series data.
            ignore_datetime_index : bool, optional
                only valid if dataframe's index is `pd.DateTimeIndex`. In such a case, if
                you want to ignore time index on x-axis, set this to True.
            **kwargs :
                ary arguments for pandas plot method_

        .. _method:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html

        Example
        -------
            >>> from ai4water.datasets import busan_beach
            >>> eda = EDA(busan_beach())
            >>> eda.plot_data(subplots=True, figsize=(12, 14), sharex=True)
            >>> eda.plot_data(freq='monthly', subplots=True, figsize=(12, 14), sharex=True)

        """
        return self._call_method("_plot_df",
                                 st=st,
                                 en=en,
                                 cols=cols,
                                 freq=freq,
                                 max_cols_in_plot=max_cols_in_plot,
                                 ignore_datetime_index=ignore_datetime_index,
                                 **kwargs)

    def _plot_df(self,
                 df,
                 st=None,
                 en=None,
                 cols=None,
                 freq=None,
                 max_cols_in_plot=10,
                 prefix='',
                 leg_kws=None,
                 label_kws=None,
                 tick_kws=None,
                 ignore_datetime_index=False,
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

        df = _preprocess_df(df, st, en, cols, ignore_datetime_index=ignore_datetime_index)

        if df.shape[1] <= max_cols_in_plot:

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
            tot_plots = find_tot_plots(df.shape[1], max_cols_in_plot)

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
                    self._plot_df_with_freq(sub_df, freq,
                                            prefix=f'{prefix}_{st}_{en}',
                                            **kwargs)
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

    def parallel_corrdinates(
            self,
            cols=None,
            st=None,
            en=100,
            color=None,
            **kwargs
    ):
        """
        Plots data as parallel coordinates.

        Arguments
        ----------
            st :
                start of data to be considered
            en :
                end of data to be considered
            cols :
                columns from data to be considered.
            color :
                color or colormap to be used.
            **kwargs :
                any additional keyword arguments to be passed to easy_mpl.parallel_coordinates_

        .. _easy_mpl.parallel_coordinates:
            https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.parallel_coordinates

        """
        return self._call_method(
            "_pcorrd_df",
            cols=cols,
            st=st,
            en=en,
            color=color,
            **kwargs
        )

    def _pcorrd_df(self,
                   data,
                   st=None,
                   en=100,
                   cols=None,
                   color=None,
                   prefix="",
                   **kwargs):
        data = _preprocess_df(data, st, en, cols)
        if data.isna().sum().sum() > 0:
            warnings.warn("Dropping rows from data which contain nans.")
            data = data.dropna()
        if data.shape[0]>1:
            categories = None
            if self.out_cols and len(self.out_cols)==1:
                out_col = self.out_cols[0]
                if out_col in data:
                    categories = data.pop(out_col)
                #else:
                    ... # todo categories = self.data[out_col]
            ep.parallel_coordinates(data, cmap=color, categories=categories,
                                 show=False, **kwargs)
            return self._save_or_show(fname=f"parallel_coord_{prefix}")
        else:
            warnings.warn("""
            Not plotting parallel_coordinates because number of rows are below 2.""")

    def normality_test(
            self,
            method="shapiro",
            cols=None,
            st=None,
            en=None,
            orientation="h",
            color=None,
            figsize: tuple = None,
    ):
        """plots the statistics of nromality test as bar charts. The statistics
        for each feature are calculated either Shapiro-wilke_
        test or Anderson-Darling test][] or Kolmogorov-Smirnov test using
        scipy.stats.shapiro or scipy.stats.anderson functions respectively.

        Arguments
        ---------
            method :
                either "shapiro" or "anderson", or "kolmogorov" default is "shapiro"
            cols :
                columns to use
            st : optional
                start of data
            en : optional
                end of data to use
            orientation : optional
                orientation of bars
            color :
                color to use
            figsize : tuple, optional
                figure size (width, height)

        Example
        -------
            >>> from ai4water.eda import EDA
            >>> from ai4water.datasets import busan_beach
            >>> eda = EDA(data=busan_beach())
            >>> eda.normality_test()

        .. _Shapiro-wilke:
            https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
        """
        return self._call_method(
            "_normality_test_df",
            method=method,
            cols=cols,
            st=st,
            en=en,
            orientation=orientation,
            color=color,
            figsize=figsize
        )

    def _normality_test_df(
            self, data, cols=None, st=None, en=None,
            method="shapiro",
            orientation="h",
            prefix="",
            color=None,
            figsize=None,
    ):
        """calculates normality test for each column of a DataFrame"""
        assert method in ("shapiro", "anderson", "kolmogorov")
        data = _preprocess_df(data, st, en, cols)
        ranks = []
        # calculate stats for each column
        for col in data.columns:
            x = data[col].dropna().values
            if method=="shapiro":
                s, _ = stats.shapiro(x)
            elif method == "kolmogorov":
                s, _ = stats.kstest(x, "norm")
            else:
                s, _, _ = stats.anderson(x, "norm")

            ranks.append(s)

        _, ax = plt.subplots(figsize=figsize)

        ep.bar_chart(labels=data.columns.tolist(),
                  values=ranks,
                  orient=orientation,
                  show=False,
                  sort=True,
                  color=color,
                  ax=ax
                  )
        return self._save_or_show(fname=f"shapiro_normality_test_{prefix}")

    def correlation(
            self,
            remove_targets=False,
            st=None,
            en=None,
            cols = None,
            method: str = "pearson",
            split: str = None,
            **kwargs
    ):
        """
        Plots correlation between features.

        Arguments
        ---------
            remove_targets : bool, optional
                whether to remove the output/target column or not
            st :
                starting row/index in data to be used for plotting
            en :
                end row/index in data to be used for plotting
            cols :
                columns to use
            method : str, optional
                     {"pearson", "spearman", "kendall", "covariance"}, by default "pearson"
            split : str
                To plot only positive correlations, set it to "pos" or to plot
                only negative correlations, set it to "neg".
            **kwargs : keyword Args
                Any additional keyword arguments for seaborn.heatmap

        Example
        -------
            >>> from ai4water.eda import EDA
            >>> from ai4water.datasets import busan_beach
            >>> vis = EDA(busan_beach())
            >>> vis.correlation()
        """
        # todo, by default it is using corr_coeff, added other possible correlation methods such as
        #  rank correlation etc
        if cols is None:
            if remove_targets:
                cols = self.in_cols
            else:
                cols = self.in_cols + self.out_cols
            if isinstance(cols, dict):
                cols = None

        if sns is None:
            raise SeabornNotFound()

        return self._call_method("_feature_feature_corr_df",
                                 cols=cols,
                                 st=st,
                                 en=en,
                                 method=method,
                                 split=split,
                                 **kwargs)

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
        Type of split to be performed {None, "pos", "neg", "high", "low"}, by default
            None
        method : str, optional
                 {"pearson", "spearman", "kendall"}, by default "pearson"

        kwargs
            * vmax: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 or vmin <= vmax <= 1, limits the range of the cbar.
            * vmin: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 <= vmin <= 1 or vmax, limits the range of the cbar.
        """
        plt.close('all')

        if cols is None:
            cols = data.columns.to_list()

        data = _preprocess_df(data, st, en)

        if method == "covariance":
            corr = np.cov(data[cols].values.transpose())
            corr = pd.DataFrame(corr, columns=cols)
        else:
            corr = data[cols].corr(method=method)

        if split == "pos":
            corr = corr.where((corr >= threshold) & (corr > 0))
        elif split == "neg":
            corr = corr.where((corr <= threshold) & (corr < 0))

        mask = np.zeros_like(corr, dtype=np.bool)

        vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
        vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)

        figsize = (5 + len(cols)*0.25, 9 + len(cols)*0.1)
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
        # width x height
        _, ax = plt.subplots(figsize=figsize)

        _kwargs = dict(
            annot= True if len(cols) <= 20 else False,
            cmap="BrBG",
            vmax=vmax,
            vmin=vmin,
            linewidths=0.5,
            annot_kws={"size": 10},
            cbar_kws={"shrink": 0.95, "aspect": 30},
            fmt='.2f',
            center=0
        )

        if kwargs:
            # pass any keyword argument provided by the user to sns.heatmap
            _kwargs.update(kwargs)

        ax = sns.heatmap(corr, ax=ax, **_kwargs)
        ax.set(frame_on=True)

        self._save_or_show(fname=f"{split if split else ''}_feature_corr_{prefix}")

        return ax

    def plot_pcs(self, num_pcs=None, st=None, en=None, save_as_csv=False,
                 figsize=(12, 8), **kwargs):
        """Plots principle components.

        Arguments
        ---------
            num_pcs :
            st : starting row/index in data to be used for plotting
            en : end row/index in data to be used for plotting
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

        transformer = Transformation(data=data, method='pca', n_components=num_pcs,
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
                # output columns contains nans, so don't use it as hue.
                if df_pca[hue].isna().sum() > 0:
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
            cols=None,
            st=None,
            en=None,
            max_subplots: int = 8,
            **kwargs
    ):
        """Makes scatter plot for each of feature in data.

        Arguments
        ----------
            st :
                starting row/index in data to be used for plotting
            en :
                end row/index in data to be used for plotting
            cols :
            max_subplots : int, optional
                it can be set to large number to show all the scatter plots on one
                axis.
            kwargs :
                keyword arguments for sns.pariplot
        """
        if sns is None:
            raise SeabornNotFound()

        self._call_method('_grouped_scatter_plot_df',
                          max_subplots=max_subplots,
                          cols=cols,
                          st=st,
                          en=en,
                          **kwargs)
        return

    def _grouped_scatter_plot_df(
            self,
            data: pd.DataFrame,
            max_subplots: int = 10,
            st=None,
            en=None,
            cols = None,
            prefix='',
            **kwargs):
        """
        max_subplots: int, it can be set to large number to show all the scatter
        plots on one axis.
        """
        data = data.copy()

        data = _preprocess_df(data, st, en, cols=cols)

        if data.shape[1] <= max_subplots:
            self._grouped_scatter_plot(data, name=f'grouped_scatter_{prefix}',
                                       **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_subplots)
            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, st:en]
                self._grouped_scatter_plot(sub_df,
                                           name=f'grouped_scatter_{prefix}_{st}_{en}',
                                           **kwargs)
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
            max_subplots: int = 40,
            figsize: tuple = (20, 14),
            **kwargs
    ):
        """Plots distribution of data as histogram_.


        Arguments
        ---------
            st :
                starting index of data to use
            en :
                end index of data to use
            cols :
                columns to use
            max_subplots : int, optional
                maximum number of subplots in one figure
            figsize :
                figure size
            **kwargs : anykeyword argument for pandas.DataFrame.hist function

        .. _histogram:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
        """
        return self._call_method("_plot_hist_df", st=st, en=en, cols=cols,
                                 figsize=figsize,
                                 max_subplots=max_subplots,
                                 **kwargs)

    def _plot_hist_df(self,
                     data: pd.DataFrame,
                     cols=None,
                     st=None,
                     en=None,
                     prefix='',
                     bins=100,
                     figsize=(20, 14),
                     max_subplots: int = 40,
                     **kwargs
                     ):
        """Plots histogram of one dataframe"""

        data = _preprocess_df(data, st, en, cols)

        if data.shape[1] <= max_subplots:
            return self._hist_df(data, bins, figsize, prefix, **kwargs)

        tot_plots = find_tot_plots(data.shape[1], max_subplots)
        for i in range(len(tot_plots) - 1):
            st, en = tot_plots[i], tot_plots[i + 1]
            self._hist_df(data.iloc[:, st:en],
                          bins, figsize,
                          prefix=f'hist_{prefix}_{i}_{st}_{en}',
                          **kwargs)
        return

    def _hist_df(self, data, bins, figsize, prefix, **kwargs):
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

        idx = pd.DataFrame(np.ones(len(index)), index=index,
                           columns=['Observations'])
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
                    description[col] = ts_features(
                        _preprocess_df(self.data[col],
                                       st, en),
                        precision=precision, name=col)

            save_stats(description, self.path)

        elif isinstance(self.data, list):
            description = {}

            for idx, data in enumerate(self.data):
                _description = {}

                if isinstance(data, pd.DataFrame):

                    for col in cols:
                        if col in data:
                            _description[col] = ts_features(
                                _preprocess_df(data[col], st, en),
                                precision=precision, name=col)

                description['data' + str(idx)] = _description
                _fpath = os.path.join(self.path, fname + f'_{idx}')
                save_stats(_description, _fpath)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                _description = {}
                if isinstance(data, pd.DataFrame):
                    for col in data.columns:
                        _description[col] = ts_features(
                            _preprocess_df(data[col], st, en),
                            precision=precision, name=col)

                description[f'data_{data_name}'] = _description
                _fpath = os.path.join(self.path, fname + f'_{data_name}')
                save_stats(_description, _fpath)
        else:
            print(f"description can not be found for data type of {self.data.__class__.__name__}")

        return description

    def box_plot(
            self,
            st=None,
            en=None,
            cols: Union[list, str] = None,
            violen=False,
            normalize=True,
            figsize=(12, 8),
            max_features=8,
            show_datapoints=False,
            freq=None,
            **kwargs
    ):
        """
        Plots box whisker or violen plot of data.

        Arguments
        ---------
            st : optional
                starting row/index in data to be used for plotting
            en : optional
                end row/index in data to be used for plotting
            cols : list,
                the name of columns from data to be plotted.
            normalize :
                If True, then each feature/column is rescaled between 0 and 1.
            figsize :
                figure size
            freq : str,
                one of 'weekly', 'monthly', 'yearly'. If given, box plot will be
                plotted for these intervals.
            max_features : int,
                maximum number of features to appear in one plot.
            violen : bool,
                if True, then violen plot will be plotted else box_whisker plot
            show_datapoints : bool
                if True, sns.swarmplot() will be plotted. Will be time
                consuming for bigger data.
            **kwargs :
                any args for seaborn.boxplot/seaborn.violenplot or seaborn.swarmplot.
        """
        if sns is None:
            raise SeabornNotFound()

        return self._call_method("_box_plot",
                                 st=st, en=en, cols=cols,
                                 normalize=normalize,
                                 max_features=max_features,
                                 figsize=figsize,
                                 show_datapoints=show_datapoints,
                                 freq=freq,
                                 #prefix=fname,
                                 violen=violen,
                                 **kwargs)

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
        data = _preprocess_df(data, st, en, cols)
        axis = None

        if data.shape[1] <= max_features:
            axis = self._box_plot_df(data,
                                     normalize=normalize,
                                     show_datapoints=show_datapoints,
                                     violen=violen,
                                     freq=freq,
                                     prefix=f"{'violen' if violen else 'box'}_{prefix}",
                                     figsize=figsize,
                                     **kwargs
                                     )
        else:
            tot_plots = find_tot_plots(data.shape[1], max_features)
            for i in range(len(tot_plots) - 1):
                _st, _en = tot_plots[i], tot_plots[i + 1]
                self._box_plot_df(data.iloc[:, _st:_en],
                                  normalize=normalize,
                                  show_datapoints=show_datapoints,
                                  violen=violen,
                                  figsize=figsize,
                                  freq=freq,
                                  prefix=f"{'violen' if violen else 'box'}_{prefix}_{_st}_{_en}",
                                  **kwargs)
        return axis

    def _box_plot_df(self,
                     data,
                     normalize=True,
                     show_datapoints=False,
                     violen=False,
                     figsize=(12, 8),
                     prefix="box_plot",
                     freq=None,
                     **kwargs
                     ):

        data = data.copy()

        # if data contains duplicated columns, transformation will not work
        data = data.loc[:, ~data.columns.duplicated()]
        if normalize:
            transformer = Transformation()
            data = transformer.fit_transform(data)

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
                            figsize=(12, 8),
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
            n_lags: int = 10,
            cols: Union[list, str] = None,
            figsize: tuple = None,
    ):
        """autocorrelation of individual features of data

        Arguments
        ---------
            n_lags : int, optional
                number of lag steps to consider
            cols : str, list, optional
                columns to use. If not defined then all the columns are used
            figsize : tuple, optional
                figure size
        """
        return self._call_method("_autocorr_df", partial=False,
                                 n_lags=n_lags,
                                 cols=cols,
                                 figsize=figsize
                                 )

    def partial_autocorrelation(
            self,
            n_lags: int = 10,
            cols: Union[list, str] = None,
    ):
        """Partial autocorrelation of individual features of data

        Arguments
        ---------
            n_lags : int, optional
                number of lag steps to consider
            cols : str, list, optional
                columns to use. If not defined then all the columns are used
        """
        return self._call_method("_autocorr_df", partial=True, n_lags=n_lags,
                                 cols=cols)

    def _autocorr_df(
            self,
            data: pd.DataFrame,
            n_lags: int,
            partial: bool = False,
            cols=None,
            figsize=None,
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
        num_subplots = max(math.ceil(len(non_nan[non_nan == 0])/2)*2, 1)

        fig, axis = create_subplots(naxes=num_subplots, figsize=figsize,
                                    sharex=True, sharey=True
                                    )

        axis = np.array(axis)  # if it is a single axis then axis.flat will not work

        nrows = axis.shape[0]

        for col, ax in zip(data.columns, axis.flat):

            x = data[col].values
            if np.isnan(x).sum() == 0:

                if partial:
                    _ac = pac_yw(x, n_lags)
                else:
                    _ac = auto_corr(x, n_lags)

                plot_autocorr(_ac, axis=ax, legend=col, show=False,
                              legend_fs=nrows*1.5)
            else:
                print(f"cannot plot autocorrelation for {col} feature")

        plt.suptitle(f"{prefix} Autocorrelation",
                     fontsize=nrows*2)

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
            cols: Union[str, list] = None
    ):
        """
        draws prbability plot using scipy.stats.probplot_ . See `scipy distributions`_

        .. _scipy.stats.probplot:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html

        .. _scipy distributions:
            https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        """
        return self._call_method("_plot_prob_df", cols=cols)

    def _plot_prob_df(
            self,
            data: pd.DataFrame,
            cols: Union[str, list] = None,
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
            data: Union[pd.DataFrame, pd.Series],
            fname: str = None
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

        fig, axis = plt.subplots(4, 4, figsize=(10, 10))

        for (idx, rv), ax in zip(enumerate(cont_distros.values()), axis.flat):

            if isinstance(rv, str):
                _name = rv
            else:
                _name = rv.dist.name

            (osm, osr), (slope, intercept, r) = stats.probplot(array, dist=rv,
                                                               plot=ax)

            h = ax.plot(osm, osr, label="bo")

            if idx % 4 == 0:
                ax.set_ylabel("Ordered Values", fontsize=12)
            else:
                ax.set_ylabel("")

            if idx > 11:
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

    def _lag_plot_series(self, series: pd.Series, n_lags: int, figsize=None,
                         **kwargs):

        if hasattr(n_lags, '__len__'):
            lags = np.array(n_lags)
            n_lags = len(lags)
        else:
            lags = range(1, n_lags+1)

        figsize = figsize or (5, 5 + n_lags*0.2)

        n_rows, n_cols = 1, 1
        if n_lags > 1:
            n_rows = (math.ceil(n_lags/2) * 2) // 2
            n_cols = 2

        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize,
                                 sharex="all")

        if n_lags == 1:
            axis = np.array([axis])

        for n, ax in zip(lags, axis.flat):
            lag_plot(series, n, ax, **kwargs)

        plt.suptitle(series.name)
        self._save_or_show(fname=f"lagplot_{series.name}")

        return axis

    def _lag_plot_df(self, data: pd.DataFrame, n_lags: int, cols=None, **kwargs):

        data = _preprocess_df(data, cols=cols)

        axes = []

        for col in data.columns:
            axes.append(self._lag_plot_series(data[col], n_lags, **kwargs))

        return axes

    def lag_plot(
            self,
            n_lags: Union[int, list] = 1,
            cols=None,
            figsize=None,
            **kwargs):
        """lag plot between an array and its lags

        Arguments
        ---------
            n_lags :
                lag step against which to plot the data, it can be integer
                or a list of integers
            cols :
                columns to use
            figsize :
                figure size
            kwargs : any keyword arguments for axis.scatter
        """
        return self._call_method("_lag_plot_df", n_lags=n_lags, cols=cols,
                                 figsize=figsize, **kwargs)

    def plot_ecdf(
            self,
            cols=None,
            figsize=None,
            **kwargs
    ):
        """plots empirical cummulative distribution function

        Arguments
        ---------
            cols :
                columns to use
            figsize :
            kwargs :
                any keyword argument for axis.plot
        """
        return self._call_method("_plot_ecdf_df", cols=cols, figsize=figsize,
                                 **kwargs)

    def _plot_ecdf_df(self, data: pd.DataFrame, cols=None, figsize=None,
                      fname=None, **kwargs):

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

    def show_unique_vals(
            self,
            threshold: int = 10,
            st = None,
            en = None,
            cols = None,
            max_subplots: int = 9,
            figsize: tuple = None,
            **kwargs
    ):
        """
        Shows percentage of unique/categorical values in data. Only those columns
        are used in which unique values are below threshold.

        Arguments
        ----------
            threshold : int, optional
            st : int, str, optional
            en : int, str, optional
            cols : str, list, optional
            max_subplots : int, optional
            figsize : tuple, optional
            **kwargs :
                Any keyword arguments for `easy_mpl.pie <https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.pie>`_

        """
        return self._call_method('_pie_df',
                                 threshold=threshold,
                                 st=st, en=en, cols=cols,
                                 max_subplots=max_subplots,
                                 figsize=figsize,
                                 **kwargs)

    def _pie_df(self, data,
                threshold, st, en, cols, 
                max_subplots=9,
                fname="",
                **kwargs):
        data = _preprocess_df(data, st, en, cols)
    
        if data.shape[1] < max_subplots:
            self._pie(data, 
                threshold = threshold,
                fname=fname, 
                **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_subplots)

            for i in range(len(tot_plots) - 1):
                _st, _en = tot_plots[i], tot_plots[i + 1]
                self._pie(data.iloc[:, _st:_en], threshold=threshold, fname=fname,
                        **kwargs)

        return

    def _pie(self, data, fname="", figsize=None, threshold=10, **kwargs):

        fractions = {}
        for col in data.columns:
            fracts = data[col].value_counts(normalize=True).values
            if len(fracts) <= threshold:
                fractions[col] = fracts
            else:
                print(f"Ignoring {col} as it contains {len(fracts)} unique values")

        if len(fractions) > 0:
            nrows, ncols = get_nrows_ncols(3, len(fractions))
            _, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize or (12, 12))

            if isinstance(axis, plt.Axes):
                axis = np.array([axis])

            for col, ax in zip(fractions.keys(), axis.flat):            
                
                ep.pie(fractions[col], ax=ax, show=False, **kwargs)
                
            self._save_or_show(fname=f"pie_{fname}")

        return


def plot_ecdf(x: Union[pd.Series, np.ndarray], ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if isinstance(x, pd.Series):
        _name = x.name
        x = x.values
    else:
        assert isinstance(x, np.ndarray)
        _name = "ecdf"

    x, y = ecdf(x)
    ax.plot(x, y, label=_name, **kwargs)
    ax.legend()

    return ax


def ecdf(x: np.ndarray):
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


def plot_style(df: pd.DataFrame, **kwargs):
    if 'style' not in kwargs and df.isna().sum().sum() > 0:
        kwargs['style'] = ['.' for _ in range(df.shape[1])]
    return kwargs


def validate_freq(df, freq):
    assert isinstance(df.index, pd.DatetimeIndex), """
        index of dataframe must be pandas DatetimeIndex"""
    assert freq in ["weekly", "monthly","yearly"], f"""
        freq must be one of {'weekly', 'monthly', 'yearly'} but it is {freq}"""
    return


def _preprocess_df(df:pd.DataFrame, st=None, en=None, cols=None,
                   ignore_datetime_index=False):

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

    if ignore_datetime_index:
        df = df.reset_index(drop=True)

    return df


class SeabornNotFound(Exception):
    def __str__(self):
        return """
        You must have seaborn library installed.
        Please install seaborn using 'pip install seaborn'
        """
