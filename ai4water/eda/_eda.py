import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from ai4water.utils.utils import _missing_vals
from ai4water.utils.utils import find_tot_plots
from ai4water.pre_processing import Transformations
from ai4water.utils.visualizations import Plot


class EDA(Plot):
    """Performns a comprehensive exploratory data analysis on a tabular data"""
    def __init__(
            self,
            data,
            in_cols=None,
            out_cols=None,
            path=None
    ):
        self.data = data
        self.in_cols = in_cols,
        self.out_cols = out_cols

        super().__init__(path)

    def heatmap(self, cols=None, **kwargs):
        """
        Plots a heatmap.
        :param cols:
        :param kwargs:
        :return:
        Examples:
        >>>vis = EDA(data)
        >>>vis.heatmap(save=False)
        """
        if isinstance(self.data, pd.DataFrame):
            self.heatmap_df(self.data, cols=cols, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    self.heatmap_df(data, cols=cols[idx] if isinstance(cols, list) else None,
                                       fname=f"data_heatmap_{idx}", **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    _cols = cols[data_name] if cols is not None else None
                    self.heatmap_df(data, _cols, fname=data_name, **kwargs)
        return

    def heatmap_df(
            self,
            data:pd.DataFrame,
            cols=None,
            spine_color: str = "#EEEEEE",
            save=True,
            title=None,
            title_fs=16,
            fname="",
            **kwargs
    ):
        """
        plots a heat map of a dataframe. Helpful to show where missing values are located in a dataframe.
        :param data: pd.DataFrame,
        :param cols: list, columns from data to be used.
        :param spine_color:
        :param save: bool
        :param title: str, title of the plot
        :param title_fs: int, font size of title
        :param fname: str, name of saved file, only valid if save is True.
        :param kwargs: following kwargs are allowed:
            xtick_labels_fs, 12
            ytick_labels_fs, 20
            figsize: tuple
            any additional keyword argument will be passed to sns.heatmap

        :return:
        """
        if cols is None:
            cols = data.columns
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

        fig, axis = plt.subplots(figsize=_kwargs['figsize'])
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

        return self.save_or_show(save=save, fname=fname+'_heat_map', where='data', dpi=500)


    def plot_missing(self, save:bool=True, cols=None, **kwargs):
        """
        cols: columns to be used.
        save: if False, plot will be shown and not plotted.
        Examples:
        >>>vis = EDA(data)
        >>>vis.plot_missing(save=False)
        """
        if isinstance(self.data, pd.DataFrame):
            self.plot_missing_df(self.data, cols=cols, save=save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                _cols = cols[idx] if isinstance(cols, list) else None
                self.plot_missing_df(data, cols=None, fname=str(idx), save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    _cols = cols[data_name] if cols else None
                    self.plot_missing_df(data, cols=_cols, fname=data_name, save=save, **kwargs)
        return

    def plot_missing_df(self,
                        data:pd.DataFrame,
                        cols=None,
                        fname:str='',
                        save:bool=True,
                        **kwargs):
        """
        kwargs:
            xtick_labels_fs
            ytick_labels_fs
            figsize
            any other keyword argument will be passed to sns.barplot()
        """
        if cols is None:
            cols = data.columns
        data = data[cols]
        # Identify missing values
        mv_total, mv_rows, mv_cols, _, mv_cols_ratio = _missing_vals(data).values()

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
            ax1 = sns.barplot(x=list(data.columns), y=np.round(mv_cols_ratio * 100, 2), ax=ax1, **kwargs)

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
            self.save_or_show(save=save, fname=fname+'_missing_vals', where='data', dpi=500)
        return


    def plot_data(self, save=True, freq=None, cols=None, max_subplots=10, **kwargs):
        """
        :param save:
        :param max_subplots: int, number of subplots within one plot. Each feature will be shown in a separate subplot.
        :param kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
        :param freq: str, one of 'daily', 'weekly', 'monthly', 'yearly', determines interval of plot of data. It is
                     valid for only time-series data.
        :param cols: columns in self.data to plot
        :return:

        ------
        examples:
        >>>eda = EDA()
        >>>eda.plot_data(subplots=True, figsize=(12, 14), sharex=True)
        >>>eda.plot_data(freq='monthly', subplots=True, figsize=(12, 14), sharex=True)
        """
        # TODO, this method should be available from `model` as well
        if isinstance(self.data, pd.DataFrame):
            self.plot_df(self.data, cols=cols, save=save, freq=freq, max_subplots=max_subplots, **kwargs)

        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self.plot_df(data, cols=cols[idx] if isinstance(cols, list) else None,
                             save=save, freq=freq, prefix=str(idx), max_subplots=max_subplots, **kwargs)
        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.plot_df(data, cols=cols, prefix=data_name, save=save, freq=freq, max_subplots=max_subplots,
                                 **kwargs)
        return

    def plot_df(self, df, cols=None, save=True, freq=None, max_subplots=10,
                prefix='',
                leg_kws=None,
                label_kws=None,
                tick_kws=None,
                **kwargs):
        """Plots each columns of dataframe and saves it if `save` is True.
         max_subplots: determines how many sub_plots are to be plotted within one plot. If dataframe contains columns
         greater than max_subplots, a separate plot will be generated for remaining columns.
         """

        assert isinstance(df, pd.DataFrame)
        if leg_kws is None:
            leg_kws = {'fontsize': 14}
        if label_kws is None:
            label_kws = {'fontsize': 14}
        if tick_kws is None:
            tick_kws = {'axis':"both", 'which':'major', 'labelsize':12}

        if cols is None:
            cols = list(df.columns)
        df = df[cols]

        if df.shape[1] <= max_subplots:

            if freq is None:
                kwargs = plot_style(df, **kwargs)
                axis = df.plot(**kwargs)
                if isinstance(axis, np.ndarray):
                    for ax in axis:
                        set_axis_paras(ax, leg_kws, label_kws, tick_kws)
                else:
                    set_axis_paras(axis, leg_kws, label_kws, tick_kws)

                self.save_or_show(save=save, fname=f"input_{prefix}",  where='data')
            else:
                self.plot_df_with_freq(df, freq, save, **kwargs)
        else:
            tot_plots = find_tot_plots(df.shape[1], max_subplots)

            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = df.iloc[:, st:en]

                if freq is None:
                    kwargs = plot_style(sub_df, **kwargs)
                    axis = sub_df.plot(**kwargs)
                    for ax in axis:
                        ax.legend(**leg_kws)
                        ax.set_ylabel(ax.get_ylabel(), **label_kws)
                        ax.set_xlabel(ax.get_xlabel(), **label_kws)
                        ax.tick_params(**tick_kws)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{st}_{en}', where='data')
                else:
                    self.plot_df_with_freq(sub_df, freq, save, prefix=f'{prefix}_{st}_{en}', **kwargs)
        return

    def plot_df_with_freq(self, df:pd.DataFrame, freq:str, save:bool=True, prefix:str='', **kwargs):
        """Plots a dataframe which has data as time-series and its index is pd.DatetimeIndex"""

        validate_freq(df, freq)

        st_year = df.index[0].year
        en_year = df.index[-1].year

        for yr in range(st_year, en_year + 1):

            _df = df[df.index.year == yr]

            if freq == 'yearly':
                kwargs = plot_style(_df, **kwargs)
                _df.plot(**kwargs)
                self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)}', where='data')

            elif freq == 'monthly':
                st_mon = _df.index[0].month
                en_mon = _df.index[-1].month

                for mon in range(st_mon, en_mon+1):

                    __df = _df[_df.index.month == mon]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(mon)}', where='data/monthly')

            elif freq == 'weekly':
                st_week = _df.index[0].isocalendar()[1]
                en_week = _df.index[-1].isocalendar()[1]

                for week in range(st_week, en_week+1):
                    __df = _df[_df.index.week == week]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(week)}', where='data/weely')
        return


    def feature_feature_corr(self, cols=None, remove_targets=True, save=True, **kwargs):
        """
        >>>from ai4water.eda import EDA
        >>>vis = EDA(data)
        >>>vis.feature_feature_corr(save=False)
        """
        if cols is None:
            cols = self.in_cols if remove_targets else self.in_cols + self.out_cols
            if isinstance(cols, dict):
                cols = None

        if isinstance(self.data, pd.DataFrame):
            self.feature_feature_corr_df(self.data, cols, save=save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    self.feature_feature_corr_df(data, cols[idx] if cols is not None else None,
                                               prefix=str(idx), save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.feature_feature_corr_df(data, cols, prefix=data_name, save=save, **kwargs)
        return

    def feature_feature_corr_df(self,
                              data,
                              cols=None,
                              prefix='',
                              save=True,
                              split=None,
                              threshold=0,
                              method='pearson',
                              **kwargs):
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
            cols = data.columns

        corr = data[cols].corr(method=method)

        if split == "pos":
            corr = corr.where((corr >= threshold) & (corr > 0))
        elif split == "neg":
            corr = corr.where((corr <= threshold) & (corr < 0))

        mask = np.zeros_like(corr, dtype=np.bool)

        vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
        vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)
        # width x height
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (5 + len(cols)*0.25, 9 + len(cols)*0.1)))

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
        self.save_or_show(save, fname=f"{split if split else ''}_feature_corr_{prefix}", where="data")
        return


    def plot_pcs(self, num_pcs=None, save=True, save_as_csv=False, figsize=(12, 8), **kwargs):
        """Plots principle components.
        kwargs will go to sns.pairplot."""
        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self._plot_pcs(data[self.in_cols],
                               num_pcs, save=save, prefix=str(idx), save_as_csv=save_as_csv,
                               hue=self.out_cols[idx], figsize=figsize, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                self._plot_pcs(data[self.in_cols], num_pcs, save=save, prefix=data_name, save_as_csv=save_as_csv,
                               hue=self.out_cols,
                               figsize=figsize, **kwargs)
        else:
            self._plot_pcs(self.data[self.in_cols], num_pcs, save=save, save_as_csv=save_as_csv, hue=self.out_cols,
                           figsize=figsize, **kwargs)
        return

    def _plot_pcs(self, data, num_pcs, save=True, prefix='', save_as_csv=False, hue=None, figsize=(12,8), **kwargs):

        if num_pcs is None:
            num_pcs = int(data.shape[1]/2)

        if num_pcs <1:
            print(f'{num_pcs} pcs can not be plotted because data has shape {data.shape}')
            return
        #df_pca = data[self.in_cols]
        #pca = PCA(n_components=num_pcs).fit(df_pca)
        #df_pca = pd.DataFrame(pca.transform(df_pca))

        transformer = Transformations(data=data, method='pca', n_components=num_pcs, replace_nans=True)
        df_pca = transformer.transform()

        pcs = ['pc' + str(i + 1) for i in range(num_pcs)]
        df_pca.columns = pcs

        if hue is not None:
            if isinstance(hue, list):
                hue = hue[0]
            if hue in data:
                df_pca[hue] = data[hue]

                if df_pca[hue].isna().sum() > 0: # output columns contains nans, so don't use it as hue.
                    hue = None
            else: # ignore it
                hue = None

        if save_as_csv:
            df_pca.to_csv(os.path.join(self.path, f"data\\first_{num_pcs}_pcs_{prefix}"))

        plt.close('all')
        plt.figure(figsize=figsize)
        sns.pairplot(data=df_pca, vars=pcs, hue=hue, **kwargs)
        self.save_or_show(fname=f"first_{num_pcs}_pcs_{prefix}", save=save, where='data')
        return

    def grouped_scatter(self, inputs=True, outputs=True, cols=None, save=True, max_subplots=8, **kwargs):

        fname = "scatter_plot_"

        if cols is None:
            cols = []

            if inputs:
                cols += self.in_cols
                fname += "inputs_"
            if outputs:
                fname += "outptuts_"
        else:
            assert isinstance(cols, list)

        if isinstance(self.data, pd.DataFrame):
            self.grouped_scatter_plot_df(self.data[cols], max_subplots, save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    _cols = cols + [self.out_cols[idx]] if outputs else cols
                    self.grouped_scatter_plot_df(data[_cols], max_subplots, save=save, prefix=str(idx), **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.grouped_scatter_plot_df(data, max_subplots, save=save, prefix=data_name,
                                                 **kwargs)
        return

    def grouped_scatter_plot_df(self, data:pd.DataFrame, max_subplots:int=10, save=True, prefix='', **kwargs):
        """
        max_subplots: int, it can be set to large number to show all the scatter plots on one axis.
        """
        data = data.copy()
        if data.shape[1] <= max_subplots:
            self._grouped_scatter_plot(data, save=save,name=f'grouped_scatter_{prefix}',  **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_subplots)
            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, st:en]
                self._grouped_scatter_plot(sub_df, name=f'grouped_scatter_{prefix}_{st}_{en}', **kwargs)
        return

    def _grouped_scatter_plot(self, df, save=True, name='grouped_scatter', **kwargs):
        plt.close('all')
        sns.set()
        sns.pairplot(df, size=2.5, **kwargs)
        self.save_or_show(fname=name, save=save, where='data')
        return

    def plot_histograms(self, save=True, cols=None, **kwargs):
        """Plots distribution of data as histogram.
        kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
        """
        if isinstance(self.data, pd.DataFrame):
            self.plot_his_df(self.data, save=save, cols=cols, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self.plot_his_df(data, prefix=str(idx), cols=cols, save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.plot_his_df(data, prefix=data_name, save=save, **kwargs)
        return

    def plot_his_df(self, data:pd.DataFrame, prefix='', cols=None, save=True, bins=100, figsize=(20, 14), **kwargs):
        if cols is None:
            cols = data.columns
        data[cols].hist(bins=bins, figsize=figsize, **kwargs)
        self.save_or_show(fname=f"hist_{prefix}", save=save, where='data')
        return


    def plot_index(self, save=True, **kwargs):
        """plots the datetime index of dataframe"""
        if isinstance(self.data, pd.DataFrame):
            self._plot_index(self.data, save=save, **kwargs)

        elif isinstance(self.data, list):
            for data in self.data:
                if isinstance(data, pd.DataFrame):
                    self._plot_index(data, save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.values():
                if isinstance(data, pd.DataFrame):
                    self._plot_index(data, save=save, **kwargs)
        return

    def _plot_index(self,
                    index,
                    save=True,
                    fname="index",
                    figsize=(10,5),
                    dpi=200,
                    label_fs=18,
                    title_fs=20,
                    leg_fs=14,
                    leg_ms=4,
                    color='r',
                    ):
        """
        Plots the index of a datafram.
        index: can be pandas dataframe or index itself. if dataframe, its index will be used for plotting
        """
        plt.close('all')
        if isinstance(index, pd.DataFrame):
            index=index.index

        idx = pd.DataFrame(np.ones(len(index)), index=index, columns=['Observations'])
        axis = idx.plot(linestyle='', marker='.', color=color, figsize=figsize)
        axis.legend(fontsize=leg_fs, markerscale=leg_ms)
        axis.set_xlabel(axis.get_xlabel(), fontdict={'fontsize': label_fs})
        axis.set_title("Temporal distribution of Observations", fontsize=title_fs)
        axis.get_yaxis().set_visible(False)
        self.save_or_show(save=save, fname=fname, where='data', dpi=dpi)
        return



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