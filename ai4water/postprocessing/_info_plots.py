
from typing import Callable, Union
from ai4water.backend import np, plt, pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from easy_mpl import bar_chart, boxplot, violin_plot, hist
from easy_mpl.utils import make_cols_from_cmap


def feature_interaction(
        predict_func:Callable,
        X,
        features,
        feature_names,
        n_classes:int = 0,
        num_grid_points=None,
        grid_types="percentile",
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_percentile=False,
        show_outliers=False,
        end_point=True,
        which_classes=None,
        predict_kwds={},
        ncols=2,
        cmap="YlGn",
        border=False,
        figsize=None,
        annotate=False,
        annotate_counts=True,
        annotate_colors=("black", "white"),
        annotate_color_threshold=None,
        annotate_fmt=None,
        annotate_fontsize=7
):
    assert isinstance(X, pd.DataFrame)

    num_grid_points = _expand_default(num_grid_points, 10)

    #assert grid_types in ['percentile', 'equal']
    grid_types = _expand_default(grid_types, 'percentile')

    percentile_ranges = _expand_default(percentile_ranges, None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    grid_ranges = _expand_default(grid_ranges, None)

    cust_grid_points = _expand_default(cust_grid_points, None)

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i in range(2):
            if (percentile_ranges[i] is None) and (grid_ranges[i] is None) and (cust_grid_points[i] is None):
                show_outliers[i] = False

    feature_types = [_check_feature(feature=features[0], df=X), _check_feature(feature=features[1], df=X)]

    # prediction
    prediction = predict_func(X, **predict_kwds)

    info_df = X[_make_list(features[0]) + _make_list(features[1])].copy()
    actual_prediction_columns = ['actual_prediction']
    if n_classes == 0:
        info_df['actual_prediction'] = prediction
    elif n_classes == 2:
        info_df['actual_prediction'] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if which_classes is not None:
            plot_classes = sorted(which_classes)

        actual_prediction_columns = []
        for class_idx in plot_classes:
            info_df['actual_prediction_%d' % class_idx] = prediction[:, class_idx]
            actual_prediction_columns.append('actual_prediction_%d' % class_idx)

    agg_dict = {}
    actual_prediction_columns_qs = []
    for idx in range(len(actual_prediction_columns)):
        agg_dict[actual_prediction_columns[idx]] = [q1, q2, q3]
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
    agg_dict['fake_count'] = 'count'

    data_x, actual_plot_data, prepared_results = _prepare_info_plot_interact_data(
        data_input=info_df, features=features, feature_types=feature_types, num_grid_points=num_grid_points,
        grid_types=grid_types, percentile_ranges=percentile_ranges, grid_ranges=grid_ranges,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=end_point, agg_dict=agg_dict)

    actual_plot_data.columns = ['_'.join(col) if col[1] != '' else col[0] for col in actual_plot_data.columns]
    actual_plot_data = actual_plot_data.rename(columns={'fake_count_count': 'fake_count'})

    # prepare summary data frame
    summary_df, info_cols = _prepare_info_plot_interact_summary(
        data_x=data_x, plot_data=actual_plot_data, prepared_results=prepared_results, feature_types=feature_types)
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]

    vals = []
    for i in np.unique(summary_df['x1']):
        row = summary_df['actual_prediction_q2'].loc[summary_df['x1'] == i]
        for j in np.unique(summary_df['x2'])[::-1]:
            vals.append(row.iloc[j])

    counts = []
    for i in np.unique(summary_df['x1']):
        row = summary_df['count'].loc[summary_df['x1'] == i]
        for j in np.unique(summary_df['x2'])[::-1]:
            counts.append(row.iloc[j])

    xticklabels = summary_df.loc[summary_df['x1'] == 0]['display_column_2'].values[::-1]

    #if yticklabels is None:
    yticklabels = summary_df.loc[summary_df['x2'] == 0]['display_column_1'].values

    x = np.array(vals).reshape(len(yticklabels), len(xticklabels))
    df = pd.DataFrame(x, columns=xticklabels, index=yticklabels)

    counts = np.array(counts).reshape(len(yticklabels), len(xticklabels))
    counts = pd.DataFrame(counts, columns=xticklabels, index=yticklabels, dtype=int)

    fig, axes = plt.subplots(figsize=figsize)
    im, cbar = heatmap(
        df,
        row_labels=df.index,
        col_labels=df.columns,
        ax=axes,
        cmap=cmap,
        cbarlabel="Median Prediction",
        border=border
    )
    axes.set_ylabel(features[0])
    axes.set_xlabel(features[1])

    if annotate:
        if annotate_counts:
            annotate_imshow(
                im, counts.values,
                fmt=annotate_fmt or "{:n}",
                fontsize=annotate_fontsize,
                textcolors=annotate_colors,
                threshold=annotate_color_threshold)
        else:
            annotate_imshow(
                im,
                fmt=annotate_fmt or "{:.2f}",
                fontsize=annotate_fontsize,
                textcolors=annotate_colors,
                threshold=annotate_color_threshold,
            )

    return axes, summary_df


def prediction_distribution_plot(
        mode:str,
        inputs,
        prediction,
        feature,
        feature_name,
        n_classes: int = None,
        num_grid_points=10,
        grid_type='percentile',
        percentile_range=None,
        grid_range=None,
        cust_grid_points=None,
        show_percentile=False,
        show_outliers=False,
        end_point=True,
        kind:str = "bar",
        classes=None,
        ncols=2,
        figsize=None,
        show=True,
):
    """
    data = busan_beach()

    model = Model(model="XGBRegressor")
    model.fit(data=data)

    y = model.predict_on_training_data(data=data)
    x, _ = model.training_data(data=data)

    prediction_distribution_plot(
        model.mode,
        inputs=pd.DataFrame(x,
                            columns=model.input_features),
        prediction=y,
        feature='tide_cm',
        feature_name='tide_cm',
        show_percentile=True,
        n_classes=model.num_classes
    )

    plt.show()
    """
    if mode != "regression":
        assert n_classes is not None

    is_binary = False
    if n_classes == 2:
        is_binary = True

    # check inputs

    feature_type, show_outliers = _check_info_plot_params(
        df=inputs, feature=feature, grid_type=grid_type, percentile_range=percentile_range,
        grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_outliers=show_outliers)

    # make predictions
    # info_df only contains feature value and actual predictions
    info_df = inputs[_make_list(feature)]
    actual_prediction_columns = ['actual_prediction']
    if mode == "regression":
        info_df['actual_prediction'] = prediction
    elif is_binary:
        info_df['actual_prediction'] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if classes is not None:
            _check_classes(classes_list=classes, n_classes=n_classes)
            plot_classes = sorted(classes)

        actual_prediction_columns = []
        for class_idx in plot_classes:
            info_df['actual_prediction_%d' % class_idx] = prediction[:, class_idx]
            actual_prediction_columns.append('actual_prediction_%d' % class_idx)

    info_df_x, summary_df, info_cols = _prepare_info_plot_data(
        feature=feature, feature_type=feature_type, data=info_df, num_grid_points=num_grid_points,
        grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=end_point)

    # prepare data for box lines
    # each box line contains 'x' and actual prediction q1, q2, q3
    box_lines = []
    actual_prediction_columns_qs = []
    for idx in range(len(actual_prediction_columns)):
        box_line = info_df_x.groupby('x', as_index=False).agg(
            {actual_prediction_columns[idx]: [q1, q2, q3]}).sort_values('x', ascending=True)
        box_line.columns = ['_'.join(col) if col[1] != '' else col[0] for col in box_line.columns]
        box_lines.append(box_line)
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
        summary_df = summary_df.merge(box_line, on='x', how='outer').fillna(0)
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]

    ax = _dist_plot(summary_df,
                    X=inputs,
                    prediction=prediction,
                    feature=feature,
                    feature_name=feature_name,
                    figsize=figsize,
                    kind=kind or "bar")

    if show:
        plt.tight_layout()
        plt.show()

    return ax, summary_df


def heatmap(
        data,
        row_labels,
        col_labels,
        ax=None,
        cbar_kw={},
        cbarlabel="",
        xlabel_on_top=True,
        border:bool = False,
        **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    border :
    xlabel_on_top
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig: plt.Figure = plt.gcf()
    cbar = fig.colorbar(im, orientation="vertical", pad=0.2, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)


    if xlabel_on_top:
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
    #else:


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    if not border:
        # Turn spines off and create white grid.
        # in older versions ax.spines is dict and in newer versions it is list
        if isinstance(ax.spines, dict):
            for v in ax.spines.values():
                v.set_visible(False)
        else:
            ax.spines[:].set_visible(False)

    if xlabel_on_top:
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    else:
        ax.set_xticks(np.arange(data.shape[1] - 1) + .5, minor=True)

    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_imshow(
        im,
        data:np.ndarray=None,
        textcolors:Union[tuple, np.ndarray]=("black", "white"),
        threshold=None,
        fmt = "{:.2f}",
        **text_kws
):
    """annotates imshow
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if data is None:
        data = im.get_array()

    use_threshold = True
    if isinstance(textcolors, np.ndarray) and textcolors.shape == data.shape:
        assert threshold is None, f"if textcolors is given as array then threshold should be None"
        use_threshold = False
    else:
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            s = fmt.format(float(data[i, j]))
            if use_threshold:
                _ = im.axes.text(j, i, s,
                        color=textcolors[int(im.norm(data[i, j]) > threshold)],
                        **text_kws)
            else:
                _ = im.axes.text(j, i, s,
                        color=textcolors[i, j],
                        **text_kws)
    return


def _make_list(x):
    if not isinstance(x, list):
        return [x]
    return x


def q1(x):
    return x.quantile(0.25)


def q2(x):
    return x.quantile(0.5)


def q3(x):
    return x.quantile(0.75)


def _check_classes(classes_list, n_classes):
    """Makre sure classes list is valid

    Notes
    -----
    class index starts from 0

    """
    if len(classes_list) > 0 and n_classes > 2:
        if np.min(classes_list) < 0:
            raise ValueError('class index should be >= 0.')
        if np.max(classes_list) > n_classes - 1:
            raise ValueError('class index should be < n_classes.')
    return


def _expand_default(x, default):
    """Create a list of default values"""
    if x is None:
        return [default] * 2
    return x


def _check_percentile_range(percentile_range):
    """Make sure percentile range is valid"""
    if percentile_range is not None:
        if type(percentile_range) != tuple:
            raise ValueError('percentile_range: should be a tuple')
        if len(percentile_range) != 2:
            raise ValueError('percentile_range: should contain 2 elements')
        if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
            raise ValueError('percentile_range: should be between 0 and 100')
    return


def _check_feature(feature, df):
    """Make sure feature exists and infer feature type

    Feature types
    -------------
    1. binary
    2. onehot
    3. numeric
    """

    if type(feature) == list:
        if len(feature) < 2:
            raise ValueError('one-hot encoding feature should contain more than 1 element')
        if not set(feature) < set(df.columns.values):
            raise ValueError('feature does not exist: %s' % str(feature))
        feature_type = 'onehot'
    else:
        if feature not in df.columns.values:
            raise ValueError('feature does not exist: %s' % feature)
        if sorted(list(np.unique(df[feature]))) == [0, 1]:
            feature_type = 'binary'
        else:
            feature_type = 'numeric'

    return feature_type


def _prepare_info_plot_interact_data(data_input, features, feature_types, num_grid_points, grid_types,
                                     percentile_ranges, grid_ranges, cust_grid_points, show_percentile,
                                     show_outliers, endpoint, agg_dict):
    """Prepare data for information interact plots"""
    prepared_results = []
    for i in range(2):
        prepared_result = _prepare_data_x(
            feature=features[i], feature_type=feature_types[i], data=data_input,
            num_grid_points=num_grid_points[i], grid_type=grid_types[i], percentile_range=percentile_ranges[i],
            grid_range=grid_ranges[i], cust_grid_points=cust_grid_points[i],
            show_percentile=show_percentile, show_outliers=show_outliers[i], endpoint=endpoint)
        prepared_results.append(prepared_result)
        if i == 0:
            data_input = prepared_result['data'].rename(columns={'x': 'x1'})

    data_x = prepared_results[1]['data'].rename(columns={'x': 'x2'})
    data_x['fake_count'] = 1
    plot_data = data_x.groupby(['x1', 'x2'], as_index=False).agg(agg_dict)

    return data_x, plot_data, prepared_results


def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                    grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
    """Map value to bucket based on feature grids"""
    display_columns = []
    bound_ups = []
    bound_lows = []
    percentile_columns = []
    percentile_bound_lows = []
    percentile_bound_ups = []
    data_x = data.copy()

    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        display_columns = ['%s_0' % feature, '%s_1' % feature]
        data_x['x'] = data_x[feature]
    if feature_type == 'numeric':
        percentile_info = None
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(
                feature_values=data_x[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))

        if not show_outliers:
            data_x = data_x[(data_x[feature] >= feature_grids[0])
                            & (data_x[feature] <= feature_grids[-1])].reset_index(drop=True)

        # map feature value into value buckets
        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids, endpoint=endpoint))
        uni_xs = sorted(data_x['x'].unique())

        # create bucket names
        display_columns, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=feature_grids, endpoint=endpoint)
        display_columns = np.array(display_columns)[range(uni_xs[0], uni_xs[-1]+1)]
        bound_lows = np.array(bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
        bound_ups = np.array(bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]

        # create percentile bucket names
        if show_percentile and grid_type == 'percentile':
            percentile_columns, percentile_bound_lows, percentile_bound_ups = \
                _make_bucket_column_names_percentile(percentile_info=percentile_info, endpoint=endpoint)
            percentile_columns = np.array(percentile_columns)[range(uni_xs[0], uni_xs[-1]+1)]
            percentile_bound_lows = np.array(percentile_bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
            percentile_bound_ups = np.array(percentile_bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]

        # adjust results
        data_x['x'] = data_x['x'] - data_x['x'].min()

    if feature_type == 'onehot':
        feature_grids = display_columns = np.array(feature)
        data_x['x'] = data_x[feature].apply(lambda x: _find_onehot_actual(x=x), axis=1)
        data_x = data_x[~data_x['x'].isnull()].reset_index(drop=True)

    data_x['x'] = data_x['x'].map(int)
    results = {
        'data': data_x,
        'value_display': (list(display_columns), list(bound_lows), list(bound_ups)),
        'percentile_display': (list(percentile_columns), list(percentile_bound_lows), list(percentile_bound_ups))
    }

    return results


def _get_grids(feature_values, num_grid_points, grid_type, percentile_range, grid_range):
    """Calculate grid points for numeric feature

    Returns
    -------
    feature_grids: 1d-array
        calculated grid points
    percentile_info: 1d-array or []
        percentile information for feature_grids
        exists when grid_type='percentile'
    """

    if grid_type == 'percentile':
        # grid points are calculated based on percentile in unique level
        # thus the final number of grid points might be smaller than num_grid_points
        start, end = 0, 100
        if percentile_range is not None:
            start, end = np.min(percentile_range), np.max(percentile_range)

        percentile_grids = np.linspace(start=start, stop=end, num=num_grid_points)
        value_grids = np.percentile(feature_values, percentile_grids)

        grids_df = pd.DataFrame()
        grids_df['percentile_grids'] = [round(v, 2) for v in percentile_grids]
        grids_df['value_grids'] = value_grids
        grids_df = grids_df.groupby(['value_grids'], as_index=False).agg(
            {'percentile_grids': lambda v: str(tuple(v)).replace(',)', ')')}).sort_values('value_grids', ascending=True)

        feature_grids, percentile_info = grids_df['value_grids'].values, grids_df['percentile_grids'].values
    else:
        if grid_range is not None:
            value_grids = np.linspace(np.min(grid_range), np.max(grid_range), num_grid_points)
        else:
            value_grids = np.linspace(np.min(feature_values), np.max(feature_values), num_grid_points)
        feature_grids, percentile_info = value_grids, []

    return feature_grids, percentile_info


def _find_onehot_actual(x):
    """Map one-hot value to one-hot name"""
    try:
        value = list(x).index(1)
    except:
        value = np.nan
    return value



def _find_bucket(x, feature_grids, endpoint):
    """Find bucket that x falls in"""
    # map value into value bucket
    if x < feature_grids[0]:
        bucket = 0
    else:
        if endpoint:
            if x > feature_grids[-1]:
                bucket = len(feature_grids)
            else:
                bucket = len(feature_grids) - 1
                for i in range(len(feature_grids) - 2):
                    if feature_grids[i] <= x < feature_grids[i + 1]:
                        bucket = i + 1
        else:
            if x >= feature_grids[-1]:
                bucket = len(feature_grids)
            else:
                bucket = len(feature_grids) - 1
                for i in range(len(feature_grids) - 2):
                    if feature_grids[i] <= x < feature_grids[i + 1]:
                        bucket = i + 1
    return bucket


def _make_bucket_column_names_percentile(percentile_info, endpoint):
    """Create bucket names based on percentile info"""
    # create percentile bucket names
    percentile_column_names = []
    percentile_info_numeric = []
    for p_idx, p in enumerate(percentile_info):
        p_array = np.array(p.replace('(', '').replace(')', '').split(', ')).astype(np.float64)
        if p_idx == 0 or p_idx == len(percentile_info) - 1:
            p_numeric = np.min(p_array)
        else:
            p_numeric = np.max(p_array)
        percentile_info_numeric.append(p_numeric)

    percentile_bound_lows = [0]
    percentile_bound_ups = [percentile_info_numeric[0]]

    for i in range(len(percentile_info) - 1):
        # for each grid point, percentile information is in tuple format
        # (percentile1, percentile2, ...)
        # some grid points would belong to multiple percentiles
        low, high = percentile_info_numeric[i], percentile_info_numeric[i + 1]
        low_str, high_str = _get_string(x=low), _get_string(x=high)

        percentile_column_name = '[%s, %s)' % (low_str, high_str)
        percentile_bound_lows.append(low)
        percentile_bound_ups.append(high)

        if i == len(percentile_info) - 2:
            if endpoint:
                percentile_column_name = '[%s, %s]' % (low_str, high_str)
            else:
                percentile_column_name = '[%s, %s)' % (low_str, high_str)

        percentile_column_names.append(percentile_column_name)

    low, high = percentile_info_numeric[0], percentile_info_numeric[-1]
    low_str, high_str = _get_string(x=low), _get_string(x=high)

    if endpoint:
        percentile_column_names = ['< %s' % low_str] + percentile_column_names + ['> %s' % high_str]
    else:
        percentile_column_names = ['< %s' % low_str] + percentile_column_names + ['>= %s' % high_str]
    percentile_bound_lows.append(high)
    percentile_bound_ups.append(100)

    return percentile_column_names, percentile_bound_lows, percentile_bound_ups


def _get_string(x):
    if int(x) == x:
        x_str = str(int(x))
    elif round(x, 1) == x:
        x_str = str(round(x, 1))
    else:
        x_str = str(round(x, 2))

    return x_str


def _make_bucket_column_names(feature_grids, endpoint):
    """Create bucket names based on feature grids"""
    # create bucket names
    column_names = []
    bound_lows = [np.nan]
    bound_ups = [feature_grids[0]]

    feature_grids_str = []
    for g in feature_grids:
        feature_grids_str.append(_get_string(x=g))

    # number of buckets: len(feature_grids_str) - 1
    for i in range(len(feature_grids_str) - 1):
        column_name = '[%s, %s)' % (feature_grids_str[i], feature_grids_str[i + 1])
        bound_lows.append(feature_grids[i])
        bound_ups.append(feature_grids[i + 1])

        if (i == len(feature_grids_str) - 2) and endpoint:
            column_name = '[%s, %s]' % (feature_grids_str[i], feature_grids_str[i + 1])

        column_names.append(column_name)

    if endpoint:
        column_names = ['< %s' % feature_grids_str[0]] + column_names + ['> %s' % feature_grids_str[-1]]
    else:
        column_names = ['< %s' % feature_grids_str[0]] + column_names + ['>= %s' % feature_grids_str[-1]]

    bound_lows.append(feature_grids[-1])
    bound_ups.append(np.nan)

    return column_names, bound_lows, bound_ups


def _check_info_plot_params(df, feature, grid_type, percentile_range, grid_range,
                            cust_grid_points, show_outliers):
    """Check information plot parameters"""

    assert isinstance(df, pd.DataFrame)

    feature_type = _check_feature(feature=feature, df=df)

    assert grid_type in ['percentile', 'equal']

    _check_percentile_range(percentile_range=percentile_range)

    # show_outliers should be only turned on when necessary
    if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
        show_outliers = False
    return feature_type, show_outliers


def _prepare_info_plot_interact_summary(data_x, plot_data, prepared_results, feature_types):
    """Prepare summary data frame for interact plots"""

    x1_values = []
    x2_values = []
    for x1_value in range(data_x['x1'].min(), data_x['x1'].max() + 1):
        for x2_value in range(data_x['x2'].min(), data_x['x2'].max() + 1):
            x1_values.append(x1_value)
            x2_values.append(x2_value)
    summary_df = pd.DataFrame()
    summary_df['x1'] = x1_values
    summary_df['x2'] = x2_values
    summary_df = summary_df.merge(plot_data.rename(columns={'fake_count': 'count'}),
                                  on=['x1', 'x2'], how='left').fillna(0)

    info_cols = ['x1', 'x2', 'display_column_1', 'display_column_2']

    for i in range(2):
        display_columns_i, bound_lows_i, bound_ups_i = prepared_results[i]['value_display']
        percentile_columns_i, percentile_bound_lows_i, percentile_bound_ups_i = prepared_results[i]['percentile_display']

        summary_df['display_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: display_columns_i[int(x)])
        if feature_types[i] == 'numeric':
            summary_df['value_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: bound_lows_i[int(x)])
            summary_df['value_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: bound_ups_i[int(x)])
            info_cols += ['value_lower_%d' % (i + 1), 'value_upper_%d' % (i + 1)]

        if len(percentile_columns_i) != 0:
            summary_df['percentile_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_columns_i[int(x)])
            summary_df['percentile_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_bound_lows_i[int(x)])
            summary_df['percentile_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_bound_ups_i[int(x)])
            info_cols += ['percentile_column_%d' % (i + 1), 'percentile_lower_%d' % (i + 1),
                          'percentile_upper_%d' % (i + 1)]

    return summary_df, info_cols


def _prepare_info_plot_data(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                            grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
    """Prepare data for information plots"""
    prepared_results = _prepare_data_x(
        feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points, grid_type=grid_type,
        percentile_range=percentile_range, grid_range=grid_range, cust_grid_points=cust_grid_points,
        show_percentile=show_percentile, show_outliers=show_outliers, endpoint=endpoint)
    data_x = prepared_results['data']
    display_columns, bound_lows, bound_ups = prepared_results['value_display']
    percentile_columns, percentile_bound_lows, percentile_bound_ups = prepared_results['percentile_display']

    data_x['fake_count'] = 1
    bar_data = data_x.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)
    summary_df = pd.DataFrame(np.arange(data_x['x'].min(), data_x['x'].max() + 1), columns=['x'])
    summary_df = summary_df.merge(bar_data.rename(columns={'fake_count': 'count'}), on='x', how='left').fillna(0)

    summary_df['display_column'] = summary_df['x'].apply(lambda x: display_columns[int(x)])
    info_cols = ['x', 'display_column']
    if feature_type == 'numeric':
        summary_df['value_lower'] = summary_df['x'].apply(lambda x: bound_lows[int(x)])
        summary_df['value_upper'] = summary_df['x'].apply(lambda x: bound_ups[int(x)])
        info_cols += ['value_lower', 'value_upper']

    if len(percentile_columns) != 0:
        summary_df['percentile_column'] = summary_df['x'].apply(lambda x: percentile_columns[int(x)])
        summary_df['percentile_lower'] = summary_df['x'].apply(lambda x: percentile_bound_lows[int(x)])
        summary_df['percentile_upper'] = summary_df['x'].apply(lambda x: percentile_bound_ups[int(x)])
        info_cols += ['percentile_column', 'percentile_lower', 'percentile_upper']

    return data_x, summary_df, info_cols


def _dist_plot(
        summary_df,
        feature,
        feature_name,
        X:pd.DataFrame,
        prediction,
        figsize,
               kind="bar"):

    # Draw the bar chart
    fig, ax = plt.subplots(figsize=figsize)

    if kind=="bar":
        color = make_cols_from_cmap("PuBu", len(summary_df), 0.2)
        return bar_chart(
            summary_df['actual_prediction_q2'],
            summary_df['display_column'],
            ax_kws={"xlabel":"Mean Prediction", "xlabel_kws":{"fontsize": 14},
                    "ylabel": f"{feature_name}", 'ylabel_kws': {'fontsize': 14}},
            show=False,
            color=color,
            bar_labels=summary_df['count'],
            bar_label_kws={"color": "black", 'label_type': 'edge', 'fontsize': 14},
            ax=ax,
        )

    preds = {}
    for interval in summary_df['display_column']:
        st, en = interval.split(',')
        st = float(''.join(e for e in st if e not in ["]", ")", "[", "("]))
        en = float(''.join(e for e in en if e not in ["]", ")", "[", "("]))
        df1 = pd.DataFrame(X.copy(), columns=X.columns)
        df1['target'] = prediction
        df1 = df1[[feature, 'target']]
        df1 = df1[(df1[feature] >= st) & (df1[feature] < en)]
        preds[interval] = df1['target'].values

    for k, v in preds.items():
        assert len(v) > 0, f"{k} has no values in it"

    if kind == "box":
        ax, _ = boxplot(
            list(preds.values()), show=False,
            fill_color="lightpink", patch_artist=True,
            medianprops={"color": "black"}, flierprops={"ms": 1.0})
    elif kind == "hist":
        hist(
            list(preds.values()), show=False,
            ax = ax, edgecolor = "k",
            share_axes=False,
            linewidth=0.5,
        )
    elif kind == "violin":
        ax = violin_plot(list(preds.values()), cut=0.4,  show=False)
        ax.set_xticks(range(len(preds)))
        ax.set_facecolor("#fbf9f4")
    else:
        raise ValueError(f"{kind}")

    if kind != "hist":
        ax.set_xlabel(feature_name)
        ax.set_xticklabels(list(preds.keys()))
        ax.set_yticklabels(ax.get_yticks().astype(int))

    return ax
