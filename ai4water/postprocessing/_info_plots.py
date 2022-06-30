


from pdpbox.info_plot_utils import (_target_plot, _info_plot_interact, _actual_plot, _prepare_info_plot_interact_data,
                              _prepare_info_plot_interact_summary, _prepare_info_plot_data,
                              _check_info_plot_interact_params, _check_info_plot_params)
from pdpbox.utils import _make_list, _check_model, _check_target, _check_classes
from pdpbox.info_plots import q1, q2, q3


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
        grid_range=None, cust_grid_points=None, show_percentile=False, show_outliers=False, endpoint=True,
        classes=None, ncols=2, figsize=None, plot_params=None
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
        df=inputs, feature=feature, grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
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

    info_df_x, bar_data, summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_data(
        feature=feature, feature_type=feature_type, data=info_df, num_grid_points=num_grid_points,
        grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=endpoint)

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

    fig, axes = _actual_plot(plot_data=info_df_x, bar_data=bar_data, box_lines=box_lines,
                             actual_prediction_columns=actual_prediction_columns, feature_name=feature_name,
                             display_columns=display_columns, percentile_columns=percentile_columns,
                             figsize=figsize,
                             ncols=ncols, plot_params=plot_params)
    return fig, axes, summary_df


# def _actual_plot(plot_data, bar_data, box_lines, actual_prediction_columns, feature_name,
#                  display_columns, percentile_columns, figsize, ncols, plot_params):
#     """Internal call for actual_plot"""
#
#     # set up graph parameters
#     width, height = 15, 9
#     nrows = 1
#
#     if plot_params is None:
#         plot_params = dict()
#
#     # font_family = plot_params.get('font_family', 'Arial')
#     box_color = plot_params.get('box_color', '#3288bd')
#     box_colors_cmap = plot_params.get('box_colors_cmap', 'tab20')
#     box_colors = plot_params.get('box_colors', plt.get_cmap(box_colors_cmap)(
#         range(np.min([20, len(actual_prediction_columns)]))))
#     title = plot_params.get('title', 'Actual predictions plot for %s' % feature_name)
#     subtitle = plot_params.get('subtitle', 'Distribution of actual prediction through different feature values.')
#
#     if len(actual_prediction_columns) > 1:
#         nrows = int(np.ceil(len(actual_prediction_columns) * 1.0 / ncols))
#         ncols = np.min([len(actual_prediction_columns), ncols])
#         width = np.min([7.5 * len(actual_prediction_columns), 15])
#         height = width * 1.0 / ncols * nrows
#
#     if figsize is not None:
#         width, height = figsize
#
#     if plot_params is None:
#         plot_params = dict()
#
#     fig = plt.figure(figsize=(width, height))
#     outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height-2])
#     title_ax = plt.subplot(outer_grid[0])
#     fig.add_subplot(title_ax)
#     _plot_title(title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params)
#
#     box_bar_params = {'bar_data': bar_data, 'feature_name': feature_name, 'display_columns': display_columns,
#                       'percentile_columns': percentile_columns, 'plot_params': plot_params}
#
#     if len(actual_prediction_columns) == 1:
#         inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1])
#
#         box_ax = plt.subplot(inner_grid[0])
#         bar_ax = plt.subplot(inner_grid[1], sharex=box_ax)
#         fig.add_subplot(box_ax)
#         fig.add_subplot(bar_ax)
#
#         if actual_prediction_columns[0] == 'actual_prediction':
#             target_ylabel = ''
#         else:
#             target_ylabel = 'target_%s: ' % actual_prediction_columns[0].split('_')[-1]
#
#         box_data = plot_data[['x', actual_prediction_columns[0]]].rename(columns={actual_prediction_columns[0]: 'y'})
#         box_line_data = box_lines[0].rename(columns={actual_prediction_columns[0] + '_q2': 'y'})
#         _draw_box_bar(bar_ax=bar_ax, box_data=box_data, box_line_data=box_line_data, box_color=box_color,
#                       box_ax=box_ax, target_ylabel=target_ylabel, **box_bar_params)
#     else:
#         inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.35)
#
#         box_ax = []
#         bar_ax = []
#
#         # get max average target value
#         ys = []
#         for idx in range(len(box_lines)):
#             ys += list(box_lines[idx][actual_prediction_columns[idx] + '_q2'].values)
#         y_max = np.max(ys)
#
#         for idx in range(len(actual_prediction_columns)):
#             box_color = box_colors[idx % len(box_colors)]
#
#             inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_grid[idx], wspace=0, hspace=0.2)
#             inner_box_ax = plt.subplot(inner[0])
#             inner_bar_ax = plt.subplot(inner[1], sharex=inner_box_ax)
#             fig.add_subplot(inner_box_ax)
#             fig.add_subplot(inner_bar_ax)
#
#             inner_box_data = plot_data[['x', actual_prediction_columns[idx]]].rename(
#                 columns={actual_prediction_columns[idx]: 'y'})
#             inner_box_line_data = box_lines[idx].rename(columns={actual_prediction_columns[idx] + '_q2': 'y'})
#             _draw_box_bar(bar_ax=inner_bar_ax, box_data=inner_box_data,
#                           box_line_data=inner_box_line_data, box_color=box_color, box_ax=inner_box_ax,
#                           target_ylabel='target_%s: ' % actual_prediction_columns[idx].split('_')[-1], **box_bar_params)
#
#             inner_box_ax.set_ylim(0., y_max)
#
#             if idx % ncols != 0:
#                 inner_bar_ax.set_yticklabels([])
#                 inner_box_ax.set_yticklabels([])
#
#             box_ax.append(inner_box_ax)
#             bar_ax.append(inner_bar_ax)
#
#     axes = {'title_ax': title_ax, 'box_ax': box_ax, 'bar_ax': bar_ax}
#     return fig, axes
#
#
# def _prepare_info_plot_data(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
#                             grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
#     """Prepare data for information plots"""
#     prepared_results = _prepare_data_x(
#         feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points, grid_type=grid_type,
#         percentile_range=percentile_range, grid_range=grid_range, cust_grid_points=cust_grid_points,
#         show_percentile=show_percentile, show_outliers=show_outliers, endpoint=endpoint)
#     data_x = prepared_results['data']
#     display_columns, bound_lows, bound_ups = prepared_results['value_display']
#     percentile_columns, percentile_bound_lows, percentile_bound_ups = prepared_results['percentile_display']
#
#     data_x['fake_count'] = 1
#     bar_data = data_x.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)
#     summary_df = pd.DataFrame(np.arange(data_x['x'].min(), data_x['x'].max() + 1), columns=['x'])
#     summary_df = summary_df.merge(bar_data.rename(columns={'fake_count': 'count'}), on='x', how='left').fillna(0)
#
#     summary_df['display_column'] = summary_df['x'].apply(lambda x: display_columns[int(x)])
#     info_cols = ['x', 'display_column']
#     if feature_type == 'numeric':
#         summary_df['value_lower'] = summary_df['x'].apply(lambda x: bound_lows[int(x)])
#         summary_df['value_upper'] = summary_df['x'].apply(lambda x: bound_ups[int(x)])
#         info_cols += ['value_lower', 'value_upper']
#
#     if len(percentile_columns) != 0:
#         summary_df['percentile_column'] = summary_df['x'].apply(lambda x: percentile_columns[int(x)])
#         summary_df['percentile_lower'] = summary_df['x'].apply(lambda x: percentile_bound_lows[int(x)])
#         summary_df['percentile_upper'] = summary_df['x'].apply(lambda x: percentile_bound_ups[int(x)])
#         info_cols += ['percentile_column', 'percentile_lower', 'percentile_upper']
#
#     return data_x, bar_data, summary_df, info_cols, display_columns, percentile_columns
#
#
# def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
#                     grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
#     """Map value to bucket based on feature grids"""
#     display_columns = []
#     bound_ups = []
#     bound_lows = []
#     percentile_columns = []
#     percentile_bound_lows = []
#     percentile_bound_ups = []
#     data_x = data.copy()
#
#     if feature_type == 'binary':
#         feature_grids = np.array([0, 1])
#         display_columns = ['%s_0' % feature, '%s_1' % feature]
#         data_x['x'] = data_x[feature]
#     if feature_type == 'numeric':
#         percentile_info = None
#         if cust_grid_points is None:
#             feature_grids, percentile_info = _get_grids(
#                 feature_values=data_x[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
#                 percentile_range=percentile_range, grid_range=grid_range)
#         else:
#             feature_grids = np.array(sorted(cust_grid_points))
#
#         if not show_outliers:
#             data_x = data_x[(data_x[feature] >= feature_grids[0])
#                             & (data_x[feature] <= feature_grids[-1])].reset_index(drop=True)
#
#         # map feature value into value buckets
#         data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids, endpoint=endpoint))
#         uni_xs = sorted(data_x['x'].unique())
#
#         # create bucket names
#         display_columns, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=feature_grids, endpoint=endpoint)
#         display_columns = np.array(display_columns)[range(uni_xs[0], uni_xs[-1]+1)]
#         bound_lows = np.array(bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
#         bound_ups = np.array(bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]
#
#         # create percentile bucket names
#         if show_percentile and grid_type == 'percentile':
#             percentile_columns, percentile_bound_lows, percentile_bound_ups = \
#                 _make_bucket_column_names_percentile(percentile_info=percentile_info, endpoint=endpoint)
#             percentile_columns = np.array(percentile_columns)[range(uni_xs[0], uni_xs[-1]+1)]
#             percentile_bound_lows = np.array(percentile_bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
#             percentile_bound_ups = np.array(percentile_bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]
#
#         # adjust results
#         data_x['x'] = data_x['x'] - data_x['x'].min()
#
#     if feature_type == 'onehot':
#         feature_grids = display_columns = np.array(feature)
#         data_x['x'] = data_x[feature].apply(lambda x: _find_onehot_actual(x=x), axis=1)
#         data_x = data_x[~data_x['x'].isnull()].reset_index(drop=True)
#
#     data_x['x'] = data_x['x'].map(int)
#     results = {
#         'data': data_x,
#         'value_display': (list(display_columns), list(bound_lows), list(bound_ups)),
#         'percentile_display': (list(percentile_columns), list(percentile_bound_lows), list(percentile_bound_ups))
#     }
#
#     return results
#
#
# def _find_onehot_actual(x):
#     """Map one-hot value to one-hot name"""
#     try:
#         value = list(x).index(1)
#     except:
#         value = np.nan
#     return value
#
# def _find_bucket(x, feature_grids, endpoint):
#     """Find bucket that x falls in"""
#     # map value into value bucket
#     if x < feature_grids[0]:
#         bucket = 0
#     else:
#         if endpoint:
#             if x > feature_grids[-1]:
#                 bucket = len(feature_grids)
#             else:
#                 bucket = len(feature_grids) - 1
#                 for i in range(len(feature_grids) - 2):
#                     if feature_grids[i] <= x < feature_grids[i + 1]:
#                         bucket = i + 1
#         else:
#             if x >= feature_grids[-1]:
#                 bucket = len(feature_grids)
#             else:
#                 bucket = len(feature_grids) - 1
#                 for i in range(len(feature_grids) - 2):
#                     if feature_grids[i] <= x < feature_grids[i + 1]:
#                         bucket = i + 1
#     return bucket
#
#
# def _get_grids(feature_values, num_grid_points, grid_type, percentile_range, grid_range):
#     """Calculate grid points for numeric feature
#
#     Returns
#     -------
#     feature_grids: 1d-array
#         calculated grid points
#     percentile_info: 1d-array or []
#         percentile information for feature_grids
#         exists when grid_type='percentile'
#     """
#
#     if grid_type == 'percentile':
#         # grid points are calculated based on percentile in unique level
#         # thus the final number of grid points might be smaller than num_grid_points
#         start, end = 0, 100
#         if percentile_range is not None:
#             start, end = np.min(percentile_range), np.max(percentile_range)
#
#         percentile_grids = np.linspace(start=start, stop=end, num=num_grid_points)
#         value_grids = np.percentile(feature_values, percentile_grids)
#
#         grids_df = pd.DataFrame()
#         grids_df['percentile_grids'] = [round(v, 2) for v in percentile_grids]
#         grids_df['value_grids'] = value_grids
#         grids_df = grids_df.groupby(['value_grids'], as_index=False).agg(
#             {'percentile_grids': lambda v: str(tuple(v)).replace(',)', ')')}).sort_values('value_grids', ascending=True)
#
#         feature_grids, percentile_info = grids_df['value_grids'].values, grids_df['percentile_grids'].values
#     else:
#         if grid_range is not None:
#             value_grids = np.linspace(np.min(grid_range), np.max(grid_range), num_grid_points)
#         else:
#             value_grids = np.linspace(np.min(feature_values), np.max(feature_values), num_grid_points)
#         feature_grids, percentile_info = value_grids, []
#
#     return feature_grids, percentile_info
#
#
# def _make_bucket_column_names(feature_grids, endpoint):
#     """Create bucket names based on feature grids"""
#     # create bucket names
#     column_names = []
#     bound_lows = [np.nan]
#     bound_ups = [feature_grids[0]]
#
#     feature_grids_str = []
#     for g in feature_grids:
#         feature_grids_str.append(_get_string(x=g))
#
#     # number of buckets: len(feature_grids_str) - 1
#     for i in range(len(feature_grids_str) - 1):
#         column_name = '[%s, %s)' % (feature_grids_str[i], feature_grids_str[i + 1])
#         bound_lows.append(feature_grids[i])
#         bound_ups.append(feature_grids[i + 1])
#
#         if (i == len(feature_grids_str) - 2) and endpoint:
#             column_name = '[%s, %s]' % (feature_grids_str[i], feature_grids_str[i + 1])
#
#         column_names.append(column_name)
#
#     if endpoint:
#         column_names = ['< %s' % feature_grids_str[0]] + column_names + ['> %s' % feature_grids_str[-1]]
#     else:
#         column_names = ['< %s' % feature_grids_str[0]] + column_names + ['>= %s' % feature_grids_str[-1]]
#
#     bound_lows.append(feature_grids[-1])
#     bound_ups.append(np.nan)
#
#     return column_names, bound_lows, bound_ups
#
#
#
# def _make_bucket_column_names_percentile(percentile_info, endpoint):
#     """Create bucket names based on percentile info"""
#     # create percentile bucket names
#     percentile_column_names = []
#     percentile_info_numeric = []
#     for p_idx, p in enumerate(percentile_info):
#         p_array = np.array(p.replace('(', '').replace(')', '').split(', ')).astype(np.float64)
#         if p_idx == 0 or p_idx == len(percentile_info) - 1:
#             p_numeric = np.min(p_array)
#         else:
#             p_numeric = np.max(p_array)
#         percentile_info_numeric.append(p_numeric)
#
#     percentile_bound_lows = [0]
#     percentile_bound_ups = [percentile_info_numeric[0]]
#
#     for i in range(len(percentile_info) - 1):
#         # for each grid point, percentile information is in tuple format
#         # (percentile1, percentile2, ...)
#         # some grid points would belong to multiple percentiles
#         low, high = percentile_info_numeric[i], percentile_info_numeric[i + 1]
#         low_str, high_str = _get_string(x=low), _get_string(x=high)
#
#         percentile_column_name = '[%s, %s)' % (low_str, high_str)
#         percentile_bound_lows.append(low)
#         percentile_bound_ups.append(high)
#
#         if i == len(percentile_info) - 2:
#             if endpoint:
#                 percentile_column_name = '[%s, %s]' % (low_str, high_str)
#             else:
#                 percentile_column_name = '[%s, %s)' % (low_str, high_str)
#
#         percentile_column_names.append(percentile_column_name)
#
#     low, high = percentile_info_numeric[0], percentile_info_numeric[-1]
#     low_str, high_str = _get_string(x=low), _get_string(x=high)
#
#     if endpoint:
#         percentile_column_names = ['< %s' % low_str] + percentile_column_names + ['> %s' % high_str]
#     else:
#         percentile_column_names = ['< %s' % low_str] + percentile_column_names + ['>= %s' % high_str]
#     percentile_bound_lows.append(high)
#     percentile_bound_ups.append(100)
#
#     return percentile_column_names, percentile_bound_lows, percentile_bound_ups
#
#
# def _get_string(x):
#     if int(x) == x:
#         x_str = str(int(x))
#     elif round(x, 1) == x:
#         x_str = str(round(x, 1))
#     else:
#         x_str = str(round(x, 2))
#
#     return x_str
#
# def _plot_title(title, subtitle, title_ax, plot_params):
#     """Add plot title."""
#
#     title_params = {'fontname': plot_params.get('font_family', 'Arial'), 'x': 0, 'va': 'top', 'ha': 'left'}
#     title_fontsize = plot_params.get('title_fontsize', 15)
#     subtitle_fontsize = plot_params.get('subtitle_fontsize', 12)
#
#     title_ax.set_facecolor('white')
#     title_ax.text(y=0.7, s=title, fontsize=title_fontsize, **title_params)
#     title_ax.text(y=0.5, s=subtitle, fontsize=subtitle_fontsize, color='grey', **title_params)
#     title_ax.axis('off')
#
#
# def _draw_box_bar(bar_data, bar_ax, box_data, box_line_data, box_color, box_ax,
#                   feature_name, display_columns, percentile_columns, plot_params, target_ylabel):
#     """Draw box plot and bar plot"""
#
#     font_family = plot_params.get('font_family', 'Arial')
#     xticks_rotation = plot_params.get('xticks_rotation', 0)
#
#     _draw_boxplot(box_data=box_data, box_line_data=box_line_data, box_ax=box_ax,
#                   display_columns=display_columns, box_color=box_color, plot_params=plot_params)
#     box_ax.set_ylabel('%sprediction dist' % target_ylabel)
#     box_ax.set_xticklabels([])
#
#     _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)
#
#     # bar plot
#     bar_ax.set_xlabel(feature_name)
#     bar_ax.set_ylabel('count')
#
#     bar_ax.set_xticks(range(len(display_columns)))
#     bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
#     bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)
#
#     plt.setp(box_ax.get_xticklabels(), visible=False)
#
#     # display percentile
#     if len(percentile_columns) > 0:
#         percentile_ax = box_ax.twiny()
#         percentile_ax.set_xticks(box_ax.get_xticks())
#         percentile_ax.set_xbound(box_ax.get_xbound())
#         percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
#         percentile_ax.set_xlabel('percentile buckets')
#         _axes_modify(font_family=font_family, ax=percentile_ax, top=True)
#
#
# def _draw_boxplot(box_data, box_line_data, box_ax, display_columns, box_color, plot_params):
#     """Draw box plot"""
#     font_family = plot_params.get('font_family', 'Arial')
#     box_line_width = plot_params.get('box_line_width', 1.5)
#     box_width = plot_params.get('box_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))
#
#     xs = sorted(box_data['x'].unique())
#     ys = []
#     for x in xs:
#         ys.append(box_data[box_data['x'] == x]['y'].values)
#
#     boxprops = dict(linewidth=box_line_width, color=box_color)
#     medianprops = dict(linewidth=0)
#     whiskerprops = dict(linewidth=box_line_width, color=box_color)
#     capprops = dict(linewidth=box_line_width, color=box_color)
#
#     box_ax.boxplot(ys, positions=xs, showfliers=False, widths=box_width, whiskerprops=whiskerprops, capprops=capprops,
#                    boxprops=boxprops, medianprops=medianprops)
#
#     _axes_modify(font_family=font_family, ax=box_ax)
#
#     box_ax.plot(box_line_data['x'], box_line_data['y'], linewidth=1, c=box_color, linestyle='--')
#     for idx in box_line_data.index.values:
#         bbox_props = {'facecolor': 'white', 'edgecolor': box_color, 'boxstyle': "square,pad=0.5", 'lw': 1}
#         box_ax.text(box_line_data.loc[idx, 'x'], box_line_data.loc[idx, 'y'], '%.3f' % box_line_data.loc[idx, 'y'],
#                     ha="center", va="top", size=10, bbox=bbox_props, color=box_color)
#
#
#
# def _draw_barplot(bar_data, bar_ax, display_columns, plot_params):
#     """Draw bar plot"""
#
#     font_family = plot_params.get('font_family', 'Arial')
#     bar_color = plot_params.get('bar_color', '#5BB573')
#     bar_width = plot_params.get('bar_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))
#
#     # add value label for bar plot
#     rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=bar_width, color=bar_color, alpha=0.5)
#     _autolabel(rects=rects, ax=bar_ax, bar_color=bar_color)
#     _axes_modify(font_family=font_family, ax=bar_ax)
#
#
# def _axes_modify(font_family, ax, top=False, right=False, grid=False):
#     """Modify matplotlib Axes
#
#     Parameters
#     ----------
#     top: bool, default=False
#         xticks location=top
#     right: bool, default=False
#         yticks, location=right
#     grid: bool, default=False
#         whether it is for grid plot
#     """
#
#     ax.set_facecolor('white')
#     ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='#424242', colors='#9E9E9E')
#
#     for tick in ax.get_xticklabels():
#         tick.set_fontname(font_family)
#     for tick in ax.get_yticklabels():
#         tick.set_fontname(font_family)
#
#     ax.set_frame_on(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#
#     if top:
#         ax.get_xaxis().tick_top()
#     if right:
#         ax.get_yaxis().tick_right()
#     if not grid:
#         ax.grid(True, 'major', 'x', ls='--', lw=.5, c='k', alpha=.3)
#         ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
#
#
# def _autolabel(rects, ax, bar_color):
#     """Create label for bar plot"""
#     for rect in rects:
#         height = rect.get_height()
#         bbox_props = {'facecolor': 'white', 'edgecolor': bar_color, 'boxstyle': "square,pad=0.5"}
#         ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
#                 ha='center', va='center', bbox=bbox_props, color=bar_color)
#
# def _check_classes(classes_list, n_classes):
#     """Makre sure classes list is valid
#
#     Notes
#     -----
#     class index starts from 0
#
#     """
#     if len(classes_list) > 0 and n_classes > 2:
#         if np.min(classes_list) < 0:
#             raise ValueError('class index should be >= 0.')
#         if np.max(classes_list) > n_classes - 1:
#             raise ValueError('class index should be < n_classes.')
#     return
#
#
# def _check_dataset(df):
#     """Make sure input dataset is pandas DataFrame"""
#     if type(df) != pd.core.frame.DataFrame:
#         raise ValueError('only accept pandas DataFrame')
#
# def _check_info_plot_params(df, feature, grid_type, percentile_range, grid_range,
#                             cust_grid_points, show_outliers):
#     """Check information plot parameters"""
#
#     _check_dataset(df=df)
#     feature_type = _check_feature(feature=feature, df=df)
#     _check_grid_type(grid_type=grid_type)
#     _check_percentile_range(percentile_range=percentile_range)
#
#     # show_outliers should be only turned on when necessary
#     if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
#         show_outliers = False
#     return feature_type, show_outliers
#
#
# def _check_grid_type(grid_type):
#     """Make sure grid type is percentile or equal"""
#     if grid_type not in ['percentile', 'equal']:
#         raise ValueError('grid_type should be "percentile" or "equal".')
#
#
# def _check_feature(feature, df):
#     """Make sure feature exists and infer feature type
#
#     Feature types
#     -------------
#     1. binary
#     2. onehot
#     3. numeric
#     """
#
#     if type(feature) == list:
#         if len(feature) < 2:
#             raise ValueError('one-hot encoding feature should contain more than 1 element')
#         if not set(feature) < set(df.columns.values):
#             raise ValueError('feature does not exist: %s' % str(feature))
#         feature_type = 'onehot'
#     else:
#         if feature not in df.columns.values:
#             raise ValueError('feature does not exist: %s' % feature)
#         if sorted(list(np.unique(df[feature]))) == [0, 1]:
#             feature_type = 'binary'
#         else:
#             feature_type = 'numeric'
#
#     return feature_type
#
#
# def _check_percentile_range(percentile_range):
#     """Make sure percentile range is valid"""
#     if percentile_range is not None:
#         if type(percentile_range) != tuple:
#             raise ValueError('percentile_range: should be a tuple')
#         if len(percentile_range) != 2:
#             raise ValueError('percentile_range: should contain 2 elements')
#         if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
#             raise ValueError('percentile_range: should be between 0 and 100')
#     return
#
#
# def _make_list(x):
#     """Make list when it is necessary"""
#     if type(x) == list:
#         return x
#     return [x]
#
#
# def q1(x):
#     return x.quantile(0.25)
#
#
# def q2(x):
#     return x.quantile(0.5)
#
#
# def q3(x):
#     return x.quantile(0.75)