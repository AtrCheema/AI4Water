
from typing import Callable
from ai4water.backend import np, plt, pd


from pdpbox1.info_plot_utils import (_target_plot, _info_plot_interact, _actual_plot, _prepare_info_plot_interact_data,
                              _prepare_info_plot_interact_summary, _prepare_info_plot_data,
                              _check_info_plot_interact_params, _check_info_plot_params)
from pdpbox1.utils import _make_list, _check_model, _check_target, _check_classes
from pdpbox1.info_plots import q1, q2, q3, heatmap, annotate_heatmap


def feature_interaction(
        predict_func:Callable,
        X,
        features,
        feature_names,
        n_classes:int = 0,
        num_grid_points=None,
        grid_types=None,
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_percentile=False,
        show_outliers=False,
        endpoint=True,
        which_classes=None,
        predict_kwds={},
        ncols=2,
        figsize=None,
        annotate=False,
        plot_params=None,
        annotate_counts=True,
        plot_type="circles",
        annotate_colors=("black", "white"),
        annotate_color_threshold=None,
        annotate_fmt=None,
        annotate_fontsize=7
):
    check_results = _check_info_plot_interact_params(
        num_grid_points=num_grid_points, grid_types=grid_types, percentile_ranges=percentile_ranges,
        grid_ranges=grid_ranges, cust_grid_points=cust_grid_points, show_outliers=show_outliers,
        plot_params=plot_params, features=features, df=X)

    num_grid_points = check_results['num_grid_points']
    grid_types = check_results['grid_types']
    percentile_ranges = check_results['percentile_ranges']
    grid_ranges = check_results['grid_ranges']
    cust_grid_points = check_results['cust_grid_points']
    show_outliers = check_results['show_outliers']
    plot_params = check_results['plot_params']
    feature_types = check_results['feature_types']

    # prediction
    prediction = predict_func(X, **predict_kwds)

    info_df = X[_make_list(features[0]) + _make_list(features[1])]
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
        show_outliers=show_outliers, endpoint=endpoint, agg_dict=agg_dict)

    actual_plot_data.columns = ['_'.join(col) if col[1] != '' else col[0] for col in actual_plot_data.columns]
    actual_plot_data = actual_plot_data.rename(columns={'fake_count_count': 'fake_count'})

    # prepare summary data frame
    summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_interact_summary(
        data_x=data_x, plot_data=actual_plot_data, prepared_results=prepared_results, feature_types=feature_types)
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]

    title = plot_params.get('title', 'Actual predictions plot for %s' % ' & '.join(feature_names))
    subtitle = plot_params.get('subtitle',
                               'Medium value of actual prediction through different feature value combinations.')

    if plot_type== "circles":
        fig, axes = _info_plot_interact(
            feature_names=feature_names, display_columns=display_columns,
            percentile_columns=percentile_columns, ys=[col + '_q2' for col in actual_prediction_columns],
            plot_data=actual_plot_data, title=title, subtitle=subtitle, figsize=figsize,
            ncols=ncols, annotate=annotate, plot_params=plot_params, is_target_plot=False,
            annotate_counts=annotate_counts)
    else:
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
            cmap="YlGn",
            cbarlabel="Median Prediction"
        )
        axes.set_ylabel(features[0])
        axes.set_xlabel(features[1])
        if annotate:
            texts = annotate_heatmap(
                im,
                valfmt=annotate_fmt or "{x:.1f}",
                fontsize=annotate_fontsize,
                textcolors=annotate_colors,
                threshold=annotate_color_threshold,
            )
        elif annotate_counts:
            texts = annotate_heatmap(
                im, counts.values,
                valfmt=annotate_fmt or "{x}",
                fontsize=annotate_fontsize,
                textcolors=annotate_colors,
                threshold=annotate_color_threshold)

    return fig, axes, summary_df


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
        endpoint=True,
        classes=None,
        ncols=2,
        figsize=None,
        plot_params=None
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
