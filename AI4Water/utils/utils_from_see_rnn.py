# The following functions are obtained and modified fromt the repository see-rnn
# The original functions had some deficiencies (e.g It requires model object to be given to it while we give only
# data to it) due to which they could
# not be used as it is, hence they have been put here and modified.

import matplotlib.pyplot as plt
import numpy as np

from see_rnn import scalefig
from see_rnn.inspect_gen import detect_nans
from see_rnn.utils import _kw_from_configs, _save_rnn_fig

def rnn_histogram(data, rnn_info, equate_axes=1, configs=None,
                  **kwargs):
    """Plots histogram grid of RNN weights/gradients by kernel, gate (if gated),
       and direction (if bidirectional). Also detects NaNs and shows on plots.

    Arguments:
        rnn_info: dict

        equate_axes: int: 0, 1, 2. 0 --> auto-managed axes. 1 --> kernel &
                     recurrent subplots' x- & y-axes lims set to common value.
                     2 --> 1, but lims shared for forward & backward plots.
                     Bias plot lims never affected.
        data: np.ndarray. Pre-fetched data to plot directly - e.g., returned by
              `get_rnn_weights`. Overrides `input_data`, `labels` and `mode`.
              `model` and layer args are still needed to fetch RNN-specific info.
        configs: dict. kwargs to customize various plot schemes:
            'plot':      passed to ax.imshow();    ax  = subplots axis
            'subplot':   passed to plt.subplots()
            'tight':     passed to fig.subplots_adjust(); fig = subplots figure
            'title':     passed to fig.suptitle()
            'annot':     passed to ax.annotate()
            'annot-nan': passed to ax.annotate() for `nan_txt`
            'save':      passed to fig.savefig() if `savepath` is not None.
    (1): tf.data.Dataset, generators, .tfrecords, & other supported TensorFlow
         input data formats

    kwargs:
        w: float. Scale width  of resulting plot by a factor.
        h: float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plots.
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        bins: int. Pyplot `hist` kwarg: number of histogram bins per subplot.
        savepath: str/None. Path to save resulting figure to. Also see `configs`.
               If None, doesn't save.

    Returns:
        (subplots_figs, subplots_axes) of generated subplots. If layer is
            bidirectional, len(subplots_figs) == 2, and latter's is also doubled.
    """

    w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
    show_borders  = kwargs.get('show_borders', False)
    show_xy_ticks = kwargs.get('show_xy_ticks', [1, 1])
    show_bias     = kwargs.get('show_bias', True)
    bins          = kwargs.get('bins', 150)
    savepath      = kwargs.get('savepath', False)

    def _process_configs(configs, w, h, equate_axes):
        defaults = {
            'plot':    dict(),
            'subplot': dict(sharex=True, sharey=True, dpi=76, figsize=(9, 9)),
            'tight':   dict(),
            'title':   dict(weight='bold', fontsize=12, y=1.05),
            'annot':     dict(fontsize=12, weight='bold',
                              xy=(.90, .93), xycoords='axes fraction'),
            'annot-nan': dict(fontsize=12, weight='bold', color='red',
                              xy=(.05, .63), xycoords='axes fraction'),
            'save': dict(),
            }
        # deepcopy configs, and override defaults dicts or dict values
        kw = _kw_from_configs(configs, defaults)

        if not equate_axes:
            kw['subplot'].update({'sharex': False, 'sharey': False})
        size = kw['subplot']['figsize']
        kw['subplot']['figsize'] = (size[0] * w, size[1] * h)
        return kw

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('w', 'h', 'show_borders', 'show_xy_ticks', 'bins',
                          'savepath')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _detect_and_zero_nans(matrix_data):
        nan_txt = detect_nans(matrix_data, include_inf=True)
        if nan_txt is not None:  # NaN/Inf detected
            matrix_data[np.isnan(matrix_data)] = 0  # set NaNs to zero
            matrix_data[np.isinf(matrix_data)] = 0  # set Infs to zero
            if ', ' in nan_txt:
                nan_txt = '\n'.join(nan_txt.split(', '))
            else:
                nan_txt = '\n'.join(nan_txt.split(' '))
        return matrix_data, nan_txt

    def _plot_bias(data, axes, direction_idx, bins, d, kw):
        gs = axes[0, 0].get_gridspec()
        for ax in axes[-1, :]:
            ax.remove()
        axbig = fig.add_subplot(gs[-1, :])

        matrix_data = data[2 + direction_idx * 3].ravel()
        matrix_data, nan_txt = _detect_and_zero_nans(matrix_data)
        _pretty_hist(matrix_data, bins, ax=axbig)

        d['gate_names'].append('BIAS')
        _style_axis(axbig, gate_idx=-1, kernel_type=None, nan_txt=nan_txt,
                    show_borders=show_borders, d=d, kw=kw)
        for ax in axes[-2, :].flat:
            # display x labels on bottom row above bias plot as it'll differ
            # per bias row not sharing axes
            ax.tick_params(axis='both', which='both', labelbottom=True)

    def _pretty_hist(matrix_data, bins, ax):
        # hist w/ looping gradient coloring & nan detection
        N, bins, patches = ax.hist(matrix_data, bins=bins, density=True)

        if len(matrix_data) < 1000:
            return  # fewer bins look better monochrome

        bins_norm = bins / bins.max()
        n_loops = 8   # number of gradient loops
        alpha = 0.94  # graph opacity

        for bin_norm, patch in zip(bins_norm, patches):
            grad = np.sin(np.pi * n_loops * bin_norm) / 15 + .04
            color = (0.121569 + grad * 1.2, 0.466667 + grad, 0.705882 + grad,
                     alpha)  # [.121569, .466667, ...] == matplotlib default blue
            patch.set_facecolor(color)

    def _get_axes_extrema(axes):
        axes = np.array(axes)
        is_bidir = len(axes.shape) == 3 and axes.shape[0] != 1
        x_new, y_new = [], []

        for direction_idx in range(1 + is_bidir):
            axis = np.array(axes[direction_idx]).T
            for type_idx in range(2):  # 2 == len(kernel_types)
                x_new += [np.max(np.abs([ax.get_xlim() for ax in axis[type_idx]]))]
                y_new += [np.max(np.abs([ax.get_ylim() for ax in axis[type_idx]]))]
        return max(x_new), max(y_new)

    def _set_axes_limits(axes, x_new, y_new, d):
        axes = np.array(axes)
        is_bidir = len(axes.shape) == 3 and axes.shape[0] != 1

        for direction_idx in range(1 + is_bidir):
            axis = np.array(axes[direction_idx]).T
            for type_idx in range(2):
                for gate_idx in range(d['n_gates']):
                    axis[type_idx][gate_idx].set_xlim(-x_new, x_new)
                    axis[type_idx][gate_idx].set_ylim(0,      y_new)

    def _style_axis(ax, gate_idx, kernel_type, nan_txt, show_borders, d, kw):
        if nan_txt is not None:
            ax.annotate(nan_txt, **kw['annot-nan'])

        is_gated = d['rnn_type'] in gated_types
        if gate_idx == 0:
            title = kernel_type + ' GATES' * is_gated
            ax.set_title(title, weight='bold')
        if is_gated:
            ax.annotate(d['gate_names'][gate_idx], **kw['annot'])

        if not show_borders:
            ax.set_frame_on(False)
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])

    def _get_plot_data(data, direction_idx, type_idx, gate_idx, d):
        matrix_idx = type_idx + direction_idx * (2 + d['uses_bias'])
        matrix_data = data[matrix_idx]

        if d['rnn_type'] in d['gated_types']:
            start = gate_idx * d['rnn_dim']
            end   = start + d['rnn_dim']
            matrix_data = matrix_data[:, start:end]
        return matrix_data.ravel()

    def _make_subplots(show_bias, direction_name, d, kw):
        if not (d['uses_bias'] and show_bias):
            fig, axes = plt.subplots(d['n_gates'], 2, **kw['subplot'])
            axes = np.atleast_2d(axes)
            return fig, axes

        n_rows = 2 * d['n_gates'] + 1
        fig, axes = plt.subplots(n_rows, 2, **kw['subplot'])

        # merge upper axes pairs to increase height ratio w.r.t. bias plot window
        gs = axes[0, 0].get_gridspec()
        for ax in axes[:(n_rows - 1)].flat:
            ax.remove()
        axbigs1, axbigs2 = [], []
        for row in range(n_rows // 2):
            start = 2 * row
            end = start + 2
            axbigs1.append(fig.add_subplot(gs[start:end, 0]))
            axbigs2.append(fig.add_subplot(gs[start:end, 1]))
        axes = np.vstack([np.array([axbigs1, axbigs2]).T, [*axes.flat[-2:]]])

        if direction_name != []:
            fig.suptitle(direction_name + ' LAYER', **kw['title'])
        return fig, axes

    kw = _process_configs(configs, w, h, equate_axes)
    _catch_unknown_kwargs(kwargs)

    # data, rnn_info = _process_rnn_args(model, _id, layer, input_data, labels,
    #                                    mode, data)
    d = rnn_info
    gated_types  = ('LSTM', 'GRU', 'CuDNNLSTM', 'CuDNNGRU')
    kernel_types = ('KERNEL', 'RECURRENT')
    d.update({'gated_types': gated_types, 'kernel_types': kernel_types})

    subplots_axes = []
    subplots_figs = []
    for direction_idx, direction_name in enumerate(d['direction_names']):
        fig, axes = _make_subplots(show_bias, direction_name, d, kw)
        subplots_axes.append(axes)
        subplots_figs.append(fig)

        for type_idx, kernel_type in enumerate(kernel_types):
            for gate_idx in range(d['n_gates']):
                ax = axes[gate_idx][type_idx]
                matrix_data = _get_plot_data(data, direction_idx,
                                             type_idx, gate_idx, d)

                matrix_data, nan_txt = _detect_and_zero_nans(matrix_data)
                _pretty_hist(matrix_data, bins=bins, ax=ax)

                _style_axis(ax, gate_idx, kernel_type, nan_txt, show_borders,
                            d, kw)
        if d['uses_bias'] and show_bias:
            _plot_bias(data, axes, direction_idx, bins, d, kw)

        if kw['tight']:
            fig.subplots_adjust(**kw['tight'])
        else:
            fig.tight_layout()

    if equate_axes == 2:
        x_new, y_new = _get_axes_extrema(subplots_axes)
        _set_axes_limits(subplots_axes, x_new, y_new, d)

    for fig in subplots_figs:
        scalefig(fig)

    if savepath:
        _save_rnn_fig(subplots_figs, savepath, kw['save'])
    else:
        plt.show()

    return subplots_figs, subplots_axes