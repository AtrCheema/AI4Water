import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as FA
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as GF

from TSErrors import FindErrors

COLORS = np.array([
       [0.89411765, 0.10196078, 0.10980392, 1.        ],
       [0.21568627, 0.49411765, 0.72156863, 1.        ],
       [0.30196078, 0.68627451, 0.29019608, 1.        ],
       [0.59607843, 0.30588235, 0.63921569, 1.        ],
       [0.95896577, 0.58394066, 0.04189788, 1.        ],
       [0.96862745, 0.50588235, 0.74901961, 1.        ],
       [0.17877267, 0.78893675, 0.92613355, 1.        ],
       [0.38079258, 0.17830983, 0.78165943, 1.        ],
       [0.13778617, 0.06228198, 0.33547859, 1.        ],
       [0.6       , 0.6       , 0.6       , 1.        ]])

RECTS = {1: (111,),
         2: (211, 212),
         3: (311, 312, 313),
         4: (211, 212, 221, 222)}

class TaylorDiagram(object):
    """
    Taylor diagram.
    Modified after https://gist.github.com/ycopin/3342888
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self,
                 true,
                 fig=None,
                 rect=111,
                 label='_',
                 srange=(0, 1.5),
                 extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        #self.refstd = refstd            # Reference standard deviation
        self.refstd = np.std(true)

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


def plot_taylor(trues:dict, simulations:dict, rects:dict=None, **kwargs):
    """
    :param trues: a dictionary of length > 1, whose keys are scenarios and values represent true/observations at that
                  scenarios.
    :param simulations: A dictionary of length > 1 whose keys are scenarios and whose values are also dictionary. Each
                        sub-dictionary i.e. dictionary of scenario consist of models/simulations.
    :param rects: dict, dictionary defining axis orientation of figure. For example with two scenarios named 'scenario1'
                  and 'scenario2', if we want to plot two plots in one column, then this argument will be
                  {'scenario1': 211,
                   'scenario2': 212}.
                   Default is None.

    :param kwargs: Following keyword arguments are optional
      - add_ith_interval: bool
      - add_grid: bool
      - plot_bias: bool, if True, the size of the markers will be used to represent bias. The markers will be
                   triangles with their sides up/down depending upon value of bias.
      - ref_color: str, color of refrence dot
      - intervals: list, if add_ith_interval is True, then this argument is used. It must be list of lists or list of
                   tuples, where the inner tuple/list must consist of two values one each for x and y.
      - colors: 2d numpy array, defining colors. The first dimension should be equal to number of models.
      - extend: bool, default False, if True, will plot negative correlation
      - save: bool, if True, will save the plot
      - leg_size: string defining size of legend
      - figsize: tuple defining figsize, default is (11,8).
    :return:
    """
    scenarios = trues.keys()

    add_ith_interval = kwargs.get('add_idth_interval', False)
    add_grid = kwargs.get('add_grid', True)
    ref_color = kwargs.get('ref_color', 'r')
    intervals = kwargs.get('intervals', [])
    colors = kwargs.get('colors', COLORS)
    extend = kwargs.get('extend', False)
    save = kwargs.get('save', True)
    name = kwargs.get('name', 'taylor.png')
    plot_bias = kwargs.get('plot_bias', False)
    leg_size = kwargs.get('leg_size', 'medium')
    title = kwargs.get('title', "")
    figsize = kwargs.get("figsize", (11, 8))

    if rects is None:
        rects = {k:v for k,v in zip(scenarios, RECTS[len(scenarios)])}

    n_plots = len(trues)
    assert n_plots == len(simulations)

    sims = list(simulations.values())
    models = len(sims[0])

    for m in sims:
        assert len(m) == models

    def msg(key, where="simulations"):
        return f"Scenario {key} does not match any of the provided scenarios in {where}"

    for scen in scenarios:
        if scen not in simulations:
            raise KeyError(msg(scen))
        if scen not in rects:
            raise KeyError(msg(scen, "rects"))

    def get_marker(er, idx):
        ls = ''
        ms = 10
        marker = '$%d$' % (idx + 1)

        if plot_bias:
            pbias = er.pbias()
            if pbias >= 0.0:
                marker = "^"
            else:
                marker = "v"

        return marker, ms, ls

    plt.close('all')
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=18)

    for season in scenarios:

        dia = TaylorDiagram(trues[season],
                            fig=fig,
                            rect=rects[season],
                            label='Reference',
                            extend=extend)

        dia.samplePoints[0].set_color(ref_color)  # Mark reference point as a red star

        if add_ith_interval:
            for interval in intervals:
                dia.ax.plot(*interval, color='k')

        # Add samples to Taylor diagram
        idx = 0
        for model_name, model in simulations[season].items():
            er = FindErrors(trues[season], model)
            stddev = np.std(model)
            corrcoef = er.corr_coeff()

            marker, ms, ls, = get_marker(er, idx)

            dia.add_sample(stddev, corrcoef,
                           marker=marker,
                           ms=ms,
                           ls=ls,
                           # mfc='k', mec='k', # B&W
                           mfc=colors[idx], mec=colors[idx],  # Colors
                           label=model_name)

            idx += 1

        # Add RMS contours, and label them
        contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels
        dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')

        if add_grid:
            dia.add_grid()  # Add grid
            dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        if len(scenarios) > 1:
            dia._ax.set_title(season.capitalize(), fontdict={'fontsize':14})

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html

    leg_pos = "center" if len(scenarios) == 4 else "upper right"
    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               numpoints=1, prop=dict(size=leg_size), loc=leg_pos)

    fig.tight_layout()

    if save:
        plt.savefig(name, dpi=400)
    plt.show()
    plt.close('all')
    return


if __name__ == "__main__":

    trues = {
        'site1': np.random.normal(20, 40, 10),
        'site2': np.random.normal(20, 40, 10),
        'site3': np.random.normal(20, 40, 10),
        'site4': np.random.normal(20, 40, 10),
    }

    simulations = {
        "site1": {"LSTM": np.random.normal(20, 40, 10),
                  "CNN": np.random.normal(20, 40, 10),
                  "TCN": np.random.normal(20, 40, 10),
                  "CNN-LSTM": np.random.normal(20, 40, 10)},

        "site2": {"LSTM": np.random.normal(20, 40, 10),
                  "CNN": np.random.normal(20, 40, 10),
                  "TCN": np.random.normal(20, 40, 10),
                  "CNN-LSTM": np.random.normal(20, 40, 10)},

        "site3": {"LSTM": np.random.normal(20, 40, 10),
                  "CNN": np.random.normal(20, 40, 10),
                  "TCN": np.random.normal(20, 40, 10),
                  "CNN-LSTM": np.random.normal(20, 40, 10)},

        "site4": {"LSTM": np.random.normal(20, 40, 10),
                  "CNN": np.random.normal(20, 40, 10),
                  "TCN": np.random.normal(20, 40, 10),
                  "CNN-LSTM": np.random.normal(20, 40, 10)},
    }

    x95 = [0.05, 13.9]  # For Prcp, this is for 95th level (r = 0.195)
    y95 = [0.0, 71.0]
    x99 = [0.05, 19.0]  # For Prcp, this is for 99th level (r = 0.254)
    y99 = [0.0, 70.0]

    intervals = [[x95, y95], [x99, y99]]

    rects = dict(site1=221,
                 site2=222,
                 site3=223,
                 site4=224)

    add_ith_interval = True
    add_grid = True

    plot_taylor(trues=trues,
                simulations=simulations,
                rects=rects,
                add_grid=add_grid,
                plot_bias=True,
                add_idth_interval=add_ith_interval,
                intervals=intervals,
                title="Test")