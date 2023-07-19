## matrix_histogram, taken from qutip https://github.com/qutip/qutip/blob/master/qutip/visualization.py




import itertools as it
import numpy as np
from numpy import pi, array, sin, cos, angle, log2


# from .matplotlib_utilities import complex_phase_cmap

# try:
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from packaging.version import parse as parse_version

# Define a custom _axes3D function based on the matplotlib version.
# The auto_add_to_figure keyword is new for matplotlib>=3.4.
if parse_version(mpl.__version__) >= parse_version('3.4'):
    def _axes3D(fig, *args, **kwargs):
        ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
        return fig.add_axes(ax)
else:
    def _axes3D(*args, **kwargs):
        return Axes3D(*args, **kwargs)
# except:
#     print('does not work')


def matrix_histogram(M, xlabels=None, ylabels=None, title=None, limits=None,
                     colorbar=True, fig=None, ax=None, options=None):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.
    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize
    xlabels : list of strings
        list of x labels
    ylabels : list of strings
        list of y labels
    title : string
        title of the plot (optional)
    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)
    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.
    colorbar : bool (default: True)
        show colorbar
    options : dict
        A dictionary containing extra options for the plot.
        The names (keys) and values of the options are
        described below:
        'zticks' : list of numbers
            A list of z-axis tick locations.
        'cmap' : string (default: 'jet')
            The name of the color map to use.
        'cmap_min' : float (default: 0.0)
            The lower bound to truncate the color map at.
            A value in range 0 - 1. The default, 0, leaves the lower
            bound of the map unchanged.
        'cmap_max' : float (default: 1.0)
            The upper bound to truncate the color map at.
            A value in range 0 - 1. The default, 1, leaves the upper
            bound of the map unchanged.
        'bars_spacing' : float (default: 0.1)
            spacing between bars.
        'bars_alpha' : float (default: 1.)
            transparency of bars, should be in range 0 - 1
        'bars_lw' : float (default: 0.5)
            linewidth of bars' edges.
        'bars_edgecolor' : color (default: 'k')
            The colors of the bars' edges.
            Examples: 'k', (0.1, 0.2, 0.5) or '#0f0f0f80'.
        'shade' : bool (default: True)
            Whether to shade the dark sides of the bars (True) or not (False).
            The shading is relative to plot's source of light.
        'azim' : float
            The azimuthal viewing angle.
        'elev' : float
            The elevation viewing angle.
        'proj_type' : string (default: 'ortho' if ax is not passed)
            The type of projection ('ortho' or 'persp')
        'stick' : bool (default: False)
            Changes xlim and ylim in such a way that bars next to
            XZ and YZ planes will stick to those planes.
            This option has no effect if ``ax`` is passed as a parameter.
        'cbar_pad' : float (default: 0.04)
            The fraction of the original axes between the colorbar
            and the new image axes.
            (i.e. the padding between the 3D figure and the colorbar).
        'cbar_to_z' : bool (default: False)
            Whether to set the color of maximum and minimum z-values to the
            maximum and minimum colors in the colorbar (True) or not (False).
        'figsize' : tuple of two numbers
            The size of the figure.
    Returns :
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    Raises
    ------
    ValueError
        Input argument is not valid.
    """

    # default options
    default_opts = {'figsize': None, 'cmap': 'jet', 'cmap_min': 0.,
                    'cmap_max': 1., 'zticks': None, 'bars_spacing': 0.2,
                    'bars_alpha': 1., 'bars_lw': 0.5, 'bars_edgecolor': 'k',
                    'shade': False, 'azim': -35, 'elev': 35,
                    'proj_type': 'ortho', 'stick': False,
                    'cbar_pad': 0.04, 'cbar_to_z': False, 'colorbar_alpha': 1}

    # update default_opts from input options
    if options is None:
        pass
    elif isinstance(options, dict):
        # check if keys in options dict are valid
        if set(options) - set(default_opts):
            raise ValueError("invalid key(s) found in options: "
                             f"{', '.join(set(options) - set(default_opts))}")
        else:
            # updating default options
            default_opts.update(options)
    else:
        raise ValueError("options must be a dictionary")

    # if isinstance(M, Qobj):
    #     # extract matrix data from Qobj
    #     M = M.full()

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() + 0.5
    ypos = ypos.T.flatten() + 0.5
    zpos = np.zeros(n)
    dx = dy = (1 - default_opts['bars_spacing']) * np.ones(n)
    dz = np.real(M.flatten())

    if isinstance(limits, list) and len(limits) == 2:
        z_min = limits[0]
        z_max = limits[1]
    else:
        z_min = min(dz)
        z_max = max(dz)
        if z_min == z_max:
            z_min -= 0.1
            z_max += 0.1

    if default_opts['cbar_to_z']:
        norm = mpl.colors.Normalize(min(dz), max(dz))
    else:
        norm = mpl.colors.Normalize(z_min, z_max)
    cmap = _truncate_colormap(default_opts['cmap'],
                              default_opts['cmap_min'],
                              default_opts['cmap_max'])
    print(type(cmap))
    colors = cmap(norm(dz))

    if ax is None:
        fig = plt.figure(figsize=default_opts['figsize'])
        ax = _axes3D(fig,
                     azim=default_opts['azim'] % 360,
                     elev=default_opts['elev'] % 360)
        ax.set_proj_type(default_opts['proj_type'])

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors,
             edgecolors=default_opts['bars_edgecolor'],
             linewidths=default_opts['bars_lw'],
             alpha=default_opts['bars_alpha'],
             shade=default_opts['shade'])
    # remove vertical lines on xz and yz plane
    ax.yaxis._axinfo["grid"]['linewidth'] = 0
    ax.xaxis._axinfo["grid"]['linewidth'] = 0

    if title:
        ax.set_title(title)

    # x axis
    _update_xaxis(default_opts['bars_spacing'], M, ax, xlabels)

    # y axis
    _update_yaxis(default_opts['bars_spacing'], M, ax, ylabels)

    # z axis
    _update_zaxis(ax, z_min, z_max, default_opts['zticks'])

    # stick to xz and yz plane
    _stick_to_planes(default_opts['stick'],
                     default_opts['azim'], ax, M,
                     default_opts['bars_spacing'])

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75,
                                         pad=default_opts['cbar_pad'], orientation='horizontal')
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', alpha=default_opts["colorbar_alpha"])

    # removing margins
    _remove_margins(ax.xaxis)
    _remove_margins(ax.yaxis)
    _remove_margins(ax.zaxis)

    return fig, ax


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    truncates portion of a colormap and returns the new one
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
        print(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(
            n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



def _update_yaxis(spacing, M, ax, ylabels):
    """
    updates the y-axis
    """
    ytics = [x + (1 - (spacing / 2)) for x in range(M.shape[1])]
    ax.axes.w_yaxis.set_major_locator(plt.FixedLocator(ytics))
    if ylabels:
        nylabels = len(ylabels)
        if nylabels != len(ytics):
            raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
        ax.set_yticklabels(ylabels)
    else:
        ax.set_yticklabels([str(y + 1) for y in range(M.shape[1])])
        ax.set_yticklabels([str(i) for i in range(M.shape[1])])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yticks([y + (1 - (spacing / 2)) for y in range(M.shape[1])])


def _update_xaxis(spacing, M, ax, xlabels):
    """
    updates the x-axis
    """
    xtics = [x + (1 - (spacing / 2)) for x in range(M.shape[1])]
    ax.axes.w_xaxis.set_major_locator(plt.FixedLocator(xtics))
    if xlabels:
        nxlabels = len(xlabels)
        if nxlabels != len(xtics):
            raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xticklabels([str(x + 1) for x in range(M.shape[0])])
        ax.set_xticklabels([str(i) for i in range(M.shape[0])])
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xticks([x + (1 - (spacing / 2)) for x in range(M.shape[0])])


def _update_zaxis(ax, z_min, z_max, zticks):
    """
    updates the z-axis
    """
    ax.axes.w_zaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if isinstance(zticks, list):
        ax.set_zticks(zticks)
    ax.set_zlim3d([min(z_min, 0), z_max])
    
    
def _stick_to_planes(stick, azim, ax, M, spacing):
    """adjusts xlim and ylim in way that bars will
    Stick to xz and yz planes
    """
    if stick is True:
        azim = azim % 360
        if 0 <= azim <= 90:
            ax.set_ylim(1 - .5,)
            ax.set_xlim(1 - .5,)
        elif 90 < azim <= 180:
            ax.set_ylim(1 - .5,)
            ax.set_xlim(0, M.shape[0] + (.5 - spacing))
        elif 180 < azim <= 270:
            ax.set_ylim(0, M.shape[1] + (.5 - spacing))
            ax.set_xlim(0, M.shape[0] + (.5 - spacing))
        elif 270 < azim < 360:
            ax.set_ylim(0, M.shape[1] + (.5 - spacing))
            ax.set_xlim(1 - .5,)
            
def _remove_margins(axis):
    """
    removes margins about z = 0 and improves the style
    by monkey patching
    """
    def _get_coord_info_new(renderer):
        mins, maxs, centers, deltas, tc, highs = \
            _get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs

    _get_coord_info_old = axis._get_coord_info
    axis._get_coord_info = _get_coord_info_new
    
    
def matrix_histogram_complex(M, xlabels=None, ylabels=None,
                             title=None, limits=None, phase_limits=None,
                             colorbar=True, fig=None, ax=None,
                             threshold=None):
    """
    Draw a histogram for the amplitudes of matrix M, using the argument
    of each element for coloring the bars, with the given x and y labels
    and title.
    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize
    xlabels : list of strings
        list of x labels
    ylabels : list of strings
        list of y labels
    title : string
        title of the plot (optional)
    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)
    phase_limits : list/array with two float numbers
        The phase-axis (colorbar) limits [min, max] (optional)
    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.
    threshold: float (None)
        Threshold for when bars of smaller height should be transparent. If
        not set, all bars are colored according to the color map.
    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    Raises
    ------
    ValueError
        Input argument is not valid.
    """

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    Mvec = M.flatten()
    dz = abs(Mvec)

    # make small numbers real, to avoid random colors
    idx, = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -pi
        phase_max = pi

    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()

    colors = cmap(norm(angle(Mvec)))
    if threshold is not None:
        colors[:, 3] = 1 * (dz > threshold)

    if ax is None:
        fig = plt.figure()
        ax = _axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    if title:
        ax.set_title(title)

    # x axis
    xtics = -0.5 + np.arange(M.shape[0])
    ax.axes.w_xaxis.set_major_locator(plt.FixedLocator(xtics))
    if xlabels:
        nxlabels = len(xlabels)
        if nxlabels != len(xtics):
            raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=12)

    # y axis
    ytics = -0.5 + np.arange(M.shape[1])
    ax.axes.w_yaxis.set_major_locator(plt.FixedLocator(ytics))
    if ylabels:
        nylabels = len(ylabels)
        if nylabels != len(ytics):
            raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=12)

    # z axis
    if limits and isinstance(limits, list):
        ax.set_zlim3d(limits)
    else:
        ax.set_zlim3d([0, 1])  # use min/max
    # ax.set_zlabel('abs')

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
        cb.set_ticklabels(
            (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        cb.set_label('arg')

    return fig, ax



def complex_phase_cmap():
    """
    Create a cyclic colormap for representing the phase of complex variables
    Returns
    -------
    cmap :
        A matplotlib linear segmented colormap.
    """
    cdict = {'blue': ((0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
             'green': ((0.00, 0.0, 0.0),
                       (0.25, 1.0, 1.0),
                       (0.50, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 0.0)),
             'red': ((0.00, 1.0, 1.0),
                     (0.25, 0.5, 0.5),
                     (0.50, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 1.0, 1.0))}

    cmap = mpl.colors.LinearSegmentedColormap('phase_colormap', cdict, 256)

    return cmap
