import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import sys
import matplotlib.cm as cm
from matplotlib.colors import Normalize

RED = '\033[91m'
NC = '\033[0m'


def plot_function(x_, y_, ylabel_, label_='', styles=None, xlabel_=r'$E_{\nu}$ [\si{MeV}]',
                  xlim=None, ylim=None, logx=False, figsize_=[7., 4.], base_major=2.0, base_minor=0.5):
    if len(x_) != len(y_):
        print(f"{RED}Error in plot_function: different lengths of x array ({len(x_)}) and y array ({len(y_)}){NC}")
        print(x_)
        print(y_)
        sys.exit()

    fig = plt.figure(figsize=figsize_, constrained_layout=True)
    # fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.14)
    # fig.subplots_adjust(left=0.11, right=0.91, top=0.95, bottom=0.14)
    ax_ = fig.add_subplot(111)
    if styles is None:
        styles = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    for i_ in np.arange(len(x_)):
        if not logx:
            ax_.plot(x_[i_], y_[i_], styles[i_], linewidth=1., label=label_[i_], markersize=4)
        else:
            ax_.semilogx(x_[i_], y_[i_], styles[i_], linewidth=1., label=label_[i_])

    ax_.grid(alpha=0.65)
    ax_.set_xlabel(xlabel_)
    ax_.set_ylabel(ylabel_)

    if xlim is not None:
        ax_.set_xlim(xlim)
    if ylim is not None:
        ax_.set_ylim(ylim)

    if not logx:
        ax_.xaxis.set_major_locator(plticker.MultipleLocator(base=base_major))
        ax_.xaxis.set_minor_locator(plticker.MultipleLocator(base=base_minor))
    ax_.tick_params('both', direction='out', which='both')
    ax_.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_.legend(labelspacing=0.2)

    return ax_


def plot_function_residual(x_, y_, label_, ylabel_, styles=None, xlabel_=r'$E_{\nu}$ [\si{MeV}]',
                           xlim=None, ylim=None, ylim2=None, logx=False, y2_sci=False,
                           figsize_=[7, 6], base_major=2.0, base_minor=0.5):
    if len(x_) != len(y_):
        print(f"{RED}Error in plot_function: different lengths of x array ({len(x_)}) and y array ({len(y_)}){NC}")
        sys.exit()

    if styles is None:
        styles = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_, gridspec_kw={'height_ratios': [2, 1]}, 
                                   sharex=True, constrained_layout=True)
    # fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.093)

    for i_ in np.arange(len(x_)):
        if not logx:
            ax1.plot(x_[i_], y_[i_], styles[i_], linewidth=1., label=label_[i_], markersize=4)
        else:
            ax1.semilogx(x_[i_], y_[i_], styles[i_], linewidth=1., label=label_[i_])
    
    for j_ in range(1, len(x_)):
        if len(x_[0]) == len(x_[j_]):
            ll = r"("+label_[j_]+" - "+label_[0]+")/"+label_[0]
            if not logx:
                ax2.plot(x_[0], (y_[j_] - y_[0])/y_[0]*100., styles[j_], linewidth=1., label=ll)
            else:
                ax2.semilogx(x_[0], (y_[j_] - y_[0])/y_[0]*100., styles[j_], linewidth=1., label=ll)
        else:
            continue

    ax1.grid(alpha=0.65)
    # ax1.set_xlabel(xlabel_)
    ax1.set_ylabel(ylabel_)
    ax2.grid(alpha=0.65)
    ax2.set_xlabel(xlabel_)
    ax2.set_ylabel(r'relative difference [\%]')

    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)
    if ylim2 is not None:
        ax2.set_ylim(ylim2)

    if not logx:
        ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=base_major))
        ax2.xaxis.set_major_locator(plticker.MultipleLocator(base=base_major))
        ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=base_minor))
        ax2.xaxis.set_minor_locator(plticker.MultipleLocator(base=base_minor))
    ax1.tick_params('both', direction='out', which='both')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.legend(labelspacing=0.2)
    ax2.tick_params('both', direction='out', which='both')
    if y2_sci:
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax2.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax2.legend(labelspacing=0.2, loc='upper left')

    return ax1, ax2


def plot_matrix(m_, xlabel_, ylabel_, label_='', min_=None, max_=None, cmap_='viridis', origin_='lower',
                figsize_=[6.15, 5.], title_=None):

    fig = plt.figure(figsize=figsize_, constrained_layout=True)
    ax_ = fig.add_subplot(111)

    if min_ is None:
        min_ = m_.min()

    if max_ is None:
        max_ = m_.max()

    cmap = cm.get_cmap(cmap_)
    normalizer = Normalize(min_, max_)
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    ax_.imshow(m_, origin=origin_, norm=normalizer, cmap=cmap)
    ax_.set_xlabel(xlabel_)
    ax_.set_ylabel(ylabel_)
    if title_ is not None:
        ax_.set_title(title_)
    # ax_.colorbar()

    cbar = fig.colorbar(im, ax=ax_, label=label_)
    cbar.formatter.set_powerlimits((0, 0))

    return ax_
