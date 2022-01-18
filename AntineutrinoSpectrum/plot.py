import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np


def plot_function(x_, y_, label_, ylabel_, styles=None, xlabel_=r'$E_{\nu}$ [\si{MeV}]', xlim=None, ylim=None, logx=False):
    if len(x_) != len(y_):
        print("Error in plot_function: different lengths of x array and y array - skip plotting")
        return 1

    loc = plticker.MultipleLocator(base=2.0)
    loc1 = plticker.MultipleLocator(base=0.5)

    fig = plt.figure(figsize=[7., 4.])
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.14)
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
        ax_.xaxis.set_major_locator(loc)
        ax_.xaxis.set_minor_locator(loc1)
    ax_.tick_params('both', direction='out', which='both')
    ax_.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_.legend(labelspacing=0.2)

    return ax_


def plot_function_residual(x_, y_, label_, ylabel_, styles=None, xlabel_=r'$E_{\nu}$ [\si{MeV}]', xlim=None, ylim=None, ylim2=None, logx=False):
    if len(x_) != len(y_):
        print("Error in plot_function: different lengths of x array and y array - skip plotting")
        return 1

    loc = plticker.MultipleLocator(base=2.0)
    loc1 = plticker.MultipleLocator(base=0.5)
    if styles is None:
        styles = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    fig, (ax_, ax2) = plt.subplots(2, 1, figsize=[7, 6], gridspec_kw={'height_ratios': [2, 1]})
    # fig = plt.figure(figsize=[7, 8])
    fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.093)

    # ax_ = fig.add_subplot(211)
    for i_ in np.arange(len(x_)):
        if not logx:
            ax_.plot(x_[i_], y_[i_], styles[i_], linewidth=1., label=label_[i_], markersize=4)
        else:
            ax_.semilogx(x_[i_], y_[i_], styles[i_], linewidth=1., label=label_[i_])
    
    # ax2 = fig.add_subplot(212)
    for j_ in range(1, len(x_)):
        if len(x_[0]) == len(x_[j_]):
            ll = r"("+label_[j_]+" - "+label_[0]+")/"+label_[0]
            if not logx:
                ax2.plot(x_[0], (y_[j_] - y_[0])/y_[0]*100., styles[j_], linewidth=1., label=ll)
            else:
                ax2.semilogx(x_[0], (y_[j_] - y_[0])/y_[0]*100., styles[j_], linewidth=1., label=ll)
        else:
            continue

    ax_.grid(alpha=0.65)
    # ax_.set_xlabel(xlabel_)
    ax_.set_ylabel(ylabel_)
    ax2.grid(alpha=0.65)
    ax2.set_xlabel(xlabel_)
    ax2.set_ylabel(r'relative difference [\%]')

    if xlim is not None:
        ax_.set_xlim(xlim)
        ax2.set_xlim(xlim)
    if ylim is not None:
        ax_.set_ylim(ylim)
    if ylim2 is not None:
        ax2.set_ylim(ylim2)

    if not logx:
        ax_.xaxis.set_major_locator(loc)
        ax2.xaxis.set_major_locator(loc)
        ax_.xaxis.set_minor_locator(loc1)
        ax2.xaxis.set_minor_locator(loc1)
    ax_.tick_params('both', direction='out', which='both')
    ax_.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_.legend(labelspacing=0.2)
    ax2.tick_params('both', direction='out', which='both')
    ax2.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax2.legend(labelspacing=0.2, loc='upper left')

    return ax_, ax2
