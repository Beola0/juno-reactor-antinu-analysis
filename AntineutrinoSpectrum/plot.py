import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

modes = ['log', 'pow_law', 'power_law', 'powerlaw', 'centered', 'norm']


def _forward(x):
    scale_ = 0.5
    return np.sign(x) * np.abs(x)**scale_


def _inverse(x):
    scale_ = 0.5
    return np.sign(x) * np.abs(x)**(1./scale_)


def plot_function(x_, y_, ylabel_, label_='', styles=None, xlabel_=r'$E_{\nu}$ [MeV]', y_sci=False,
                  xlim=None, ylim=None, logx=False, fig_length=7, fig_height=4, base_major=2.0, base_minor=0.5,
                  constrained_layout_=False):
    if len(x_) != len(y_):
        raise ValueError(f"Error in plot_function: different lengths of x array ({len(x_)}) and y array ({len(y_)})")

    fig = plt.figure(figsize=[fig_length, fig_height], constrained_layout=constrained_layout_)
    if not constrained_layout_:
        fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.14)
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
    if y_sci:
        ax_.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax_.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax_.legend(labelspacing=0.2)

    return ax_


def plot_function_residual(x_, y_, label_, ylabel_, styles=None, xlabel_=r'$E_{\nu}$ [MeV]',
                           xlim=None, ylim=None, ylim2=None, logx=False, y2_sci=False,
                           fig_length=7, fig_height=6, base_major=2.0, base_minor=0.5,
                           constrained_layout_=False):
    if len(x_) != len(y_):
        raise ValueError(f"Different lengths of x array ({len(x_)}) and y array ({len(y_)})")

    if styles is None:
        styles = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[fig_length, fig_height], gridspec_kw={'height_ratios': [2, 1]},
                                   sharex=True, constrained_layout=constrained_layout_)
    if not constrained_layout_:
        fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.093)

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


def plot_matrix(m_, xlabel_, ylabel_, label_='', min_=None, max_=None, cmap_='seismic', origin_='lower',
                fig_length=6.15, fig_height=5, title_=None, mode_='centered', sci_mode_=False, shrink=1.,
                constrained_layout_=False):

    fig = plt.figure(figsize=[fig_length, fig_height], constrained_layout=constrained_layout_)
    ax_ = fig.add_subplot(111)

    if min_ is None:
        min_ = m_.min()

    if max_ is None:
        max_ = m_.max()

    limit = max(np.abs(min_), max_)
    cmap = cm.get_cmap(cmap_)

    if mode_ == 'log':
        norm = colors.LogNorm(vmin=min_, vmax=max_)
    elif mode_ == 'pow_law' or mode_ == 'power_law' or mode_ == 'powerlaw':
        norm = colors.FuncNorm((_forward, _inverse), vmin=-limit, vmax=limit)
    elif mode_ == 'centered':
        norm = colors.CenteredNorm()
    elif mode_ == 'norm':
        norm = colors.Normalize(vmin=min_, vmax=max_)
    else:
        raise ValueError("Invalid mode_ value. Expected one of: %s" % modes)

    im = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax_.imshow(m_, origin=origin_, norm=norm, cmap=cmap)

    ax_.set_xlabel(xlabel_)
    ax_.set_ylabel(ylabel_)
    if title_ is not None:
        ax_.set_title(title_)

    cbar = fig.colorbar(im, ax=ax_, label=label_, shrink=shrink)
    if sci_mode_:
        cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_offset_position('left')

    return ax_


def plot_two_matrices(m_, xlabel_, ylabel_, label_='', min_=None, max_=None, cmap_='seismic', origin_='lower',
                      fig_length=9, fig_height=6, titles_=None, mode_='centered', sci_mode_=False, shrink=0.7,
                      constrained_layout_=False):
    if len(m_) != 2:
        raise ValueError("Invalid input length. Expected input with two matrices.")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[fig_length, fig_height], sharex=True, sharey=True,
                             constrained_layout=constrained_layout_)

    if min_ is None:
        min_ = min(m_[0].min(), m_[1].min())

    if max_ is None:
        max_ = min(m_[0].max(), m_[1].max())

    limit = max(np.abs(min_), max_)
    cmap = cm.get_cmap(cmap_)

    if mode_ == 'log':
        norm = colors.LogNorm(vmin=min_, vmax=max_)
    elif mode_ == 'pow_law' or mode_ == 'power_law' or mode_ == 'powerlaw':
        norm = colors.FuncNorm((_forward, _inverse), vmin=-limit, vmax=limit)
    elif mode_ == 'centered':
        norm = colors.CenteredNorm()
    elif mode_ == 'norm':
        norm = colors.Normalize(vmin=min_, vmax=max_)
    else:
        raise ValueError("Invalid mode_ value. Expected one of: %s" % modes)

    im = cm.ScalarMappable(norm=norm, cmap=cmap)
    axes[0].imshow(m_[0], origin=origin_, norm=norm, cmap=cmap)
    axes[1].imshow(m_[1], origin=origin_, norm=norm, cmap=cmap)

    axes[0].set_xlabel(xlabel_)
    axes[1].set_xlabel(xlabel_)
    axes[0].set_ylabel(ylabel_)
    if titles_ is not None:
        axes[0].set_title(titles_[0])
        axes[1].set_title(titles_[1])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=label_, shrink=shrink)
    if sci_mode_:
        cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_offset_position('left')

    return axes


def plot_six_matrices(m_, xlabel_, ylabel_, label_='', min_=None, max_=None, cmap_='seismic', origin_='lower',
                      fig_length=9, fig_height=6, titles_=None, mode_='centered', sci_mode_=False, shrink=0.7,
                      constrained_layout_=False):
    if len(m_) > 6:
        raise ValueError("Invalid input length. Expected input with no more than 6 matrices.")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[fig_length, fig_height], sharex=True, sharey=True,
                             constrained_layout=constrained_layout_)

    if min_ is None:
        min_ = m_[-1].min()

    if max_ is None:
        max_ = m_[-1].max()

    limit = max(np.abs(min_), max_)
    cmap = cm.get_cmap(cmap_)

    if mode_ == 'log':
        norm = colors.LogNorm(vmin=min_, vmax=max_)
    elif mode_ == 'pow_law' or mode_ == 'power_law' or mode_ == 'powerlaw':
        norm = colors.FuncNorm((_forward, _inverse), vmin=-limit, vmax=limit)
    elif mode_ == 'centered':
        norm = colors.CenteredNorm()
    elif mode_ == 'norm':
        norm = colors.Normalize(vmin=min_, vmax=max_)
    else:
        raise ValueError("Invalid mode_ value. Expected one of: %s" % modes)

    im = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i_, ax in enumerate(axes.flat):
        try:
            ax.imshow(m_[i_], origin=origin_, cmap=cmap, norm=norm)
        except IndexError:
            continue
        ax.set_title(titles_[i_])
        if i_ == 3 or i_ == 4 or i_ == 5:
            ax.set_xlabel(xlabel_)
        if i_ == 0 or i_ == 3:
            ax.set_ylabel(ylabel_)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=label_, shrink=shrink)
    if sci_mode_:
        cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_offset_position('left')

    if len(m_) < 6:
        axes[-1, -1].axis('off')
    if len(m_) < 5:
        axes[-1, -2].axis('off')

    return axes
