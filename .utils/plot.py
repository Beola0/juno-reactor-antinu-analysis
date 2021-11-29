import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import time
import math

import matplotlib as mpl
from matplotlib import rc

# plt.rc('text',usetex=True)
mpl.rcParams['text.latex.preamble'] = ['\\usepackage{siunitx}']
# mpl.rcParams['text.latex.preamble'] = ['\\usepackage{mhchem}']
pgf_with_latex = {                      # setup matplotlib to use latex for output# {{{
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ['Computer Modern'],                   # blank entries should cause plots
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 16,               # LaTeX default is 10pt font.
    "font.size": 16,
    "legend.fontsize": 13,               # Make the legend/label fonts
    "xtick.labelsize": 13,               # a little smaller
    "ytick.labelsize": 13,
#    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts
        r"\usepackage[T1]{fontenc}",        # plots will be generated
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ]                                   # using this preamble
    }
# }}}
mpl.rcParams.update(pgf_with_latex)

cwd = os.getcwd()

sys.path.insert(0, cwd + '/AntineutrinoSpectrum')

# import latex
from reactor import ReactorSpectrum
# from oscillation import OscillationProbability
from spectrum import OscillatedSpectrum
# from convolution import Convolution


def sin2(x, dm2):

    appo = 1.27 * dm2 * x  # x in [m/MeV]
    return np.power(np.sin(appo), 2)

##### MAIN PROGRAM #####

time_start = time.perf_counter_ns()

E = np.arange(1.806, 10.01, 0.01)  # in MeV
# change step from 0.01 to 0.001 or viceversa

react = ReactorSpectrum()
# flux = react.flux(E, plot_this=False)
# xsec = react.cross_section(E, plot_this=False)
reactor_spectrum = react.unosc_spectrum(E, plot_this=False)

# spectrum = OscillatedSpectrum(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3,
#                               t13_I=0.02263, m3l_I=-2.512e-3)  # 2019
spectrum = OscillatedSpectrum(t12=0.307, m21=7.54e-5, t13_N=0.0241, m3l_N=2.468e-3,
                              t13_I=0.0241, m3l_I=-2.458e-3)  # 2012
spectrum_N, spectrum_I = spectrum.osc_spectrum(E, 0, plot_this=False, normalize=False,
                                               plot_un=False)  # 1 for NO, -1 for IO, 0 for both (plotting)

spectrum_theta13zero = spectrum.eval_NO(E, 0.310, 7.39e-5, 0, 2.525e-3)

baseline = 52
x_E = baseline * 1000 / E
A_N = math.pow(1 - 0.02240, 2) * 4. * 0.310 * (1 - 0.310)
spectrum_theta13nominal = (1 - A_N * sin2(x_E, 7.39e-5)) * reactor_spectrum

loc = plticker.MultipleLocator(base=2.0)  # this locator puts ticks at regular intervals
loc1 = plticker.MultipleLocator(base=0.5)
loc2 = plticker.MultipleLocator(base=0.05)
loc3 = plticker.MultipleLocator(base=0.1)

fig = plt.figure(figsize=[8., 5.5])
# fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.10, right=0.97, top=0.97)
ax.plot(E, reactor_spectrum, 'k', linewidth=1., label=r'Unoscillated spectrum')
# ax.plot(E, spectrum_theta13zero, 'g-.', linewidth=1.5, label=r'$P_{ee} = 1 - \sin^2(2\theta_{12}) \sin^2 (\Delta_{21})$')
# ax.plot(E, spectrum_theta13nominal, 'g-.', linewidth=1., label=r'$P_{ee} = 1 - \cos^4(\theta_{13}) \sin^2(2\theta_{12}) \sin^2 (\Delta_{21})$')
ax.plot(E, spectrum_N, 'b', linewidth=1., label=r'Normal ordering')
ax.plot(E, spectrum_I, 'r--', linewidth=1., label=r'Inverted ordering')
ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
ax.set_xlim(0., 10.5)
ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
ax.set_ylim(0., 0.35)
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_minor_locator(loc1)
# ax.set_title(r'')
ax.legend()
ax.grid(alpha=0.45)
# ax.arrow(0, 0.03, 2.98, 0.0, head_width=0.01, head_length=0.2, length_includes_head=True,
#          linewidth=1.5, label=r'$\Delta m^2_{21}$', color='k')
ax.annotate("", xy=(2.98, 0.03), xytext=(0., 0.03), arrowprops=dict(arrowstyle="<|-|>", shrinkA=0., shrinkB=0.,
                                                                    linewidth=1., color='k'))
ax.annotate("", xy=(2.98, 0.040), xytext=(2.98, 0.264), arrowprops=dict(arrowstyle="<|-|>", shrinkA=0., shrinkB=0.,
                                                                        linewidth=1., color='k'))
ax.text(0.46, 0.033, r'$\Delta m^2_{21}$', fontsize=14)
ax.text(3.05, 0.155, r'$\sin^2 (2\theta_{12})$', fontsize=14)
# fig.savefig('plot_theta13/Fig_theta13_nominal.pdf', format='pdf', transparent=True)
fig.savefig('Fig.pdf', format='pdf', transparent=True)
# fig.savefig('Fig_noarrow_2012.pdf', format='pdf', transparent=True)
# fig.savefig('unoscillated.pdf', format='pdf', transparent=True)

elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
