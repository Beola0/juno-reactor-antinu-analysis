import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import json
# import math
# import pandas as pd
from scipy.interpolate import interp1d, Akima1DInterpolator, UnivariateSpline
from scipy import integrate
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')
import latex
from plot import plot_function, plot_function_residual
from reactor import UnoscillatedReactorSpectrum
from oscillation import OscillationProbability

delta = 1.293  # MeV
m_p = 938.272  # MeV


def get_e2(x_):
    return x_ - delta


def get_e1(x_):
    return x_ - delta - 2.*(x_-delta)*x_/m_p


def eval_xs_strumiavissani_approx(nu_energy_, e_energy_):

    alpha = -0.07056
    beta = 0.02018
    gamma = -0.001953
    # delta = 1.293  # MeV, mass(n)-mass(p)
    m_e = 0.510999  # MeV
    const = 1.e-43  # cm^2

    # positron_energy = np.subtract(nu_energy_, delta)  # positron's energy
    positron_energy = e_energy_

    appo = np.power(positron_energy, 2) - m_e ** 2
    p_e = np.sqrt(appo)  # positron's momentum

    appo_exp = alpha + beta * np.log(nu_energy_) + gamma * np.power(np.log(nu_energy_), 3)
    energy_exp = np.power(nu_energy_, appo_exp)

    return const * p_e * positron_energy * energy_exp


E = np.arange(1.81, 10.01, 0.005)  # in MeV

E_2 = 6
E_e_mean = (get_e1(E_2)+get_e2(E_2))/2.
E_e = np.arange(get_e1(E_2), get_e2(E_2), 0.0005)
appo = np.zeros(len(E_e))
for i_ in np.arange(len(E_e)):
    appo[i_] = eval_xs_strumiavissani_approx(E_2, E_e[i_])


plot_function(
    x_=[E, E], y_=[E-get_e1(E), E-get_e2(E)], label_=[r'E-E1', r'E-E2'], styles=['r', 'b'],
    ylabel_=r'E$_{\nu}$-E$_e$ [\si{\MeV}]', ylim=[1.2, 1.5]
)

plot_function(
    x_=[E_e-E_e_mean], y_=[appo/eval_xs_strumiavissani_approx(E_2, E_2-delta)], label_=[r'2 MeV'],
    ylabel_=r'sigma', xlabel_=r'$\Delta$ Ee', xlim=[-0.1, 0.1]
)



plt.ion()
plt.show()
