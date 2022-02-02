import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import json
# import pandas as pd
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')
import latex
from plot import plot_function
from reactor import UnoscillatedReactorSpectrum


### MAIN ###
time_start = time.perf_counter_ns()

f = open('Inputs/nufit_inputs.json')
# f = open('Inputs/nominal_inputs.json')
# f = open('Inputs/YB_inputs.json')
inputs_json = json.load(f)

# E = np.arange(1.806, 10.01, 0.01)  # in MeV
# E = np.arange(1., 12.01, 0.01)  # in MeV
E = np.arange(1.925, 8.65, 0.01)  # in MeV (for DYB reactor model)


### REACTOR SPECTRUM
react = UnoscillatedReactorSpectrum(inputs_json)
flux_v = react.isotopic_spectrum_vogel_parametric(E, plot_this=False)
flux_hm = react.isotopic_spectrum_hubermueller_parametric(E, plot_this=False)
flux_dyb = react.isotopic_spectrum_dyb(E, plot_this=False)
reactor_spectrum_v = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='V', plot_this=False)
reactor_spectrum_hm = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', plot_this=False)
reactor_spectrum_dyb = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='DYB', plot_this=False)

nominal = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', plot_this=False)
nominal_snf = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', bool_snf=True, plot_this=False)
nominal_noneq = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', bool_noneq=True, plot_this=False)
nominal_snf_noneq = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', bool_snf=True, bool_noneq=True, plot_this=False)
nominal_hm = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='DYB', plot_this=False)

xsec_sv = react.eval_xs(E, which_xs="SV_CI")
# xsec_vb = react.cross_section_vb(E)
# xsec = react.cross_section(E)

params_u235 = [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]
params_pu239 = [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]
params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]

u235 = react.isotopic_spectrum_exp(E, params_u235)
pu239 = react.isotopic_spectrum_exp(E, params_pu239)
u238 = react.isotopic_spectrum_exp(E, params_u238)
pu241 = react.isotopic_spectrum_exp(E, params_pu241)

dyb_correction = react.get_dybfluxbump_ratio(E)
hm_corrected = reactor_spectrum_hm * dyb_correction

plot_function(x_=[E, E, E, E, E, E, E],
              y_=[flux_hm, react.fiss_frac_235u * u235, react.fiss_frac_239pu * pu239, react.fiss_frac_238u * u238,
                  react.fiss_frac_241pu * pu241, xsec_sv*2.e8, nominal*2500.],
              label_=[r'total', r'$^{235}$U', r'$^{239}$Pu', r'$^{238}$U', r'$^{241}$Pu', r'$\sigma_{\text{IBD}}$',
                      r'$\bar{\nu}$ spectrum'],
              styles=['m', 'b--', 'r-.', 'g:', 'y', 'c', 'k'],
              ylabel_='', xlim=None, ylim=None)

plot_function(x_=[E, E, E], y_=[flux_v, flux_hm, flux_dyb], label_=[r'Vogel', r'H + M', r'DYB'], styles=['g-', 'r--', 'k'],
              ylabel_=r'isotopic spectrum [\#$\nu$/fission/MeV]', xlabel_=r'nu energy [MeV]', xlim=None, ylim=None)

plot_function(x_=[E, E, E], y_=[reactor_spectrum_v, reactor_spectrum_hm, reactor_spectrum_dyb],
              label_=[r'Vogel', r'H + M', r'DYB'], styles=['g-', 'r--', 'k'],
              ylabel_=r'antinu spectrum [\#$\nu$/s/MeV]', xlabel_=r'nu energy [MeV]', xlim=None, ylim=None)

plot_function(x_=[E, E, E, E], y_=[nominal, nominal_snf, nominal_noneq, nominal_snf_noneq],
              label_=[r'nominal', r'SNF', r'NonEq', r'SNF+NonEq'], styles=['k', 'b:', 'r-', 'g--'],
              ylabel_=r'antinu spectrum [\#$\nu$/s/MeV]', xlabel_=r'nu energy [MeV]', xlim=None, ylim=None)

plot_function(x_=[E, E, E, E],
              y_=[nominal, (nominal_snf-nominal)/nominal*100., (nominal_noneq-nominal)/nominal*100.,
                  (nominal_snf_noneq-nominal)/nominal*100.],
              label_=[r'nominal', r'SNF', r'NonEq', r'SNF+NonEq'], styles=['k', 'b:', 'r-', 'g--'],
              ylabel_=r'(corr - nom)/nom [\%]', xlabel_=r'nu energy [MeV]', xlim=None, ylim=None)

plot_function(x_=[E, E, E], y_=[reactor_spectrum_hm, hm_corrected, reactor_spectrum_dyb],
              label_=[r'HM', r'HM - corrected', r'DYB'], styles=['g:', 'r--', 'k'],
              ylabel_=r'reactor spectrum [\#$\nu$/s/MeV]', xlabel_=r'nu energy [MeV]', xlim=None, ylim=None)

elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
