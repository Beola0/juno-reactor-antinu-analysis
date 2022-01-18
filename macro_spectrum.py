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
from spectrum import OscillatedSpectrum


### MAIN ###
time_start = time.perf_counter_ns()

# f = open('Inputs/nufit_inputs.json')
f = open('Inputs/nominal_inputs.json')
# f = open('Inputs/YB_inputs.json')
inputs_json = json.load(f)

# E = np.arange(1.806, 10.01, 0.01)  # in MeV
# E = np.arange(1., 12.01, 0.01)  # in MeV
E = np.arange(1.925, 8.65, 0.01)  # in MeV (for DYB)


### OSCILLATED SPECTRUM
spectrum = OscillatedSpectrum(inputs_json)
s_N, s_I = spectrum.osc_spectrum(E, plot_this=True, which_isospectrum='HM', plot_un=False, runtime=False, matter=False)
s_N_m, s_I_m = spectrum.osc_spectrum(E, plot_this=True, which_isospectrum='HM', plot_un=False, matter=True)

sres_N, sres_I = spectrum.resol_spectrum(E-0.78, which_isospectrum='HM', matter=False, plot_this=False)
sres_N_m, sres_I_m = spectrum.resol_spectrum(E-0.78, which_isospectrum='HM', matter=True, plot_this=False)

s_N_v = spectrum.osc_spectrum_no(E, plot_this=False, which_isospectrum='V', matter=True)
s_N_hm = spectrum.osc_spectrum_no(E, plot_this=False, which_isospectrum='HM', matter=True)
s_N_dyb = spectrum.osc_spectrum_no(E, plot_this=False, which_isospectrum='DYB', matter=True)
s_N_v_res = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_isospectrum='V', matter=True)
s_N_hm_res = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_isospectrum='HM', matter=True)
s_N_dyb_res = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_isospectrum='DYB', matter=True)

plot_function(x_=[E, E], y_=[s_N, s_N_m],
              label_=[r'NO - vacuum', r'NO - matter'], styles=['k-', 'g--'],
              ylabel_=r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]',
              xlabel_=r'$E_{\nu}$ [\si{\MeV}]', xlim=None, ylim=None)

plot_function(x_=[E-0.78, E-0.78], y_=[sres_N, sres_N_m],
              label_=[r'NO - vacuum', r'NO - matter'], styles=['k-', 'g--'],
              ylabel_=r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]',
              xlabel_=r'$E_{\text{vis}}$ [\si{\MeV}]', xlim=None, ylim=None)

plot_function(x_=[E, E, E], y_=[s_N_dyb, s_N_hm, s_N_v],
              label_=[r'NO - DYB', r'NO - HM', r'NO - V'], styles=['k', 'r--', 'g-.'],
              ylabel_=r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]',
              xlabel_=r'$E_{\nu}$ [\si{\MeV}]', xlim=None, ylim=None)

plot_function(x_=[E-0.78, E-0.78, E-0.78], y_=[s_N_dyb_res, s_N_hm_res, s_N_v_res],
              label_=[r'NO - DYB', r'NO - HM', r'NO - V'], styles=['k', 'r--', 'g-.'],
              ylabel_=r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]',
              xlabel_=r'$E_{\text{vis}}$ [\si{\MeV}]', xlim=None, ylim=None)

# aa = spectrum.osc_spectrum_no(E, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=True,
#                               plot_this=False, plot_un=False, plot_singles=True, runtime=False)
# cc = spectrum.resol_spectrum_no(E-0.78, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=True,
#                                 plot_this=False, plot_singles=True, runtime=False)

elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
