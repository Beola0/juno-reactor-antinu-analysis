import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import time
import json
# import pandas as pd
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')
import latex
from plot import plot_function
from oscillation import OscillationProbability


### MAIN ###
time_start = time.perf_counter_ns()

f = open('Inputs/nufit_inputs.json')  # NuFIT 2019
# f = open('Inputs/nominal_inputs.json')  # PDG 2020
# f = open('Inputs/YB_inputs.json')
inputs_json = json.load(f)

# E = np.arange(1.806, 10.01, 0.01)  # in MeV
E = np.arange(1., 12.01, 0.01)  # in MeV
# E = np.arange(1.925, 8.65, 0.01)  # in MeV (for DYB)


### OSCILLATION PROBABILITY
prob = OscillationProbability(inputs_json)
prob_N_v, prob_I_v = prob.eval_vacuum_prob(plot_this=True)
prob_N_ve, prob_I_ve = prob.eval_vacuum_prob(E, plot_this=True)

xxx = np.arange(0.01, 500000, 1.)
yyy = (prob_N_v-prob_I_v)/prob_N_v
plot_function(x_=[xxx/1000.], y_=[yyy], label_=[r'NO-IO (2019)'], ylabel_=r'(NO-IO)/NO [\%]', styles=['k'], xlim=[0.04, 70],
              xlabel_=r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]', logx=True)

prob_N_m, prob_I_m = prob.eval_matter_prob(plot_this=False)
prob_N_me, prob_I_me = prob.eval_matter_prob(E, plot_this=False)

# xxx = np.arange(0.01, 500000, 1.)
# yyy = (prob_N_m-prob_I_m)/prob_N_m
# plot_function(x_=[xxx/1000.], y_=[yyy], label_=[r'NO-IO (2019)'], ylabel_=r'(NO-IO)/NO [\%] (in matter)', styles=['k'],
#               xlim=[0.04, 50], xlabel_=r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]', logx=True)

ax = plot_function(x_=[E, E], y_=[(prob_N_me-prob_N_ve)/prob_N_me*100., (prob_I_me-prob_I_ve)/prob_I_me*100.],
                   label_=[r'NO', r'IO'], styles=['b-', 'r--'],
                   ylabel_=r'$(P_{\text{mat}} - P_{\text{vac}}) / P_{\text{mat}}$ [\si{\percent}]',
                   xlabel_=r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]', xlim=None, ylim=None)
ax.axvline(1.806, 0, 1, color='k', linestyle=':')

ax1 = plot_function(x_=[np.arange(0.01, 500000, 1.)/1000, np.arange(0.01, 500000, 1.)/1000],
                    y_=[(prob_N_m-prob_N_v)/prob_N_m*100., (prob_I_m-prob_I_v)/prob_I_m*100.],
                    label_=[r'NO', r'IO'], styles=['b-', 'r--'],
                    ylabel_=r'$(P_{\text{mat}} - P_{\text{vac}}) / P_{\text{mat}}$ [\si{\percent}]',
                    xlabel_=r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]', xlim=[0., 40], ylim=[-1, 4])
ax1.axvline(1.806, 0, 1, color='k', linestyle=':')


f = open('Inputs/nominal_inputs.json')  # PDG 2020
inputs_json = json.load(f)
prob = OscillationProbability(inputs_json)

prob_N_ve, prob_I_ve = prob.eval_vacuum_prob(E, plot_this=True)
prob_N_me, prob_I_me = prob.eval_matter_prob(E, plot_this=True)

xxx = np.arange(0.01, 500000, 1.)
yyy = (prob_N_v-prob_I_v)/prob_N_v
plot_function(x_=[xxx/1000.], y_=[yyy], label_=[r'NO-IO (2020)'], ylabel_=r'(NO-IO)/NO [\%]', styles=['k'], xlim=[0.04, 70],
              xlabel_=r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]', logx=True)

ax = plot_function(x_=[E, E], y_=[(prob_N_me-prob_N_ve)/prob_N_me*100., (prob_I_me-prob_I_ve)/prob_I_me*100.],
                   label_=[r'NO', r'IO'], styles=['b-', 'r--'],
                   ylabel_=r'$(P_{\text{mat}} - P_{\text{vac}}) / P_{\text{mat}}$ [\si{\percent}]',
                   xlabel_=r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]', xlim=[1.5, 10.5], ylim=[-0.4, 4])
ax.axvline(1.806, 0, 1, color='k', linestyle=':')


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
