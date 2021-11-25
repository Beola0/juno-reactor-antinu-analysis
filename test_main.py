import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import time
import json
import pandas as pd

cwd = os.getcwd()

sys.path.insert(0, cwd + '/AntineutrinoSpectrum')

import latex
from reactor import ReactorSpectrum
from oscillation import OscillationProbability
from spectrum import OscillatedSpectrum


def plot_function(x_, y_, label_, ylabel_, styles=None, xlabel_=r'$E_{\nu}$ [\si{MeV}]', xlim=None, ylim=None, logx=False):
    if len(x_) != len(y_):
        print("Error in plot_function: different lengths - skip plotting")
        return 1

    loc = plticker.MultipleLocator(base=2.0)
    loc1 = plticker.MultipleLocator(base=0.5)

    fig = plt.figure(figsize=[8, 5.5])
    fig.subplots_adjust(left=0.09, right=0.97, top=0.95)
    ax_ = fig.add_subplot(111)
    if styles is None:
        styles = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    for i_ in np.arange(len(x_)):
        if not logx:
            ax_.plot(x_[i_], y_[i_], styles[i_], linewidth=1., label=label_[i_])
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
    ax_.legend()

    return ax_


### MAIN ###
time_start = time.perf_counter_ns()

f = open('Inputs/nufit_inputs.json')
# f = open('Inputs/nominal_inputs.json')
# f = open('Inputs/YB_inputs.json')
inputs_json = json.load(f)

# E = np.arange(1.806, 10.01, 0.01)  # in MeV
# E = np.arange(1., 12.01, 0.01)  # in MeV
E = np.arange(1.925, 8.65, 0.01)  # in MeV (for DYB)

'''
### REACTOR SPECTRUM
react = ReactorSpectrum(inputs_json)
flux_v = react.isotopic_spectrum_vogel(E, plot_this=False)
flux_hm = react.isotopic_spectrum_hubermueller(E, plot_this=False)
flux_dyb = react.isotopic_spectrum_DYB(E, plot_this=False)
reactor_spectrum_v = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='V', plot_this=True)
reactor_spectrum_hm = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='HM', plot_this=True)
reactor_spectrum_dyb = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', plot_this=True)

nominal = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', plot_this=False)
nominal_snf = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', bool_snf=True, plot_this=False)
nominal_noneq = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', bool_noneq=True, plot_this=False)
nominal_snf_noneq = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', bool_snf=True, bool_noneq=True, plot_this=False)
nominal_hm = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='HM', plot_this=False)

xsec_sv = react.cross_section_sv(E)
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

plot_function(x_=[E, E, E, E, E, E, E],
              y_=[flux_hm, react.fiss_frac_235u * u235, react.fiss_frac_239pu * pu239, react.fiss_frac_238u * u238,
                  react.fiss_frac_241pu * pu241, xsec_sv*2.e8, nominal_hm*2500.],
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
'''

'''
### OSCILLATION PROBABILITY 
prob = OscillationProbability(inputs_json)
prob_N_v, prob_I_v = prob.eval_vacuum_prob(plot_this=True)
prob_N_ve, prob_I_ve = prob.eval_vacuum_prob(E, plot_this=False)

xxx = np.arange(0.01, 500000, 1.)
yyy = (prob_N_v-prob_I_v)/prob_N_v
plot_function(x_=[xxx/1000.], y_=[yyy], label_=[r'NO-IO'], ylabel_=r'(NO-IO)/NO [\%]', styles=['k'], xlim=[0.04, 70],
              xlabel_=r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]', logx=True)

prob_N_m, prob_I_m = prob.eval_matter_prob(plot_this=False)
prob_N_me, prob_I_me = prob.eval_matter_prob(E, plot_this=False)

xxx = np.arange(0.01, 500000, 1.)
yyy = (prob_N_m-prob_I_m)/prob_N_m
plot_function(x_=[xxx/1000.], y_=[yyy], label_=[r'NO-IO'], ylabel_=r'(NO-IO)/NO [\%] (in matter)', styles=['k'],
              xlim=[0.04, 50], xlabel_=r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]', logx=True)

ax = plot_function(x_=[E, E], y_=[(prob_N_me-prob_N_ve)/prob_N_me*100., (prob_I_me-prob_I_ve)/prob_I_me*100.],
                   label_=[r'NO', r'IO'], styles=['b-', 'r--'],
                   ylabel_=r'$(P_{\text{mat}} - P_{\text{vac}}) / P_{\text{mat}}$ [\si{\percent}]',
                   xlabel_=r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]', xlim=None, ylim=None)
ax.axvline(1.806, 0, 1, color='k', linestyle=':')

ax1 = plot_function(x_=[np.arange(0.01, 500000, 1.)/1000, np.arange(0.01, 500000, 1.)/1000],
                    y_=[(prob_N_m-prob_N_v)/prob_N_m*100., (prob_I_m-prob_I_v)/prob_I_m*100.],
                    label_=[r'NO', r'IO'], styles=['b-', 'r--'],
                    ylabel_=r'$(P_{\text{mat}} - P_{\text{vac}}) / P_{\text{mat}}$ [\si{\percent}]',
                    xlabel_=r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]', xlim=[0.,40], ylim=[-1,4])
ax1.axvline(1.806, 0, 1, color='k', linestyle=':')
'''

print('\nOSCILLATED SPECTRUM\n')
### OSCILLATED SPECTRUM
spectrum = OscillatedSpectrum(inputs_json)
# s_N, s_I = spectrum.osc_spectrum(E, plot_this=True, which_isospectrum='HM', plot_un=False, runtime=False, matter=False)
# a = spectrum.osc_spectrum_no(E, plot_this=True, plot_un=False, runtime=True, matter=False)
# b = spectrum.osc_spectrum_no(E, plot_this=True, plot_un=False, runtime=False)
# s_N_m, s_I_m = spectrum.osc_spectrum(E, plot_this=False, plot_un=True, matter=True)

# sres_N, sres_I = spectrum.resol_spectrum(E-0.78, matter=False, plot_this=False)
# sres_N_m, sres_I_m = spectrum.resol_spectrum(E-0.78, matter=True, plot_this=False)

# s_N_v = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_isospectrum='V', matter=True)
# s_N_hm = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_isospectrum='HM', matter=True)
# s_N_dyb = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_isospectrum='DYB', matter=True)

# plot_function(x_=[E-0.78, E-0.78], y_=[sres_N, sres_N_m],
#               label_=[r'NO - vacuum', r'NO - matter'], styles=['b-', 'r--'],
#               ylabel_=r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]',
#               xlabel_=r'$E_{\text{vis}}$ [\si{\MeV}]', xlim=None, ylim=None)
#
# plot_function(x_=[E-0.78, E-0.78, E-0.78], y_=[s_N_dyb, s_N_hm, s_N_v],
#               label_=[r'NO - DYB', r'NO - HM', r'NO - V'], styles=['k', 'r--', 'g-.'],
#               ylabel_=r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]',
#               xlabel_=r'$E_{\text{vis}}$ [\si{\MeV}]', xlim=None, ylim=None)

# r_list = pd.read_csv("Inputs/list_of_reactors.txt", sep=",",
#                      names=["baseline", "thermal_power", "name"], header=0)
# nn = len(r_list["baseline"])
# ee = []
# ss = []
# for j_ in np.arange(nn):
#     spectrum.set_baseline(r_list["baseline"][j_])
#     spectrum.set_th_power(r_list["thermal_power"][j_])
#     appo = spectrum.antinu_spectrum_no_osc(E)
#     ee.append(E)
#     ss.append(appo)
#
# ee1 = []
# ss1 = []
# for j_ in np.arange(nn):
#     spectrum.set_baseline(r_list["baseline"][j_])
#     spectrum.set_th_power(r_list["thermal_power"][j_])
#     appo = spectrum.eval_vacuum_prob_no(E)
#     ee1.append(E)
#     ss1.append(appo)

# plot_function(x_=ee, y_=ss, label_=r_list["name"], ylabel_=r'$S_{\bar{\nu}}$ [N$_{\nu}$/\si{\MeV}/\si{s}]', xlim=[1.5,10])
# plot_function(x_=ee1, y_=ss1, label_=r_list["name"], ylabel_=r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$', xlim=[1.5,10])

aa, ssa = spectrum.osc_spectrum_no(E, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=False,
                                    plot_this=True, plot_un=True, plot_singles=True, runtime=False)
bb, ssb = spectrum.osc_spectrum_io(E, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=False,
                                    plot_this=True, plot_un=True, plot_singles=True, runtime=False)
a, b = spectrum.osc_spectrum(E, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=False,
                                    plot_this=True, plot_un=True, runtime=False)
# c, ssc = spectrum.osc_spectrum_sum_no(E, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=False,
#                                       plot_sum=True, plot_baselines=True, runtime=False)
# d, ssd = spectrum.resol_spectrum_sum_no(E-0.78, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=False,
#                                         plot_sum=False, plot_baselines=False, runtime=False)

elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
