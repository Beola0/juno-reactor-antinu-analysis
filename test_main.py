import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as plticker
import time
import json

cwd = os.getcwd()

sys.path.insert(0, cwd + '/AntineutrinoSpectrum')

import latex
from reactor import ReactorSpectrum
from oscillation import OscillationProbability
from spectrum import OscillatedSpectrum

time_start = time.perf_counter_ns()

f = open('Inputs/nufit_inputs.json')
inputs_json = json.load(f)

# E = np.arange(1.806, 10.01, 0.01)  # in MeV
# E = np.arange(1., 12.01, 0.01)  # in MeV
E = np.arange(1.925, 8.65, 0.01)  # in MeV


react = ReactorSpectrum(inputs_json)
flux_v = react.isotopic_spectrum_vogel(E, plot_this=False)
flux_hm = react.isotopic_spectrum_hubermueller(E, plot_this=False)
flux_dyb = react.isotopic_spectrum_DYB(E, plot_this=False)
# xsec = react.cross_section(E, plot_this=True)
reactor_spectrum_v = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='V', plot_this=False)
reactor_spectrum_hm = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='HM', plot_this=False)
reactor_spectrum_dyb = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', plot_this=False)

# nominal = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', plot_this=False)
# nominal_snf = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', bool_snf=True, plot_this=False)
# nominal_noneq = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', bool_noneq=True, plot_this=False)
# nominal_snf_noneq = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='DYB', bool_snf=True, bool_noneq=True, plot_this=False)
# nominal_hm = react.antinu_spectrum_no_osc(E, which_xsec='SV', which_isospectrum='HM', plot_this=False)

# appo = react.reactor_spectrum(E, plot_this=False)
# appo1 = react.reactor_flux_no_osc(E, plot_this=False)

# xsec_sv = react.cross_section_sv(E)
# xsec_vb = react.cross_section_vb(E)
# xsec = react.cross_section(E)

# params_u235 = [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]
# params_pu239 = [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]
# params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
# params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]
#
# u235 = react.isotopic_spectrum_exp(E, params_u235)
# pu239 = react.isotopic_spectrum_exp(E, params_pu239)
# u238 = react.isotopic_spectrum_exp(E, params_u238)
# pu241 = react.isotopic_spectrum_exp(E, params_pu241)
#
# plt.figure()
# plt.plot(E, flux_hm, 'm', linewidth=1.5, label='total')
# plt.plot(E, react.fiss_frac_235u * u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
# plt.plot(E, react.fiss_frac_239pu * pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
# plt.plot(E, react.fiss_frac_238u * u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
# plt.plot(E, react.fiss_frac_241pu * pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
# plt.plot(E, xsec_sv*2.e8, 'c', linewidth=1.5, label=r'$\sigma_{\text{IBD}}$')
# plt.plot(E, nominal_hm*2500., 'k', linewidth=1.5, label=r'$\bar{\nu}$ spectrum')
# plt.grid()
# # plt.legend()


plt.figure()
plt.plot(E, flux_v, 'g-.', linewidth=1., label='Vogel')
plt.plot(E, flux_hm, 'r--', linewidth=1., label='H + M')
plt.plot(E, flux_dyb, 'k', linewidth=1., label='DYB')
plt.xlabel(r'nu energy [MeV]')
plt.ylabel(r'isotopic spectrum [\#$\nu$/fission/MeV]')
plt.legend()
plt.grid()

plt.figure()
plt.plot(E, reactor_spectrum_v, 'g-.', linewidth=1., label='Vogel')
plt.plot(E, reactor_spectrum_hm, 'r--', linewidth=1., label='H + M')
plt.plot(E, reactor_spectrum_dyb, 'k', linewidth=1., label='DYB')
plt.xlabel(r'nu energy [MeV]')
plt.ylabel(r'antinu spectrum [\#$\nu$/s/MeV]')
plt.legend()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.grid()

# plt.figure()
# plt.plot(E, nominal, 'k', linewidth=1., label='nominal')
# plt.plot(E, nominal_snf, 'b:', linewidth=1., label='SNF')
# plt.plot(E, nominal_noneq, 'r-.', linewidth=1., label='NonEq')
# plt.plot(E, nominal_snf_noneq, 'g--', linewidth=1., label='SNF+NonEq')
# plt.xlabel(r'nu energy [MeV]')
# plt.ylabel(r'antinu spectrum [\#$\nu$/s/MeV]')
# plt.legend()
# plt.grid()
# 
# plt.figure()
# # plt.plot(E, nominal, 'k', linewidth=1., label='nominal')
# plt.plot(E, (nominal_snf-nominal)/nominal*100., 'b:', linewidth=1., label='SNF')
# plt.plot(E, (nominal_noneq-nominal)/nominal*100., 'r-.', linewidth=1., label='NonEq')
# plt.plot(E, (nominal_snf_noneq-nominal)/nominal*100., 'g--', linewidth=1., label='SNF+NonEq')
# plt.xlabel(r'nu energy [MeV]')
# plt.ylabel(r'(corr - nom)/nom [\%]')
# plt.legend()
# plt.grid()


'''
### OSCILLATION PROBABILITY 
prob = OscillationProbability(inputs_json)
prob_N_v, prob_I_v = prob.eval_vacuum_prob(plot_this=False)
prob_N_ve, prob_I_ve = prob.eval_vacuum_prob_energy(E, plot_this=False)

prob_N_m, prob_I_m = prob.eval_matter_prob(plot_this=False)
prob_N_me, prob_I_me = prob.eval_matter_prob_energy(E, plot_this=False)

plt.figure(figsize=[8, 5.5])
plt.semilogx(E, (prob_N_me-prob_N_ve)/prob_N_me*100., "b-", linewidth=1, label='NO')
plt.semilogx(E, (prob_I_me-prob_I_ve)/prob_I_me*100., "r--", linewidth=1, label='IO')
plt.xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
plt.ylabel(r'$(P_{\text{mat}} - P_{\text{vac}}) / P_{\text{mat}}$ [\si{\percent}]')
plt.axvline(1.806, 0, 1, color='k', linestyle=':')
plt.grid()
plt.legend()

plt.figure(figsize=[8, 5.5])
plt.plot(np.arange(0.01, 500000, 1.)/1000, (prob_N_m-prob_N_v)/prob_N_m*100., "b-", linewidth=1, label='NO')
plt.plot(np.arange(0.01, 500000, 1.)/1000, (prob_I_m-prob_I_v)/prob_I_m*100., "r--", linewidth=1, label='IO')
plt.xlim(0., 40)  # 0.04 - 100
plt.ylim(-1, 4)
plt.xlabel(r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]')
plt.ylabel(r'$(P_{\text{mat}} - P_{\text{vac}}) / P_{\text{mat}}$ [\si{\percent}]')
# plt.axvline(1.806, 0, 1, color='k', linestyle=':')
plt.grid()
plt.legend()
'''


### OSCILLATED SPECTRUM
spectrum = OscillatedSpectrum(inputs_json)
# spectrum_N, spectrum_I = spectrum.osc_spectrum_old(E, 0, plot_this=True, normalize=False, plot_un=True)
s_N, s_I = spectrum.osc_spectrum(E, plot_this=True, plot_un=True)
# s_N_m, s_I_m = spectrum.osc_spectrum(E, plot_this=True, plot_un=True, matter=True)

# sres_N = spectrum.resol_spectrum_N(E-0.78, plot_this=True)
sres_N, sres_I = spectrum.resol_spectrum(E-0.78, plot_this=True)
sres_N_m, sres_I_m = spectrum.resol_spectrum(E-0.78, matter=True, plot_this=True)


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
