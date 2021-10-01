import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import time

cwd = os.getcwd()

sys.path.insert(0, cwd + '/AntineutrinoSpectrum')

import latex
from reactor import ReactorSpectrum
from oscillation import OscillationProbability
from spectrum import OscillatedSpectrum
from convolution import Convolution

##### MAIN PROGRAM #####

time_start = time.perf_counter_ns()

E = np.arange(1.806, 10.01, 0.01)  # in MeV
# change step from 0.01 to 0.001 or viceversa

# creation of reactor's spectrum, class ReactorSpectrum
# set plot_this=True to plot the reactor's flux, the IBD cross-section and the reactor's spectrum, respectively

react = ReactorSpectrum()
flux = react.flux(E, plot_this=False)
xsec = react.cross_section(E, plot_this=False)
reactor_spectrum = react.unosc_spectrum(E, plot_this=False)

# evaluation of the survival probability, class OscillationProbability
# set plot_this_*=True to plot the survival probability
# the survival probability is plotted as function of L/E (plot_this_LE) or of E (plot_this_E)
# NO and N refer to the Normal Ordering; IO and I refer to the Inverted Ordering

# input: sin^2(theta_12), deltam^2_21, NO: sin^2(theta_13), deltam^2_3l, IO: sin^2(theta_13), deltam^2_3l
# values from JHEP01 (2019) 106, table 1 (http://www.nu-fit.org/?q=node/8) (see also for the notation)
prob = OscillationProbability(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3,
                              t13_I=0.02263, m3l_I=-2.512e-3)
prob_N, prob_I = prob.eval_prob(E, 0, plot_this_LE=False,
                                plot_this_E=False)  # 1 for NO, -1 for IO, 0 for both (plotting)

# creation of the oscillated spectrum, class OscillatedSpectrum
# set plot_this=True to plot the oscillated spectrum
# set plot_un=True to plot also the unoscillated spectrum

# parameters from nu-fit.org as above (JHEP01 (2019) 106, table 1, with SK)
# input: sin^2(theta_12), deltam^2_21, NO: sin^2(theta_13), deltam^2_3l, IO: sin^2(theta_13), deltam^2_3l
spectrum = OscillatedSpectrum(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3,
                              t13_I=0.02263, m3l_I=-2.512e-3)
# spectrum = OscillatedSpectrum(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3,
#                               t13_I=0.02240, m3l_I=-2.525e-3)
spectrum_N, spectrum_I = spectrum.osc_spectrum(E, 0, plot_this=True, normalize=True,
                                               plot_un=False)  # 1 for NO, -1 for IO, 0 for both (plotting)

# oscillated spectrum with experimental resolution via numerical convolution
# class Convolution used in the class Spectrum
# set plot_this=True to plot the numerical convolution
# set plot_start=True to plot the starting spectrum
# for further reference see https://arxiv.org/abs/1210.8141

# a = 0.029  # stochastic term
# b = 0.008  # constant term

a = 0.03
b = 0.

# convol = Convolution()
# conv_N = convol.numerical_conv_old(spectrum.norm_osc_spect_N, E, a=a, b=b, plot_this=False)
# conv_I = convol.numerical_conv_old(spectrum.norm_osc_spect_I, E, a=a, b=b, plot_this=False)

resol_spect_N, resol_spect_I = spectrum.resol_spectrum(E-0.8, a, b, 0, normalize=True,
                                                       plot_this=False)  # 1 for NO, -1 for IO, 0 for both (plotting)

loc = plticker.MultipleLocator(base=2.0)  # this locator puts ticks at regular intervals
loc1 = plticker.MultipleLocator(base=0.5)
loc2 = plticker.MultipleLocator(base=0.05)
loc3 = plticker.MultipleLocator(base=0.1)

# # fig = plt.figure(figsize=[10.5, 6.5])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# fig.subplots_adjust(left=0.12, right=0.96, top=0.95)
# ax.plot(E - 0.8, resol_spect_N, 'b', linewidth=1., label='NO')
# ax.plot(E - 0.8, resol_spect_I, 'r--', linewidth=1., label='IO')
# ax.set_xlabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
# ax.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
# ax.set_xlim(0.5, 10.5)
# ax.set_ylim(-0.005, 0.305)
# # ax.set_ylim(-0.005, 0.095)
# # ax.set_title(r'Antineutrino spectrum' + '\nwith finite energy resolution (\SI{3}{\percent} at \SI{1}{\MeV})')
# # text = r'$\frac{\sigma(\text{E})}{\text{E}} = \sqrt{ \left( \frac{\text{a}}{\sqrt{\text{E}}} \right)^2 + \text{b}^2 }$'
# # ax.text(7.05, 0.062, text)
# # ax.text(7.05, 0.052, r'a = \SI{%.1f}{\percent}' % (a * 100) + '\nb = \SI{%.1f}{\percent}' % (b * 100))
# ax.legend()
# # ax.legend(frameon=False)
# ax.grid(alpha=0.45)
# ax.xaxis.set_major_locator(loc)
# ax.xaxis.set_minor_locator(loc1)
# ax.yaxis.set_major_locator(loc2)
# # ax.yaxis.set_minor_locator(loc2)
# ax.tick_params('both', direction='out', which='both')
# # ax.tick_params('both', direction='in', which='both', right=True, top=True, grid_alpha=0.5)
# fig.savefig('oscillated_spectrum.pdf', format='pdf', transparent=True)
# print('\nThe plot has been saved in oscillated_spectrum.pdf')

# spectrum with true baseline distribution

baselines = np.array([52.75, 52.84, 52.42, 52.51, 52.12, 52.21, 52.76, 52.63, 52.32, 52.20, 215, 265])  # [km]
power_GW = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 4.6, 4.6, 17.4, 17.4])
sum_N_12, sum_I_12 = spectrum.sum(baselines, power_GW, E, plot_sum=False, normalize=True, plot_baselines=False)
# sum_res_N_12, sum_res_I_12 = spectrum.sum_resol(baselines, power_GW, E-0.8, a=0.029, b=0.008, normalize=True,
#                                                 plot_sum=False, plot_baselines=False)
sum_res_N_12, sum_res_I_12 = spectrum.sum_resol(baselines, power_GW, E-0.8, a=0.03, b=0., normalize=True,
                                                plot_sum=False, plot_baselines=False)

baselines = np.array([52.75, 52.84, 52.42, 52.51, 52.12, 52.21, 52.76, 52.63, 52.32, 52.20])  # [km]
power_GW = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 4.6, 4.6])
sum_N_10, sum_I_10 = spectrum.sum(baselines, power_GW, E, plot_sum=False, normalize=True, plot_baselines=False)
sum_res_N_10, sum_res_I_10 = spectrum.sum_resol(baselines, power_GW, E-0.8, a=0.029, b=0.008, normalize=True,
                                                plot_sum=False, plot_baselines=False)


# fig_c = plt.figure(figsize=[10.5, 6.5])
fig_c = plt.figure()
ax_c = fig_c.add_subplot(111)
fig_c.subplots_adjust(left=0.12, right=0.96, top=0.95)
ax_c.plot(E, spectrum_N, 'b', linewidth=1., label=r'1 core')
ax_c.plot(E, sum_N_10, 'r-.', linewidth=1., label=r'10 cores')
ax_c.plot(E, sum_N_12, 'g--', linewidth=1., label=r'12 cores')
# ax_c.plot(E, sum_I, 'r--', linewidth=1.5, label=r'IO')
ax_c.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
ax_c.set_xlim(1.5, 10.5)
ax_c.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
ax_c.set_ylim(-0.005, 0.305)
ax_c.xaxis.set_major_locator(loc)
ax_c.xaxis.set_minor_locator(loc1)
# ax_c.set_title(r'Antineutrino spectra with true baseline distribution')
ax_c.legend()
ax_c.grid(alpha=0.45)
fig_c.savefig('SpectrumPlots/1_10_12_baseline.pdf', format='pdf', transparent=True)

# fig_d = plt.figure(figsize=[10.5, 6.5])
fig_d = plt.figure()
ax_d = fig_d.add_subplot(111)
fig_d.subplots_adjust(left=0.12, right=0.96, top=0.95)
ax_d.plot(E - 0.8, resol_spect_N, 'b', linewidth=1., label=r'1 core')
ax_d.plot(E - 0.8, sum_res_N_10, 'r-.', linewidth=1., label=r'10 cores')
ax_d.plot(E - 0.8, sum_res_N_12, 'g--', linewidth=1., label=r'12 cores')
# ax_d.plot(E - 0.8, sum_res_I, 'r--', linewidth=1.5, label=r'IO')
ax_d.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
ax_d.set_xlim(0.5, 9.5)
ax_d.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
ax_d.set_ylim(-0.005, 0.305)
ax_d.xaxis.set_major_locator(loc)
ax_d.xaxis.set_minor_locator(loc1)
# ax_d.set_title(
#     r'Antineutrino spectra with true baseline distribution' + '\nwith energy resolution (\SI{3}{\percent} at \SI{1}{\MeV})')
ax_d.legend()
ax_d.grid(alpha=0.45)
fig_d.savefig('SpectrumPlots/1_10_12_baseline_resol.pdf', format='pdf', transparent=True)

''''# binning of the spectrum
events = 100000
# spectrum = Oscillated_Spectrum(t12=0.308, m21=7.54*10**(-5), t13_N=0.02340, m3l_N=2.47*10**(-3), t13_I=0.0240, m3l_I=7.54*10**(-5)-2.42*10**(-3))
# a = 0.029
# b = 0.008

delta_bins = 0.04  # [Mev]  # 40 keV step for 205 bins between 1 MeV and 9.2 MeV
bins_edges = np.arange(1.806, 10.01, delta_bins)
x = np.zeros(len(bins_edges) - 1)
x[:] = bins_edges[0:len(x)] + delta_bins / 2.
Nevt_N, Nevt_I = spectrum.resol_spectrum(x, a, b, 0, plot_this=False)

appo_N = 0
appo_I = 0
for n_ in np.arange(0, len(x)):
    appo_N = appo_N + Nevt_N[n_] * delta_bins
    appo_I = appo_I + Nevt_I[n_] * delta_bins

Nevt_N = Nevt_N / appo_N * events * delta_bins
Nevt_N = np.around(Nevt_N)

Nevt_I = Nevt_I / appo_I * events * delta_bins
Nevt_I = np.around(Nevt_I)

fig_histo = plt.figure(figsize=[12., 7.5])
ax_histo = fig_histo.add_subplot(111)
ax_histo.hist(x - 0.8, bins_edges - 0.8, weights=Nevt_N, color='b', histtype='step', label=r'NO')
# ax_histo.hist(x-0.8,bins_edges-0.8,weights=Nevt_I,color='r',histtype='step',label=r'IO')
ax_histo.set_xlabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
ax_histo.set_ylabel(r'\# of IBD per %.f \si{\keV}' % (delta_bins * 1000))
ax_histo.set_title(r'Antineutrino spectra (100k IBD events)')
ax_histo.legend()
ax_histo.grid() '''

elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
