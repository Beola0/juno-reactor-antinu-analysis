import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import time
import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')

import latex
from spectrum import OscillatedSpectrum

## MAIN PROGRAM ##

time_start = time.process_time_ns()

E = np.arange(1.806, 10.01, 0.005)  # in MeV

# input: sin2(theta12), deltam2_21, NO: sin2(theta13), deltam2_3l, IO: sin2(theta13), deltam2_3l
spectrum = OscillatedSpectrum(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3, t13_I=0.02263,
                              m3l_I=-2.512e-3)

a = 0.029
b = 0.008

resol_spect_N, resol_spect_I = spectrum.resol_spectrum(E-0.8, a, b, 0, plot_this=False, normalize=True)

loc = plticker.MultipleLocator(base=2.0)
loc1 = plticker.MultipleLocator(base=0.5)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(E - 0.8, resol_spect_N, 'b', linewidth=1.5, label='NO')
ax.plot(E - 0.8, resol_spect_I, 'r--', linewidth=1.5, label='IO')
ax.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
ax.set_xlim(0.5, 9.5)
ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
ax.set_ylim(-0.005, 0.305)
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_minor_locator(loc1)
ax.tick_params('both', direction='out', which='both')
ax.set_title(r'Antineutrino spectrum with finite energy resolution')
ax.text(8.05, 0.05, r'a = \SI{%.1f}{\percent}' % (a * 100) + '\nb = \SI{%.1f}{\percent}' % (b * 100))
ax.legend()
ax.grid(alpha=0.45)

### spectrum with different baselines

baselines = np.array([52.74, 52.82, 52.41, 52.49, 52.11, 52.19, 52.77, 52.64, 215, 265])  # [km]
power_GW = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 17.4, 17.4])
react_name = ['YJ-C1', 'YJ-C2', 'YJ-C3', 'YJ-C4', 'YJ-C5', 'YJ-C6', 'TS-C1', 'TS-C2', 'DYB', 'HZ']
A = (power_GW / baselines / baselines).sum()
weights = power_GW / baselines / baselines / A

a = 0.029
b = 0.008

# sum_N, sum_I = spectrum.sum(baselines, power_GW, E, plot_sum=False, plot_baselines=False)
# sum_res_N, sum_res_I = spectrum.sum_resol(baselines, power_GW, E-0.8, a, b, normalize=True,
#                                           plot_sum=False, plot_baselines=False)

sum_ = np.zeros(len(E))

fig_b = plt.figure(figsize=[10.5, 6.5])
ax_b = fig_b.add_subplot(111)
fig_b.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.10)
for n_ in np.arange(0, len(baselines)):
    spectrum.baseline = baselines[n_]
    # spectrum.osc_spectrum(E, 0, normalize=True)
    # ax_b.plot(E, spectrum.norm_osc_spect_N, linewidth=1., label=r'L = %.2f \si{\km} - %s' % (baselines[n_], react_name[n_]))
    spectrum.resol_spectrum(E-0.8, a, b, 0, normalize=True)
    # ax_b.plot(E-0.8, spectrum.resol_N, linewidth=1., label=r'L = %.2f \si{\km} - %s' % (baselines[n_], react_name[n_]))
    ax_b.plot(E-0.8, spectrum.resol_N * weights[n_], linewidth=1., label=r'L = %.2f \si{\km} - %s' % (baselines[n_], react_name[n_]))
    sum_ = sum_ + spectrum.resol_N * weights[n_]
ax_b.plot(E-0.8, sum_, 'k', label='total spectrum')
# ax_b.text(2.6-0.8, 0.47, 'DYB', color='tab:blue')
# ax_b.text(4.15-0.8, 0.53, 'HZ', color='tab:orange')
ax_b.text(4.55, 0.00821, 'DYB', color='k')  # tab:olive
ax_b.text(3.46, 0.0115, 'HZ', color='k')  # tab:cyan
ax_b.text(4.13, 0.0449, 'TS', color='k')
ax_b.text(4.02, 0.03, 'YJ', color='k')
ax_b.legend()
ax_b.grid(alpha=0.45)
# ax_b.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
ax_b.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
# ax_b.set_xlim(1.5, 10.5)
ax_b.set_xlim(0.5, 9.5)
ax_b.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
# ax_b.set_ylim(-0.015, 0.32)
#ax_b.set_ylim(-0.01, 0.61)
ax_b.set_ylim(-0.005, 0.278)
#ax_b.set_ylim(-0.001, 0.05)
ax_b.xaxis.set_major_locator(loc)
ax_b.xaxis.set_minor_locator(loc1)
ax_b.tick_params('both', direction='out', which='both')
# ax_b.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
fig_b.savefig('SpectrumPlots/weighted_baseline_contributions_sum.pdf', format='pdf', transparent=True)
#print('\nThe plot has been saved in SpectrumPlots/baselines_norm.pdf')


elapsed_time = time.process_time_ns() - time_start
print('elapsed time: ' + str(elapsed_time * 1.e-6) + ' ms')

plt.ion()
plt.show()

