import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from convolution import Convolution
import latex

import os
import sys

cwd = os.getcwd()
sys.path.insert(0, '/Users/beatricejelmini/Desktop/JUNO/JUNO_codes/mywork/Chi_squared_analysis/classes')

from spectrum import OscillatedSpectrum

delta_bins = 0.041  # [Mev]
bin_edges = np.arange(1.80, 10.001, delta_bins)
x = np.zeros(len(bin_edges) - 1)
x[:] = bin_edges[0:len(x)] + delta_bins / 2.

conv = Convolution()
# E_nu = np.arange(1.806, 30.01, 0.01)
# E_dep = E_nu - 0.8
a = 0.03
b = 0.

# T = 100000
T = 365 * 0.825 * 6
# T = 2000
# T = 1667
# N_IBD = 108405  # signal only, = 301 day/y * 6 y * 60 evt/day
# N_acc = N_IBD * 0.9 / 60
# N_Fn = N_IBD * 0.1 / 60
# N_alpha = N_IBD * 0.05 / 60
# N_Li = N_IBD * 1.6 / 60
# N_geo = N_IBD * 1.1 / 60
N_IBD = 60 * T
N_acc = 0.9 * T
N_Fn = 0.1 * T
N_alpha = 0.05 * T
N_Li = 1.6 * T
N_geo = 1.1 * T

# spectrum = OscillatedSpectrum(t12=0.307, m21=7.53e-5, t13_N=0.0218, m3l_N=2.444e-3+7.53e-5, t13_I=0.0218,
#                               m3l_I=-2.55e-3)  # Jinnan 2018
# spectrum = OscillatedSpectrum(t12=0.307, m21=7.54e-5, t13_N=0.0241, m3l_N=2.43e-3 + 0.5 * 7.54e-5, t13_I=0.0244,
#                               m3l_I=-0.5 * 7.54e-5 - 2.42e-3)  # 2012
spectrum = OscillatedSpectrum(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3, t13_I=0.02263,
                              m3l_I=-2.512e-3)  # 2019
baselines_12 = np.array([52.75, 52.84, 52.42, 52.51, 52.12, 52.21, 52.76, 52.63, 52.32, 52.20, 215., 265.])  # [km]
power_GW_12 = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 4.6, 4.6, 17.4, 17.4])
spectrum.norm_bool = True
spectrum.set_L_P_distribution(baselines_12, power_GW_12)
# IBD = spectrum.eval_sum_NO(x, 0.307, 7.54e-5, 0.0241, 2.43e-3+0.5*7.54e-5)  # NO, 12 cores, normalized to unit area, no smearing - 2012
# IBD = spectrum.eval_sum_NO(x, 0.307, 7.53e-5, 0.0218, 2.444e-3+7.53e-5)  # NO, 12 cores, normalized to unit area, no smearing - 2018
IBD = spectrum.eval_sum_NO(x, 0.310, 7.39e-5, 0.02240, 2.525e-3)  # NO, 12 cores, normalized to unit area, no smearing - 2019
IBD_hist = IBD * N_IBD * delta_bins

Acc = np.loadtxt('acc_bkg.txt')
norm = integrate.simps(Acc, x)
Acc = Acc / norm
Acc_hist = Acc * N_acc * delta_bins
appo = (Acc_hist.sum()/N_acc-1.) * 100
if appo > 0.1:
    print('Acc: appo greater than 0.1')
    correction = N_acc/Acc_hist.sum()
    Acc_hist = Acc * N_acc * delta_bins * correction

Fn = np.loadtxt('fn_bkg.txt')
norm = integrate.simps(Fn, x)
Fn = Fn / norm
Fn_hist = Fn * N_Fn * delta_bins
appo = (Fn_hist.sum()/N_Fn-1.) * 100
if appo > 0.1:
    print('Fn: appo greater than 0.1')
    correction = N_Fn/Fn_hist.sum()
    Fn_hist = Fn * N_Fn * delta_bins * correction

alpha = np.loadtxt('alpha_bkg.txt')
norm = integrate.simps(alpha, x)
alpha = alpha / norm
alpha_hist = alpha * N_alpha * delta_bins
appo = (alpha_hist.sum()/N_alpha-1.) * 100
if appo > 0.1:
    print('alpha: appo greater than 0.1')
    correction = N_alpha/alpha_hist.sum()
    alpha_hist = alpha * N_alpha * delta_bins * correction

Li = np.loadtxt('Li9_bkg.txt')
norm = integrate.simps(Li, x)
Li = Li / norm
Li_hist = Li * N_Li * delta_bins
appo = (Li_hist.sum()/N_Li-1.) * 100
if appo > 0.1:
    print('Li: appo greater than 0.1')
    correction = N_Li/Li_hist.sum()
    Li_hist = Li * N_Li * delta_bins * correction

Geo = np.loadtxt('geo_nu.txt')
norm = integrate.simps(Geo, x)
Geo = Geo / norm
Geo_hist = Geo * N_geo * delta_bins
appo = (Geo_hist.sum()/N_geo-1.) * 100
if appo > 0.1:
    print('Geo: appo greater than 0.1')
    correction = N_geo/Geo_hist.sum()
    Geo_hist = Geo * N_geo * delta_bins * correction

# numerical convolution of the signal spectrum
# Note: the backgrounds' spectra are already smeared
IBD_resolution = conv.numerical_conv(IBD_hist, x-0.8, x-0.8, a=a, b=b)
Sum_resolution = IBD_resolution + Acc_hist + Fn_hist + alpha_hist + Li_hist + Geo_hist

# numerical convolution of the signal spectrum normalized to unit area
IBD_res = conv.numerical_conv(IBD, x-0.8, x-0.8, a=a, b=b)
sum_appo = (IBD_res * N_IBD + Acc * N_acc + Fn * N_Fn + alpha * N_alpha + Li * N_Li + Geo * N_geo) * delta_bins

# background spectra, normalized to unit area, with experimental smearing
backgrounds = np.array([Acc, Fn, alpha, Li, Geo])
np.save('backgrounds_1area.npy', backgrounds)
backgrounds_norm = np.array([Acc_hist, Fn_hist, alpha_hist, Li_hist, Geo_hist])
np.save('backgrounds_'+str(round(N_IBD))+'IBD.npy', backgrounds_norm)
# np.save('backgrounds_'+str(round(T))+'days.npy', backgrounds_norm)

loc = plticker.MultipleLocator(base=2.0)  # this locator puts ticks at regular intervals
loc1 = plticker.MultipleLocator(base=0.5)

fig_IBD = plt.figure(figsize=[10.5, 6.5])
ax_IBD = fig_IBD.add_subplot(111)
fig_IBD.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.10)
# ax_IBD.hist(x, bin_edges, weights=IBD, color='k', histtype='step', label='signal', linewidth=1.)
ax_IBD.hist(x-0.8, bin_edges-0.8, weights=Acc, color='b', histtype='step', label=r'Accidental', linewidth=1.)
ax_IBD.hist(x-0.8, bin_edges-0.8, weights=Fn, color='g', histtype='step', label=r'Fast neutron', linewidth=1.)
ax_IBD.hist(x-0.8, bin_edges-0.8, weights=alpha, color='c', histtype='step', label=r'$^{13}\text{C} (\alpha,n) ^{16}\text{O}$', linewidth=1.)
ax_IBD.hist(x-0.8, bin_edges-0.8, weights=Li, color='r', histtype='step', label=r'$^9\text{Li}/^8\text{He}$', linewidth=1.)
ax_IBD.hist(x-0.8, bin_edges-0.8, weights=Geo, color='m', histtype='step', label=r'Geo-neutrino', linewidth=1.)
ax_IBD.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
ax_IBD.set_xlim(0.5, 9.5)
ax_IBD.set_ylabel(r'\# of events')
# ax_IBD.set_yscale('log')
# ax_IBD.set_title(r'Background spectra - normalized to unit area')
ax_IBD.legend()
ax_IBD.grid(alpha=0.45)
ax_IBD.xaxis.set_major_locator(loc)
ax_IBD.xaxis.set_minor_locator(loc1)
ax_IBD.tick_params('both', direction='out', which='both')
# fig_IBD.savefig('bkg_unitarea.pdf', format='pdf')

fig_res = plt.figure(figsize=[10.5, 6.5])
ax_res = fig_res.add_subplot(111)
fig_res.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.10)
ax_res.hist(x-0.8, bin_edges-0.8, weights=Sum_resolution, color='k', histtype='step', label=r'Signal + backgrounds', linewidth=1., zorder=10)
ax_res.hist(x-0.8, bin_edges-0.8, weights=IBD_resolution, color='k', histtype='step', label=r'Signal', linewidth=1., linestyle=':', zorder=1)
ax_res.hist(x-0.8, bin_edges-0.8, weights=Acc_hist, color='b', histtype='step', label=r'Accidental', linewidth=1.)
ax_res.hist(x-0.8, bin_edges-0.8, weights=Fn_hist, color='g', histtype='step', label=r'Fast neutron', linewidth=1.)
ax_res.hist(x-0.8, bin_edges-0.8, weights=alpha_hist, color='c', histtype='step', label=r'$^{13}\text{C} (\alpha,n) ^{16}\text{O}$', linewidth=1.)
ax_res.hist(x-0.8, bin_edges-0.8, weights=Li_hist, color='r', histtype='step', label=r'$^9\text{Li}/^8\text{He}$', linewidth=1.)
ax_res.hist(x-0.8, bin_edges-0.8, weights=Geo_hist, color='m', histtype='step', label=r'Geo-neutrino', linewidth=1.)
ax_res.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
ax_res.set_xlim(0.5, 9.5)
ax_res.set_ylabel(r'\# of events per %.f \si{keV}' % (delta_bins * 1000))
ax_res.set_ylim(0, 1200)
# ax_res.set_yscale('log')
# ax_res.set_ylim(0.1, 3500)
# ax_res.set_title('Antineutrino and background spectra - normalized to %i days' % round(T) + '\n(with experimental smearing)')
ax_res.legend()
ax_res.grid(alpha=0.45)
ax_res.xaxis.set_major_locator(loc)
ax_res.xaxis.set_minor_locator(loc1)
ax_res.tick_params('both', direction='out', which='both')
# fig_res.savefig('signal_bkg_smeared.pdf', format='pdf')

loc2 = plticker.MultipleLocator(base=50)
fig_2 = plt.figure(figsize=[10.5, 6.5])
ax_2 = fig_2.add_subplot(111)
ax_2.hist(x-0.8, bin_edges-0.8, weights=Sum_resolution, color='k', histtype='step', label='signal + bkg', linewidth=1.)
ax_2.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
ax_2.set_xlim(0.8, 9.4)
ax_2.set_ylabel(r'\# of events per %.f \si{keV}' % (delta_bins * 1000))
ax_2.set_title(r'Final spectrum - normalized to %i days' % round(T) + '\n(with experimental smearing)')
ax_2.legend()
ax_2.grid()
ax_2.xaxis.set_major_locator(loc)
ax_2.xaxis.set_minor_locator(loc1)
ax_2.yaxis.set_minor_locator(loc2)
ax_2.set_ylim(0, 1400)
ax_2.tick_params('both', direction='out', which='both')
# fig_2.savefig('smeared_final_spectrum_12cores.pdf', format='pdf')

plt.ion()
plt.show()
