import numpy as np
import matplotlib.pyplot as plt
import time
import math

from scipy import integrate
import latex
from reactor import ReactorSpectrum
from oscillation import OscillationProbability
from convolution import Convolution

##### MAIN PROGRAM #####

# years = 6
# T = 365. * years * 86400  # [s]
# eff_T = 0.822  # 300 days per year
# eff_T = 0.91 # ~330 days per year
# epsilon_IBD = 0.73  # YB table 3, pag. 28

years = float(input('Enter number of years: '))
T = 365. * years * 86400  # [s]
eff_T = float(input('Enter time efficiency (0.822 for 300 days per year): '))
epsilon_IBD = float(input('Enter IBD detection efficiency (default from YB: 0.73): '))

time_start = time.process_time_ns()

delta_bins = 0.031  # [Mev] -> 200 bins
bin_edges = np.arange(1.80, 8.001, delta_bins)
E = np.zeros(len(bin_edges) - 1)
E[:] = bin_edges[0:len(E)] + delta_bins / 2.
# E = np.arange(1.806, 8.01, 0.031)  # in MeV


N_P = 1.45e33
baselines = np.array([52.75, 52.84, 52.42, 52.51, 52.12, 52.21, 52.76, 52.63, 52.32, 52.20, 215., 265.])  # [km]
power_GW = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 4.6, 4.6, 17.4, 17.4])
react_name = ['YJ-C1', 'YJ-C2', 'YJ-C3', 'YJ-C4', 'YJ-C5', 'YJ-C6', 'TS-C1', 'TS-C2', 'TS-C3', 'TS-C4', 'DYB', 'HZ']
# baselines = np.array([52.75, 52.84, 52.42, 52.51, 52.12, 52.21, 52.76, 52.63, 52.32, 52.20])  # [km]
# power_GW = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 4.6, 4.6])
# react_name = ['YJ-C1', 'YJ-C2', 'YJ-C3', 'YJ-C4', 'YJ-C5', 'YJ-C6', 'TS-C1', 'TS-C2', 'TS-C3', 'TS-C4', 'DYB', 'HZ']
# baselines = np.array([53.,215.,265.])
# power_GW = np.array([36,17.4,17.4])
# react_name = ['1','DYB','HZ']

M = len(baselines)

react = ReactorSpectrum()
### input: sin^2(theta_12), deltam^2_21, NO: sin^2(theta_13), deltam^2_3l, IO: sin^2(theta_13), deltam^2_3l
# prob = Oscillation_Prob(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3, t13_I=0.02263, m3l_I=-2.512e-3) # 2019
prob = OscillationProbability(t12=0.307, m21=7.54e-5, t13_N=0.024, m3l_N=2.505e-3, t13_I=0.024,
                              m3l_I=-7.54e-5 - 2.42e-3)  # 2012

xsec = react.cross_section(E, plot_this=False)

react_n = np.zeros(M)
tot_events = 0
spectrum = np.zeros(len(E))

for n_ in np.arange(M):
    flux = react.flux(E, power_GW[n_], plot_this=False)

    prob.baseline = baselines[n_]
    prob_N, prob_I = prob.eval_prob(E, 1, plot_this_LE=False, plot_this_E=False)

    integrand = flux * xsec * prob_N
    integral = integrate.simps(integrand, E)

    appo = N_P * epsilon_IBD / (4. * math.pi * (baselines[n_] * 1.e5) ** 2)
    react_n[n_] = appo * integral * T * eff_T

    spectrum = spectrum + integrand * appo * T * eff_T

    tot_events = tot_events + react_n[n_]

elapsed_time = time.process_time_ns() - time_start
print('\nelapsed time: ' + str(elapsed_time * 1.e-6) + ' ms')

print('\nTotal number of IBD events in JUNO: %.i' % tot_events)
print('Effective running time: %.1f days over %.i years' % (T / 86400. * eff_T, years))
print('Time efficiency: %.3f ' % eff_T)
print('IBD detection efficiency: %.3f ' % epsilon_IBD)

print('\nContribution of each single reactor:')
print('Core  -  IBD events')
for r_ in np.arange(M):
    print(react_name[r_] + ' ' + str(react_n[r_]))

a = 0.03
b = 0.

conv = Convolution()
spectrum_resol = conv.numerical_conv_NEW(spectrum, E-0.8, E-0.8, a=a, b=b)

year = 2014
fig_histo = plt.figure(figsize=[10., 6.5])
ax_histo = fig_histo.add_subplot(111)
ax_histo.hist(E-0.8, bin_edges-0.8, weights=spectrum_resol*delta_bins,
              color='k', histtype='step', label=r'data (N)', linewidth=1.)
ax_histo.set_xlabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
ax_histo.set_ylabel(r'\# of IBD per %.f \si{\keV}' % (delta_bins * 1000))
ax_histo.set_title(r'Antineutrino spectrum (100k IBD events) - True MO: Normal - %i' % year)
ax_histo.legend()
ax_histo.grid()

plt.ion()
plt.show()