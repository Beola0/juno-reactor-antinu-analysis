import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as plticker
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import json
import math
import pandas as pd
# from scipy.interpolate import interp1d, Akima1DInterpolator, UnivariateSpline
from scipy import integrate
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')
import latex
from plot import plot_function, plot_function_residual
from reactor import UnoscillatedReactorSpectrum
from oscillation import OscillationProbability


########################################################################################################################
# useful stuff
########################################################################################################################
U5 = r'$^{235}$U'
U8 = r'$^{238}$U'
Pu9 = r'$^{239}$Pu'
Pu1 = r'$^{241}$Pu'

# f = open('Inputs/nufit_inputs.json')
f = open('Inputs/nominal_inputs.json')
inputs_json = json.load(f)

std_hm = {
    '235U': 'Huber',
    '238U': 'Mueller',
    '239Pu': 'Huber',
    '241Pu': 'Huber'
}

dyb_input = {
    'total': 'DYB',
    '235U': 'DYB',
    '238U': 'Mueller',
    '239Pu': 'DYB_combo',
    '241Pu': 'Huber'
}

e_235 = inputs_json["mean_fission_energy"]["235U"]
e_239 = inputs_json["mean_fission_energy"]["239Pu"]
e_238 = inputs_json["mean_fission_energy"]["238U"]
e_241 = inputs_json["mean_fission_energy"]["241Pu"]

react = UnoscillatedReactorSpectrum(inputs_json)
time_start = time.perf_counter_ns()


########################################################################################################################
# cross section per fission
########################################################################################################################

E_f = np.arange(1.875, 8.13, 0.005)
react.verbose = False
cross_section = react.eval_xs(E_f, which_xs="SV_CI", bool_protons=False)
xsf_235 = integrate.simps(react.eval_235u(E_f, which_input="Huber")*cross_section, E_f)
xsf_239 = integrate.simps(react.eval_239pu(E_f, which_input="Huber")*cross_section, E_f)
xsf_238 = integrate.simps(react.eval_238u(E_f, which_input="Mueller")*cross_section, E_f)
xsf_241 = integrate.simps(react.eval_241pu(E_f, which_input="Huber")*cross_section, E_f)
print(f"\nCross section per fission for each isotope; relative difference wrt prediction (HM)")
print("235U: {:.2e} cm2/fission; {:.2f}%".format(xsf_235, (xsf_235-6.69e-43)/6.69e-43*100))
print("239Pu: {:.2e} cm2/fission; {:.2f}%".format(xsf_239, (xsf_239-4.4e-43)/4.4e-43*100))
print("238U: {:.2e} cm2/fission; {:.2f}%".format(xsf_238, (xsf_238-10.1e-43)/10.1e-43*100))
print("241Pu: {:.2e} cm2/fission; {:.2f}%".format(xsf_241, (xsf_241-6.03e-43)/6.03e-43*100))

react.set_fission_fractions(0.58, 0.3, 0.07, 0.05)
flux_juno = react.reactor_model_dyb(E_f, dyb_input)
xsf_juno = integrate.simps(flux_juno*cross_section, E_f)
print("Total cross section per fission: {:.2e} cm2/fission - DYB model".format(xsf_juno))

flux_juno_hm = react.reactor_model_std(E_f, std_hm)
xsf_juno_hm = integrate.simps(flux_juno_hm*cross_section, E_f)
print("Total cross section per fission: {:.2e} cm2/fission - HM model".format(xsf_juno_hm))

react.set_fission_fractions(0.6033, 0.2744, 0.0757, 0.0466)
flux_a = react.reactor_model_dyb(E_f, dyb_input)
xsf_a = integrate.simps(flux_a*cross_section, E_f)

react.set_fission_fractions(0.5279, 0.3326, 0.0766, 0.0629)
flux_b = react.reactor_model_dyb(E_f, dyb_input)
xsf_b = integrate.simps(flux_b*cross_section, E_f)
print("\nfission fraction 239Pu: 0.2744 - total cross section per fission: {:.2e} cm2/fission".format(xsf_a))
print("fission fraction 239Pu: 0.3326 - total cross section per fission: {:.2e} cm2/fission".format(xsf_b))

########################################################################################################################
# time evolution of mean cross section per fission
########################################################################################################################

E_t = np.arange(1.875, 9.005, 0.005)
E_t2 = np.arange(1.875, 8.13, 0.005)
mean_fiss_f = pd.read_csv("Inputs/effective_fission_fractions_DYB_PRL2017.txt", sep=',', skiprows=16, header=None,
                                  names=["239_lo", "239_hi", "f_239", "f_235", "f_238", "f_241", "sig"])
mean_fiss_f = mean_fiss_f.loc[::-1].set_index(mean_fiss_f.index)  # reverse df

xs = react.eval_xs(E_t, which_xs="SV_CI", bool_protons=False)
xs2 = react.eval_xs(E_t2, which_xs="SV_CI", bool_protons=False)
xsf = np.zeros(len(mean_fiss_f.index))
xsf2 = np.zeros(len(mean_fiss_f.index))
xsf_hm = np.zeros(len(mean_fiss_f.index))
spectra = []
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=[10, 6.5], sharex=True, sharey=True, constrained_layout=True)
for i_, ax in enumerate(axes.flat):
    react.set_fission_fractions(mean_fiss_f['f_235'].iloc[i_], mean_fiss_f['f_239'].iloc[i_],
                                mean_fiss_f['f_238'].iloc[i_], mean_fiss_f['f_241'].iloc[i_])
    flux = react.reactor_model_dyb(E_t, dyb_input)
    ax.plot(E_t, flux*xs, 'k-', label=r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[i_], linewidth=1.,)
    ax.legend()
    ax.grid(alpha=0.65)
    spectra.append(flux*xs)
    xsf[i_] = integrate.simps(flux*xs, E_t)
    xsf2[i_] = integrate.simps(react.reactor_model_dyb(E_t2, dyb_input) * xs2, E_t2)
    xsf_hm[i_] = integrate.simps(react.reactor_model_std(E_t, std_hm) * xs, E_t)

plot_function(
    x_=[E_t, E_t, E_t, E_t, E_t, E_t, E_t, E_t],
    y_=spectra,
    label_=[r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[0], r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[1],
            r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[2], r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[3],
            r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[4], r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[5],
            r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[6], r'F$_{239}$ = %.4f' % mean_fiss_f['f_239'].iloc[7]],
    ylabel_=r'spectrum [a.u.]'
)

tot = mean_fiss_f['f_235'] + mean_fiss_f['f_239'] + mean_fiss_f['f_238'] + mean_fiss_f['f_241']
plot_function(
    y_=[mean_fiss_f['f_235'], mean_fiss_f['f_239'], mean_fiss_f['f_238'], mean_fiss_f['f_241']],
    x_=[mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239']],
    label_=[r'F$_{235}$', 'F$_{239}$', 'F$_{238}$', 'F$_{241}$'], base_major=0.02, base_minor=0.01,
    styles=['b^', 'rs', 'gp', 'mH'], xlabel_=r'$F_{239}$', ylabel_=r'$F_{i}$', ylim=[0, 1]
)

yl = r'$ \langle \sigma \rangle_f$ [\si{\centi\meter\squared}/fission]'
plot_function(
    x_=[mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239']],
    y_=[xsf, xsf2, xsf_hm, mean_fiss_f['sig']*1.e-43],
    xlabel_=r'$F_{239}$', ylabel_=yl, base_major=0.02, base_minor=0.01,
    styles=['ko', 'bo', 'go', 'ro'],
    label_=[r'DYB 1.875-9 MeV', r'DYB 1.875-8.125 MeV', r'HM 1.875-9 MeV', r'DYB PRL 2017']
)
print("\nMean cross section per fission decreases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(xsf[0], xsf[-1],
                                                                   (xsf[-1]-xsf[0])/xsf[0]*100))


########################################################################################################################
# time evolution of mean energy per fission, fission rate, and neutrino detection rate
########################################################################################################################

mean_e = np.zeros(len(mean_fiss_f.index))
for i_ in mean_fiss_f.index:
    mean_e[i_] = e_235*mean_fiss_f['f_235'].iloc[i_] + e_239*mean_fiss_f['f_239'].iloc[i_] \
                 + e_238*mean_fiss_f['f_238'].iloc[i_] + e_241*mean_fiss_f['f_241'].iloc[i_]

yl = r'$\langle E \rangle_f$ [\si{\MeV}/fission]'
ax = plot_function(
    x_=[mean_fiss_f['f_239']], y_=[mean_e], xlabel_=r'$F_{239}$', ylabel_=yl,
    styles=['ko'], label_=['mean e'], base_major=0.02, base_minor=0.01
)
ax.get_legend().remove()
print("\nMean energy per fission increases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(mean_e[0], mean_e[-1],
                                                                   (mean_e[-1]-mean_e[0])/mean_e[0]*100))

const = 6.241509e21
w_th = 4.6  # GW - Taishan core
L = 52.5e5  # TAO baseline in cm
Np = react.eval_n_protons()
eff = 0.822
fission_rate = w_th/mean_e * const

yl = r'fission rate [fission/s]'
ax = plot_function(
    x_=[mean_fiss_f['f_239']], y_=[fission_rate], xlabel_=r'$F_{239}$', ylabel_=yl,
    styles=['ko'], label_=['fission rate'], base_major=0.02, base_minor=0.01
)
ax.get_legend().remove()
ax.text(0.3, 1.397e20, r'Taishan: W$_{\text{th}}$ = 4.6 \si{\giga\watt}')
print("\nFission rate decreases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(fission_rate[0], fission_rate[-1],
                                                                   (fission_rate[-1]-fission_rate[0])/fission_rate[0]*100))

yl = r'antineutrino rate [$\bar{\nu}$/\si{s}]'
nu_rate = fission_rate*xsf*Np*eff/(4*math.pi*L*L)
ax = plot_function(
    x_=[mean_fiss_f['f_239']], y_=[nu_rate], xlabel_=r'$F_{239}$', ylabel_=yl,
    styles=['ko'], label_=['nu rate'], base_major=0.02, base_minor=0.01
)
ax.get_legend().remove()
ax.text(0.3, 2.84e-4, r'Taishan: W$_{\text{th}}$ = %f \si{\giga\watt}' % w_th, fontsize=12)
ax.text(0.3, 2.835e-4, r'Baseline: L = %.1f \si{\km}' % (L*1.e-5), fontsize=12)
ax.text(0.3, 2.83e-4, r'Number of protons: N = %.2e' % Np, fontsize=12)
ax.text(0.3, 2.825e-4, r'IBD efficiency: $\epsilon$ = %.3f' % eff, fontsize=12)
ax.text(0.3, 2.82e-4, r'NO neutrino oscillations', fontsize=12)
print("\nAntineutrino rate decreases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(nu_rate[0], nu_rate[-1],
                                                                   (nu_rate[-1]-nu_rate[0])/nu_rate[0]*100))


########################################################################################################################
# time variation per energy bin
########################################################################################################################

# E_1 = np.arange(1.875, 3.005, 0.005)
E = np.arange(1.875, 9.001, 0.005)
bin_edges = [1.875, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # 6 bins
bin_centers = [2.4375, 3.5, 4.5, 5.5, 6.5, 7.5]
errorx = [0.5625, 0.5, 0.5, 0.5, 0.5, 0.5]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[10, 6.5], sharex=True, sharey=True, constrained_layout=True)
ff_239 = np.arange(0.24, 0.365, 0.005)
coef_array = []
coef_array_hm = []
for j_, ax in enumerate(axes.flat):  # loop over energy bins
    xs_appo = np.where(E <= bin_edges[j_ + 1],
                       np.where(bin_edges[j_] <= E, react.eval_xs(E, which_xs="SV_CI", bool_protons=False), 0.), 0.)
    xsf_appo = np.zeros((len(mean_fiss_f.index)))
    xsf_appo_hm = np.zeros((len(mean_fiss_f.index)))
    for i_ in mean_fiss_f.index:  # loop over 239Pu fission fractions
        react.set_fission_fractions(mean_fiss_f['f_235'].iloc[i_], mean_fiss_f['f_239'].iloc[i_],
                                    mean_fiss_f['f_238'].iloc[i_], mean_fiss_f['f_241'].iloc[i_])
        flux_appo = np.where(E <= bin_edges[j_ + 1],
                             np.where(bin_edges[j_] <= E, react.reactor_model_dyb(E, dyb_input), 0.), 0.)
        flux_appo_hm = np.where(E <= bin_edges[j_ + 1],
                                np.where(bin_edges[j_] <= E, react.reactor_model_std(E, std_hm), 0.), 0.)
        xsf_appo[i_] = integrate.simps(flux_appo*xs_appo, E)
        xsf_appo_hm[i_] = integrate.simps(flux_appo_hm*xs_appo, E)
    ax.plot(mean_fiss_f['f_239'], xsf_appo / xsf_appo.mean(), 'bo', label='model data (DYB)')
    ax.plot(mean_fiss_f['f_239'], xsf_appo_hm / xsf_appo_hm.mean(), 'ro', label='model data (HM)')
    coef = np.polyfit(mean_fiss_f['f_239'], xsf_appo/xsf_appo.mean(), 1)
    coef_hm = np.polyfit(mean_fiss_f['f_239'], xsf_appo_hm / xsf_appo_hm.mean(), 1)
    coef_array.append(coef[0])
    coef_array_hm.append(coef_hm[0])
    fit_line = np.poly1d(coef)
    fit_line_hm = np.poly1d(coef_hm)
    ax.plot(ff_239, fit_line(ff_239), 'b--', label='best fit (DYB)')
    ax.plot(ff_239, fit_line_hm(ff_239), 'r--', label='best fit (HM)')
    ax.set_ylim([0.95, 1.05])
    ax.set_xlim([0.24, 0.36])
    if j_ == 0 or j_ == 3:
        ax.set_ylabel(r'$S_j$/$\bar{S}_j$')
    if j_ == 3 or j_ == 4 or j_ == 5:
        ax.set_xlabel(r'$F_{239}$')
    ax.grid(alpha=0.65)
    e_range = r"E$_{\nu}$ = %.2f-%.2f MeV" % (bin_edges[j_], bin_edges[j_+1])
    slope = r'$\bar{S}_j^{-1} d S_j/d F_{239}$ = %.2f' % coef[0]
    ax.text(0.25, 0.97, e_range, fontsize=12)
    ax.text(0.25, 0.96, slope, fontsize=12)
ax.legend(loc='upper right')

plt.figure(figsize=[7, 4], constrained_layout=True)
plt.plot(bin_centers, coef_array, 'bo', label='DYB model')
plt.errorbar(bin_centers, coef_array, xerr=errorx, fmt='none', ecolor='b')
plt.plot(bin_centers, coef_array_hm, 'r-', label='HM model')
# plt.errorbar(bin_centers, coef_array_hm, xerr=errorx, fmt='none', ecolor='r')
plt.grid()
plt.xlabel(r'Neutrino energy [\si{\MeV}]')
plt.ylabel(r'$\bar{S}_j^{-1} d S_j/d F_{239}$')
plt.ylim([-0.9, 0])
plt.xlim([0, 10])
plt.legend()


osc = OscillationProbability(inputs_json)
osc_no = osc.eval_vacuum_prob_no(E)
flux = react.reactor_model_dyb(E, dyb_input) * w_th * const / mean_e.mean() * eff / (4*math.pi*L*L)
xs = react.eval_xs(E, which_xs="SV_CI", bool_protons=True)
norm = integrate.simps(flux*xs, E) * 86400
norm2 = integrate.simps(flux*osc_no*xs, E) * 86400
print("\nNumber of IBD/day without neutrino oscillation: {:.2f}".format(norm))
print("\nNumber of IBD/day with neutrino oscillation: {:.2f}".format(norm2))

plt.figure()
plt.plot(E, flux*xs)
plt.plot(E, flux*osc_no*xs)


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
