import numpy as np
import matplotlib.pyplot as plt
import time
import json
import math
import pandas as pd
from scipy import integrate, stats, interpolate
import latex
from antinu_spectrum.plot import plot_function, plot_function_residual
from antinu_spectrum.reactor import UnoscillatedReactorSpectrum
from antinu_spectrum.oscillation import OscillationProbability

HEADER = '\033[95m'
BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
NC = '\033[0m'

########################################################################################################################
# useful stuff
########################################################################################################################
U5 = r'$^{235}$U'
U8 = r'$^{238}$U'
Pu9 = r'$^{239}$Pu'
Pu1 = r'$^{241}$Pu'

# f = open('data/nufit_inputs.json')
f = open('data/nominal_inputs.json')
inputs_json = json.load(f)

std_hm = {
    '235U': 'Huber',
    '238U': 'Mueller',
    '239Pu': 'Huber',
    '241Pu': 'Huber'
}

input_dyb = {
    'total': 'DYB',
    '235U': 'DYB',
    '238U': 'EF',
    '239Pu': 'DYB_combo',
    '241Pu': 'Huber'
}

path_ff = '/Users/beatricejelmini/Desktop/JUNO/JUNO_codes/JUNO_ReactorNeutrinosAnalysis/data/fission_fractions/'

e_235 = inputs_json["mean_fission_energy"]["235U"]
e_239 = inputs_json["mean_fission_energy"]["239Pu"]
e_238 = inputs_json["mean_fission_energy"]["238U"]
e_241 = inputs_json["mean_fission_energy"]["241Pu"]

react = UnoscillatedReactorSpectrum(inputs_json)
time_start = time.perf_counter_ns()

react.verbose = False


########################################################################################################################
# cross section per fission
########################################################################################################################
print(f"\n{GREEN}Evaluation of IBD yields{NC}")
# E_f = np.arange(1.875, 8.13, 0.005)
E_f = np.arange(1.875, 10.001, 0.005)
xsf_5_ref = 6.69e-43
xsf_8_ref = 10.10e-43
xsf_9_ref = 4.40e-43
xsf_1_ref = 6.03e-43
cross_section_f = react.eval_xs(E_f, which_xs="SV_approx", bool_protons=False)
xsf_235 = integrate.simps(react.eval_235u(E_f, which_input="Huber")*cross_section_f, E_f)
xsf_239 = integrate.simps(react.eval_239pu(E_f, which_input="Huber")*cross_section_f, E_f)
xsf_238 = integrate.simps(react.eval_238u(E_f, which_input="Mueller")*cross_section_f, E_f)
xsf_241 = integrate.simps(react.eval_241pu(E_f, which_input="Huber")*cross_section_f, E_f)
print(f"\nCross section per fission for each isotope; relative difference wrt prediction (HM)")
print("235U: {:.2e} cm2/fission; {:.2f}%".format(xsf_235, (xsf_235-xsf_5_ref)/xsf_5_ref*100))
print("239Pu: {:.2e} cm2/fission; {:.2f}%".format(xsf_239, (xsf_239-xsf_9_ref)/xsf_9_ref*100))
print("238U: {:.2e} cm2/fission; {:.2f}%".format(xsf_238, (xsf_238-xsf_8_ref)/xsf_8_ref*100))
print("241Pu: {:.2e} cm2/fission; {:.2f}%".format(xsf_241, (xsf_241-xsf_1_ref)/xsf_1_ref*100))

react.set_fission_fractions(0.58, 0.3, 0.07, 0.05)
flux_juno = react.reactor_model_dyb(E_f, input_dyb)
xsf_juno = integrate.simps(flux_juno*cross_section_f, E_f)
print("Total cross section per fission: {:.2e} cm2/fission - DYB model".format(xsf_juno))

flux_juno_hm = react.reactor_model_std(E_f, std_hm)
xsf_juno_hm = integrate.simps(flux_juno_hm*cross_section_f, E_f)
print("Total cross section per fission: {:.2e} cm2/fission - HM model".format(xsf_juno_hm))

react.set_fission_fractions(0.6033, 0.2744, 0.0757, 0.0466)
flux_a = react.reactor_model_dyb(E_f, input_dyb)
xsf_a = integrate.simps(flux_a*cross_section_f, E_f)

react.set_fission_fractions(0.5279, 0.3326, 0.0766, 0.0629)
flux_b = react.reactor_model_dyb(E_f, input_dyb)
xsf_b = integrate.simps(flux_b*cross_section_f, E_f)
print("\nBeginning of cycle - DYB model: fission fraction 239Pu: 0.2744 - "
      "total cross section per fission: {:.2e} cm2/fission".format(xsf_a))
print("End of cycle - DYB model: fission fraction 239Pu: 0.3326 - "
      "total cross section per fission: {:.2e} cm2/fission".format(xsf_b))


########################################################################################################################
# fission fractions
########################################################################################################################
plot = False

mean_fiss_f = pd.read_csv(path_ff+"/effective_fission_fractions_DYB_PRL2017.txt", sep=',', skiprows=16, header=None,
                          names=["239_lo", "239_hi", "f_239", "f_235", "f_238", "f_241", "sig"])
mean_fiss_f = mean_fiss_f.loc[::-1].set_index(mean_fiss_f.index)  # reverse df

fiss_f = pd.read_csv(path_ff+"/1cycle_fission_fractions_CPC2017.csv", sep=',', skiprows=1, header=None,
                     names=['burnup', 'days', 'f_235', 'f_239', 'f_238', 'f_241'])
fiss_f['sum'] = fiss_f.iloc[:, 2:6].apply(np.sum, axis=1)
for iso_ in ['f_235', 'f_239', 'f_238', 'f_241']:
    fiss_f[iso_] = fiss_f[iso_] / fiss_f['sum']
    # fiss_f_1cycle[iso_] = fiss_f_1cycle[iso_] / 100.

fiss_f_1cycle = fiss_f[fiss_f.index.isin([0, 10, 20, 35, 53, 75, 105, 132])]
fiss_f_1cycle = fiss_f_1cycle.reset_index(drop=True)

tot = mean_fiss_f['f_235'] + mean_fiss_f['f_239'] + mean_fiss_f['f_238'] + mean_fiss_f['f_241']

if plot:
    plot_function(
        x_=[mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239']],
        y_=[mean_fiss_f['f_235'], mean_fiss_f['f_239'], mean_fiss_f['f_238'], mean_fiss_f['f_241']],
        label_=[r'F$_{235}$', 'F$_{239}$', 'F$_{238}$', 'F$_{241}$'], base_major=0.02, base_minor=0.01,
        styles=['b^', 'rs', 'gp', 'mH'], xlabel_=r'$F_{239}$', ylabel_=r'$F_{i}$', ylim=[0, 1]
    )

    plot_function(
        x_=[fiss_f_1cycle['burnup'], fiss_f_1cycle['burnup'], fiss_f_1cycle['burnup'], fiss_f_1cycle['burnup']],
        y_=[fiss_f_1cycle['f_235'], fiss_f_1cycle['f_239'], fiss_f_1cycle['f_238'], fiss_f_1cycle['f_241']],
        label_=[r'f$_{235}$', r'f$_{239}$', r'f$_{238}$', 'f$_{241}$'], base_major=5000, base_minor=1000,
        styles=['b-', 'r-', 'g-', 'm-'], xlabel_=r'burnup [MWd/tU]', ylabel_=r'$f_{i}$', ylim=[0, 1], xlim=[0, 20500]
    )

    plot_function(
        x_=[fiss_f_1cycle['f_239'], fiss_f_1cycle['f_239'], fiss_f_1cycle['f_239'], fiss_f_1cycle['f_239']],
        y_=[fiss_f_1cycle['f_235'], fiss_f_1cycle['f_239'], fiss_f_1cycle['f_238'], fiss_f_1cycle['f_241']],
        label_=[r'f$_{235}$', r'f$_{239}$', r'f$_{238}$', 'f$_{241}$'], base_major=0.05, base_minor=0.025,
        styles=['b^', 'rs', 'gp', 'mH'], xlabel_=r'$f_{239}$', ylabel_=r'$f_{i}$', ylim=[0, 1],
    )

    plot_function(
        x_=[mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239'], mean_fiss_f['f_239'],
            fiss_f_1cycle['f_239'], fiss_f_1cycle['f_239'], fiss_f_1cycle['f_239'], fiss_f_1cycle['f_239']],
        y_=[mean_fiss_f['f_235'], mean_fiss_f['f_239'], mean_fiss_f['f_238'], mean_fiss_f['f_241'],
            fiss_f_1cycle['f_235'], fiss_f_1cycle['f_239'], fiss_f_1cycle['f_238'], fiss_f_1cycle['f_241']],
        label_=[r'F$_{235}$', 'F$_{239}$', 'F$_{238}$', 'F$_{241}$', r'f$_{235}$', r'f$_{239}$', r'f$_{238}$', 'f$_{241}$'],
        base_major=0.02, base_minor=0.01, xlabel_=r'$f_{239}$', ylabel_=r'$f_{i}$', ylim=[0, 1],  # xlim=[0, 20500],
        styles=['b^', 'rs', 'gp', 'mH', 'b-', 'r-', 'g-', 'm-']
    )


########################################################################################################################
# time evolution of mean cross section per fission
########################################################################################################################
print(f"\n{GREEN}Evaluation of changes in IBD yield, mean energy per fission,\nfission rate, and antinu rate"
      f" with burnup.{NC}")

plot = True

E_t = np.arange(1.875, 9.005, 0.005)
E_t2 = np.arange(1.875, 8.13, 0.005)

xs = react.eval_xs(E_t, which_xs="SV_approx", bool_protons=False)
xs2 = react.eval_xs(E_t2, which_xs="SV_approx", bool_protons=False)

xsf = np.zeros(len(fiss_f_1cycle.index))
xsf2 = np.zeros(len(fiss_f_1cycle.index))
xsf_hm = np.zeros(len(fiss_f_1cycle.index))
# xsf_1cycle = np.zeros(len(fiss_f_1cycle.index))
spectra = []
spectra_hm = []
for i_ in np.arange(len(fiss_f_1cycle)):
    react.set_fission_fractions(fiss_f_1cycle['f_235'].iloc[i_], fiss_f_1cycle['f_239'].iloc[i_],
                                fiss_f_1cycle['f_238'].iloc[i_], fiss_f_1cycle['f_241'].iloc[i_])
    flux = react.reactor_model_dyb(E_t, input_dyb)
    flux_hm = react.reactor_model_std(E_t, std_hm)
    spectra.append(flux*xs)
    spectra_hm.append(flux_hm * xs)
    xsf[i_] = integrate.simps(flux*xs, E_t)
    xsf2[i_] = integrate.simps(react.reactor_model_dyb(E_t2, input_dyb) * xs2, E_t2)
    xsf_hm[i_] = integrate.simps(flux_hm * xs, E_t)

if plot:
    labels = [r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[0], r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[1],
                r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[2], r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[3],
                r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[4], r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[5],
                r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[6], r'f$_{239}$ = %.4f' % fiss_f_1cycle['f_239'].iloc[7]]

    styles = ['navy', 'blue', 'blueviolet', 'magenta', 'hotpink', 'coral', 'red', 'brown']
    ax = plot_function_residual(
        x_=[E_t, E_t, E_t, E_t, E_t, E_t, E_t, E_t], y_=spectra, styles=styles, label_=labels,
        ylabel_=r'spectrum [a.u.]', ylim=[-0.05e-43, 2.05e-43], ylim2=[-20, 0.25]
    )
    ax[0].text(0.1, 0.08, r'DYB model', transform=ax[0].transAxes)
    ax[1].get_legend().remove()

    ax = plot_function_residual(
        x_=[E_t, E_t, E_t, E_t, E_t, E_t, E_t, E_t], y_=spectra_hm, styles=styles, label_=labels,
        ylabel_=r'spectrum [a.u.]', ylim=[-0.05e-43, 2.05e-43], ylim2=[-20, 0.25]
    )
    ax[0].text(0.1, 0.08, r'HM model', transform=ax[0].transAxes)
    ax[1].get_legend().remove()

    yl = r'$ \langle \sigma \rangle_f$ [cm$^2$/fission]'
    plot_function(
        x_=[fiss_f_1cycle['f_239'], fiss_f_1cycle['f_239'], mean_fiss_f['f_239'], fiss_f_1cycle['f_239']],
        y_=[xsf, xsf2, mean_fiss_f['sig']*1.e-43, xsf_hm],
        xlabel_=r'$f_{239}$', ylabel_=yl, base_major=0.05, base_minor=0.025, y_sci=True,
        styles=['bo', 'go', 'ko', 'ro'], ylim=[5.65e-43, 6.55e-43],
        label_=[r'DYB 1.875-9 MeV', r'DYB 1.875-8.125 MeV', r'DYB PRL 2017', r'HM 1.875-9 MeV']
    )

    bin_edges = [1.875, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]  # 6 bins
    ax = plot_function(
        x_=[E_t, E_t, E_t, E_t, E_t, E_t, E_t, E_t], y_=spectra, styles=styles, label_=labels,
        ylabel_=r'spectrum [a.u.]', ylim=[-0.05e-43, 2.05e-43], y_sci=True, fig_length=7, fig_height=6
    )
    ax.get_legend().remove()
    for edge in bin_edges:
        ax.axvline(edge, 0, 1, color='k', linestyle='--', linewidth=1.)
    xx = np.linspace(5, 6, 20)
    yy1 = np.zeros(20)
    i_ = 0
    react.set_fission_fractions(fiss_f_1cycle['f_235'].iloc[i_], fiss_f_1cycle['f_239'].iloc[i_],
                                fiss_f_1cycle['f_238'].iloc[i_], fiss_f_1cycle['f_241'].iloc[i_])
    yy2 = react.reactor_model_dyb(xx, input_dyb) * react.eval_xs(xx, which_xs='SV_approx', bool_protons=False)
    ax.fill_between(xx, yy1, yy2, edgecolor='navy',  hatch='/', facecolor='w')


print("\nMean cross section per fission decreases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(xsf[0], xsf[-1],
                                                                   (xsf[-1]-xsf[0])/xsf[0]*100))


########################################################################################################################
# time evolution of mean energy per fission, fission rate, and neutrino detection rate
########################################################################################################################
plot = False

const = 6.241509e21
w_th = 4.6  # GW - Taishan core
L = 52.77e5  # TS core - baseline in cm
Np = react.eval_n_protons()
eff = 0.822

mean_e = np.zeros(len(fiss_f_1cycle.index))
for i_ in fiss_f_1cycle.index:
    mean_e[i_] = e_235*fiss_f_1cycle['f_235'].iloc[i_] + e_239*fiss_f_1cycle['f_239'].iloc[i_] \
                 + e_238*fiss_f_1cycle['f_238'].iloc[i_] + e_241*fiss_f_1cycle['f_241'].iloc[i_]

fission_rate = w_th/mean_e * const
nu_rate = fission_rate * xsf * Np * eff / (4 * math.pi * L * L)

print("\nMean energy per fission increases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(mean_e[0], mean_e[-1],
                                                                   (mean_e[-1]-mean_e[0])/mean_e[0]*100))
print("\nFission rate decreases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(fission_rate[0], fission_rate[-1],
                                                                   (fission_rate[-1]-fission_rate[0])/fission_rate[0]*100))
print("\nAntineutrino rate decreases during burnup")
print("beginning: {:.5e}; end: {:.5e}; difference: {:.2f}%".format(nu_rate[0], nu_rate[-1],
                                                                   (nu_rate[-1]-nu_rate[0])/nu_rate[0]*100))

if plot:
    yl = r'$\langle E \rangle_f$ [MeV/fission]'
    ax = plot_function(
        x_=[fiss_f_1cycle['f_239']], y_=[mean_e], xlabel_=r'$f_{239}$', ylabel_=yl,
        styles=['ko', 'k-'], label_=['PRL 2017', '1 cycle (CPC2017)'], base_major=0.05, base_minor=0.025,
        constrained_layout_=True
    )
    ax.get_legend().remove()

    yl = r'fission rate [fission/s]'
    ax = plot_function(
        x_=[fiss_f_1cycle['f_239']], y_=[fission_rate], xlabel_=r'$f_{239}$', ylabel_=yl,
        styles=['ko', 'r-'], label_=['fission rate', 'fission rate 1 cycle'], base_major=0.05, base_minor=0.025,
        constrained_layout_=True
    )
    ax.get_legend().remove()
    ax.text(0.6, 0.8, r'Taishan: W$_{\textrm{{\small{th}}}}$ = 4.6 GW', transform=ax.transAxes)

    yl = r'antineutrino rate [$\bar{\nu}$/s]'
    # nu_rate = fission_rate*xsf*Np*eff/(4*math.pi*L*L)
    ax = plot_function(
        x_=[fiss_f_1cycle['f_239']], y_=[nu_rate], xlabel_=r'$f_{239}$', ylabel_=yl,
        styles=['ko', 'r-'], label_=['nu rate', 'nu rate 1 cycle'], base_major=0.05, base_minor=0.025,
        constrained_layout_=True
    )
    ax.get_legend().remove()
    ax.text(0.55, 0.85, r'\noindent Taishan: W$_{\textrm{{\small{th}}}}$ = %.1f GW\\Baseline: L = %.2f km\\Number '
                      r'of protons: N = %.2e\\IBD efficiency: $\epsilon$ = %.3f\\NO neutrino oscillations'
            % (w_th, L*1.e-5, Np, eff), fontsize=12, transform=ax.transAxes)


E = np.arange(1.875, 9.001, 0.005)
react.set_fission_fractions(0.58, 0.3, 0.07, 0.05)
osc = OscillationProbability(inputs_json)
osc.verbose = False
osc_no = osc.eval_vacuum_prob_no(E)
flux = react.reactor_model_dyb(E, input_dyb) * w_th * const / mean_e.mean() * eff / (4*math.pi*L*L)
xs = react.eval_xs(E, which_xs="SV_approx", bool_protons=True)
norm = integrate.simps(flux*xs, E) * 86400
norm2 = integrate.simps(flux*osc_no*xs, E) * 86400

print(f"{GREEN}\nEvaluating expected number of IBD per day.{NC}")
print("\nBeginning of cycle: {:.2f} IBD/day".format(nu_rate[0]*86400))
print("End of cycle: {:.2f} IBD/day".format(nu_rate[-1]*86400))

print("\nNumber of IBD/day without neutrino oscillation: {:.2f}".format(norm))
print("Number of IBD/day with neutrino oscillation: {:.2f}".format(norm2))


########################################################################################################################
# time variation per energy bin
########################################################################################################################

E = np.arange(1.875, 9.001, 0.005)
bin_edges = [1.875, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]  # 6 bins
bin_centers = [2.4375, 3.5, 4.5, 5.5, 6.5, 8]
errorx = [0.5625, 0.5, 0.5, 0.5, 0.5, 1]

N_energies = len(bin_centers)
N_239 = len(fiss_f_1cycle)

ff_239 = np.arange(0.15, 0.405, 0.005)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[10, 6.5], sharex=True, sharey=True, constrained_layout=True)
coef_array = np.zeros(N_energies)
coef_array_hm = np.zeros(N_energies)
xsf_appo = np.zeros((N_energies, N_239))
xsf_appo_hm = np.zeros((N_energies, N_239))
for j_, ax in enumerate(axes.flat):  # loop over energy bins
    xs_appo = np.where(E <= bin_edges[j_ + 1],
                       np.where(bin_edges[j_] <= E, react.eval_xs(E, which_xs="SV_approx", bool_protons=False), 0.), 0.)

    for i_ in fiss_f_1cycle.index:  # loop over 239Pu fission fractions
        react.set_fission_fractions(fiss_f_1cycle['f_235'].iloc[i_], fiss_f_1cycle['f_239'].iloc[i_],
                                    fiss_f_1cycle['f_238'].iloc[i_], fiss_f_1cycle['f_241'].iloc[i_])
        flux_appo = np.where(E <= bin_edges[j_ + 1],
                             np.where(bin_edges[j_] <= E, react.reactor_model_dyb(E, input_dyb), 0.), 0.)
        flux_appo_hm = np.where(E <= bin_edges[j_ + 1],
                                np.where(bin_edges[j_] <= E, react.reactor_model_std(E, std_hm), 0.), 0.)
        xsf_appo[j_][i_] = integrate.simps(flux_appo*xs_appo, E)
        xsf_appo_hm[j_][i_] = integrate.simps(flux_appo_hm*xs_appo, E)

    ax.plot(fiss_f_1cycle['f_239'], xsf_appo[j_] / xsf_appo[j_].mean(), 'bo', label='model data (DYB)')
    ax.plot(fiss_f_1cycle['f_239'], xsf_appo_hm[j_] / xsf_appo_hm[j_].mean(), 'ro', label='model data (HM)')
    coef = np.polyfit(fiss_f_1cycle['f_239'], xsf_appo[j_]/xsf_appo[j_].mean(), 1)
    coef_hm = np.polyfit(fiss_f_1cycle['f_239'], xsf_appo_hm[j_] / xsf_appo_hm[j_].mean(), 1)
    coef_array[j_] = coef[0]
    coef_array_hm[j_] = coef_hm[0]
    fit_line = np.poly1d(coef)
    fit_line_hm = np.poly1d(coef_hm)
    ax.plot(ff_239, fit_line(ff_239), 'b--', label='best fit (DYB)')
    ax.plot(ff_239, fit_line_hm(ff_239), 'r--', label='best fit (HM)')
    ax.set_ylim([0.9, 1.1])
    ax.set_xlim([0.15, 0.4])
    if j_ == 0 or j_ == 3:
        ax.set_ylabel(r'$S_j$/$\bar{S}_j$')
    if j_ == 3 or j_ == 4 or j_ == 5:
        ax.set_xlabel(r'$f_{239}$')
    ax.grid(alpha=0.65)
    e_range = r"E$_{\nu}$ = %.2f-%.2f MeV" % (bin_edges[j_], bin_edges[j_+1])
    slope = r'$\bar{S}_j^{-1} d S_j/d f_{239}$ = %.2f (DYB)' % coef[0]
    slope_hm = r'$\bar{S}_j^{-1} d S_j/d f_{239}$ = %.2f (HM)' % coef_hm[0]
    ax.text(0.05, 0.21, e_range, fontsize=12, transform=ax.transAxes)
    ax.text(0.05, 0.13, slope, fontsize=12, transform=ax.transAxes)
    ax.text(0.05, 0.05, slope_hm, fontsize=12, transform=ax.transAxes)
ax.legend(loc='upper right')


########################################################################################################################
# time variation per energy bin - N samples - estimation of uncertainty
########################################################################################################################

energy = react.get_total_dyb().index.to_numpy()
N_samples = 1000

total_cov = np.load('data/total_covariance_125x125.npy')
xs_25 = react.eval_xs(energy, which_xs='SV_approx', bool_protons=False)

all_samples = np.zeros((N_239, N_samples, len(energy)))
for i_ in np.arange(N_239):
    react.set_fission_fractions(fiss_f_1cycle['f_235'].iloc[i_], fiss_f_1cycle['f_239'].iloc[i_],
                                    fiss_f_1cycle['f_238'].iloc[i_], fiss_f_1cycle['f_241'].iloc[i_])
    R = react.get_transformation_matrix()
    cov_final = np.linalg.multi_dot([R, total_cov, R.transpose()])

    input_s = react.reactor_model_matrixform(energy, input_dyb, pu_combo=True) * xs_25
    samples = stats.multivariate_normal.rvs(mean=input_s, cov=cov_final, size=N_samples)
    all_samples[i_] = samples / xs_25

xsf_appo_samples = np.zeros((N_239, N_energies, N_samples))
slopes = np.zeros((N_energies, N_samples))
for k_ in np.arange(N_samples):

    for j_ in np.arange(N_energies):
        xs_appo = np.where(E <= bin_edges[j_ + 1],
                           np.where(bin_edges[j_] <= E,
                                    react.eval_xs(E, which_xs="SV_approx", bool_protons=False), 0.), 0.)

        for i_ in np.arange(N_239):
            f_appo = interpolate.interp1d(energy, np.log(all_samples[i_, k_]), kind='linear', fill_value="extrapolate")
            appo = np.exp(f_appo(E))

            xsf_appo_samples[i_, j_, k_] = integrate.simps(appo * xs_appo, E)

        slopes[j_, k_] = np.polyfit(fiss_f_1cycle['f_239'],
                                    xsf_appo_samples[:, j_, k_] / xsf_appo_samples[:, j_, k_].mean(), 1)[0]

yerr = slopes.std(axis=1)
plt.figure(figsize=[7, 4], constrained_layout=True)
plt.plot(bin_centers, coef_array, 'bo', label='DYB model')
plt.errorbar(bin_centers, coef_array, xerr=errorx, yerr=yerr, fmt='none', ecolor='b')
plt.plot(bin_centers, coef_array_hm, 'r-', label='HM model')
# plt.errorbar(bin_centers, coef_array_hm, xerr=errorx, fmt='none', ecolor='r')
plt.grid()
plt.xlabel(r'$E_{\nu}$ [MeV]')
plt.ylabel(r'$\bar{S}_j^{-1} d S_j/d F_{239}$')
plt.ylim([-0.9, 0])
plt.xlim([0, 10])
plt.legend()


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
