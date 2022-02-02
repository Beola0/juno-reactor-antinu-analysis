import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import math
import pandas as pd
from scipy.interpolate import interp1d, Akima1DInterpolator, UnivariateSpline
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')
import latex
from plot import plot_function, plot_function_residual
from reactor import UnoscillatedReactorSpectrum


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

df_235 = inputs_json["fission_fractions"]["235U"] - 0.564
df_239 = inputs_json["fission_fractions"]["239Pu"] - 0.304
df_238 = inputs_json["fission_fractions"]["238U"] - 0.076
df_241 = inputs_json["fission_fractions"]["241Pu"] - 0.056
# df_235 = 0.
# df_239 = 0.
# df_238 = 0.
# df_241 = 0.

f235_dyb = 0.564
f239_dyb = 0.304
f238_dyb = 0.076
f241_dyb = 0.056

# params for the HM model - exponential of polynomial of 5th order
params_u235 = [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]
params_pu239 = [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]
params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]


time_start = time.perf_counter_ns()
react = UnoscillatedReactorSpectrum(inputs_json)

########################################################################################################################
# read and plot tabulated spectra
########################################################################################################################
plot = False

# Mueller
u235_m = react.get_235u_mueller()
u238_m = react.get_238u_mueller()
pu239_m = react.get_239pu_mueller()
pu241_m = react.get_241pu_mueller()

# Huber
u235_h = react.get_235u_huber()
pu239_h = react.get_239pu_huber()
pu241_h = react.get_241pu_huber()

# EF
u235_ef = react.get_235u_ef()
u238_ef = react.get_238u_ef()
pu239_ef = react.get_239pu_ef()
pu241_ef = react.get_241pu_ef()

# DYB
unfolded_total = react.get_total_dyb()
unfolded_u235 = react.get_235u_dyb()
unfolded_pu_combo = react.get_pu_combo_dyb()

if plot:
    ylabel = r'isotopic spectrum [N$_{\nu}$/fission/MeV]'
    plot_function(
        x_=[u235_m.index, u238_m.index, pu239_m.index, pu241_m.index],
        y_=[u235_m["spectrum"], u238_m["spectrum"], pu239_m["spectrum"], pu241_m["spectrum"]],
        label_=[r'M '+U5, r'M '+U8, r'M '+Pu9, r'M '+Pu1],
        styles=['b^', 'rs', 'gp', 'mH'], ylabel_=ylabel, xlim=None, ylim=None
    )
    plot_function(
        x_=[u235_h.index, pu239_h.index, pu241_h.index],
        y_=[u235_h["spectrum"], pu239_h["spectrum"], pu241_h["spectrum"]],
        label_=[r'H '+U5, r'H '+Pu9, r'H '+Pu1],
        styles=['b^', 'gp', 'mH'], ylabel_=ylabel, xlim=None, ylim=None
    )
    ax = plot_function(
        x_=[u235_ef.index, u238_ef.index, pu239_ef.index, pu241_ef.index],
        y_=[u235_ef["spectrum"], u238_ef["spectrum"], pu239_ef["spectrum"], pu241_ef["spectrum"]],
        label_=[r'EF '+U5, r'EF '+U8, r'EF '+Pu9, r'EF '+Pu1],
        styles=['b^', 'rs', 'gp', 'mH'], ylabel_=ylabel, xlim=None, ylim=None
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')
    plot_function(
        x_=[unfolded_u235.index, unfolded_pu_combo.index, unfolded_total.index],
        y_=[unfolded_u235["spectrum"], unfolded_pu_combo["spectrum"], unfolded_total["spectrum"]],
        label_=[r'DYB '+U5, r'DYB Pu-comb', r'DYB tot'],
        styles=['b^', 'rh', 'ko'], ylabel_=ylabel, xlim=None, ylim=None
    )


########################################################################################################################
# cross section comparison
########################################################################################################################
plot = False
E = np.arange(1.81, 10.01, 0.005)  # in MeV

if plot:
    ylabel = r'$\sigma_{\text{IBD}}$ [\si{\centi\meter\squared}/proton]'
    # ylabel = r'$\sigma_{\text{IBD}} \times N_P$ [\si{\centi\meter\squared}]'
    ax1, ax2 = plot_function_residual(
        x_=[E, E, E],
        y_=[react.eval_xs(E, which_xs="SV_CI", bool_protons=False),
            react.eval_xs(E, which_xs="VB_CI", bool_protons=False),
            react.eval_xs(E, which_xs="SV_approx", bool_protons=False)],
        label_=[r'SV$_\text{ci}$', r'VB$_\text{ci}$', r'SV$_\text{approx}$'], styles=['k-', 'r--', 'b-.'],
        ylabel_=ylabel, ylim2=[-2, 5])
    ax2.legend(loc='upper right')


########################################################################################################################
# compare different interpolation (and extrapolation) methods
########################################################################################################################
plot = False
E = np.arange(1.81, 10.01, 0.005)  # in MeV - extrapolating where needed

flux_hm = react.isotopic_spectrum_hubermueller_parametric(E, plot_this=False)  # with exp-polynomial formula
xsec_sv = react.eval_xs(E, which_xs="SV_CI")  # SV IBD cross section from common inputs
xsec_sv_dyb = react.eval_xs(unfolded_total.index, which_xs="SV_CI")  # for DYB binning

# HM - exp-pol parametrization
u235_hm = react.isotopic_spectrum_exp(E, params_u235)
pu239_hm = react.isotopic_spectrum_exp(E, params_pu239)
u238_hm = react.isotopic_spectrum_exp(E, params_u238)
pu241_hm = react.isotopic_spectrum_exp(E, params_pu241)

# with DYB binning (based on HM)
u238_dyb = react.isotopic_spectrum_exp(unfolded_total.index, params_u238)
pu241_dyb = react.isotopic_spectrum_exp(unfolded_total.index, params_pu241)

# interpolating DYB spectra + JUNO spectra
juno_points = unfolded_total["spectrum"] + df_235 * unfolded_u235["spectrum"] \
              + df_239 * unfolded_pu_combo["spectrum"] + df_238 * u238_dyb + (df_241 - 0.183 * df_239) * pu241_dyb

s_total_lin = interp1d(unfolded_total.index, unfolded_total["spectrum"], kind='linear', fill_value="extrapolate")
s_235_lin = interp1d(unfolded_u235.index, unfolded_u235["spectrum"], kind='linear', fill_value="extrapolate")
s_combo_lin = interp1d(unfolded_pu_combo.index, unfolded_pu_combo["spectrum"], kind='linear', fill_value="extrapolate")
JUNO_lin = s_total_lin(E) + df_235 * s_235_lin(E) + df_239 * s_combo_lin(E) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_quad = interp1d(unfolded_total.index, unfolded_total["spectrum"], kind='quadratic', fill_value="extrapolate")
s_235_quad = interp1d(unfolded_u235.index, unfolded_u235["spectrum"], kind='quadratic', fill_value="extrapolate")
s_combo_quad = interp1d(unfolded_pu_combo.index, unfolded_pu_combo["spectrum"], kind='quadratic', fill_value="extrapolate")
JUNO_quad = s_total_quad(E) + df_235 * s_235_quad(E) + df_239 * s_combo_quad(E) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_cub = interp1d(unfolded_total.index, unfolded_total["spectrum"], kind='cubic', fill_value="extrapolate")
s_235_cub = interp1d(unfolded_u235.index, unfolded_u235["spectrum"], kind='cubic', fill_value="extrapolate")
s_combo_cub = interp1d(unfolded_pu_combo.index, unfolded_pu_combo["spectrum"], kind='cubic', fill_value="extrapolate")
JUNO_cub = s_total_cub(E) + df_235 * s_235_cub(E) + df_239 * s_combo_cub(E) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_akima = Akima1DInterpolator(unfolded_total.index, unfolded_total["spectrum"])
s_235_akima = Akima1DInterpolator(unfolded_u235.index, unfolded_u235["spectrum"])
s_combo_akima = Akima1DInterpolator(unfolded_pu_combo.index, unfolded_pu_combo["spectrum"])
JUNO_akima = s_total_akima(E, extrapolate=bool) + df_235 * s_235_akima(E, extrapolate=bool) + df_239 * s_combo_akima(E, extrapolate=bool) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_exp = react.eval_total(E, which_input='DYB')
s_235_exp = react.eval_235u(E, which_input='DYB')
s_combo_exp = react.eval_239pu(E, which_input='DYB_combo')
JUNO_exp = s_total_exp + df_235 * s_235_exp + df_239 * s_combo_exp \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

dyb_correction = react.get_dybfluxbump_ratio(E)
hm_corrected = flux_hm * xsec_sv * dyb_correction

### PLOTS
if plot:
    ylabel = r'isotopic spectrum [N$_{\nu}$/fission/MeV]'
    plot_function(
        x_=[E, E, E, unfolded_total.index], y_=[s_total_lin(E), s_total_quad(E), s_total_cub(E), unfolded_total["spectrum"]],
        label_=[r'linear', r'quadratic', r'cubic', r'DYB'], styles=['r', 'b--', 'g:', 'ko'],
        ylabel_=ylabel, xlim=None, ylim=None
    )
    plot_function_residual(
        x_=[E, E], y_=[s_total_exp*xsec_sv, JUNO_exp*xsec_sv],
        label_=[r'DYB total', r'JUNO total'], styles=['r', 'b--'], ylabel_=r'tot spectrum [a.u.]',
        xlim=None, ylim=None, ylim2=[-3, 3]
    )
    plot_function_residual(
        x_=[E, E, E, unfolded_total.index],
        y_=[s_total_quad(E)*xsec_sv, s_total_cub(E)*xsec_sv, s_total_lin(E)*xsec_sv, unfolded_total["spectrum"]*xsec_sv_dyb],
        label_=[r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['b--', 'g:', 'r', 'ko'],
        ylabel_=r'DYB tot spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-2.5, 7.5]
    )
    plot_function_residual(
        x_=[E, E, E, unfolded_total.index],
        y_=[JUNO_quad*xsec_sv, JUNO_cub*xsec_sv, JUNO_lin*xsec_sv, juno_points*xsec_sv_dyb],
        label_=[r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['b--', 'g:', 'r', 'ko'],
        ylabel_=r'JUNO unosc. spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-2.5, 7.5]
    )
    plot_function_residual(
        x_=[E, E, E, E, unfolded_total.index],
        y_=[JUNO_akima*xsec_sv, JUNO_quad*xsec_sv, JUNO_cub*xsec_sv, JUNO_lin*xsec_sv, juno_points*xsec_sv_dyb],
        label_=[r'Akima', r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['c-', 'b--', 'g:', 'r', 'ko'],
        ylabel_=r'JUNO unosc. spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-2.5, 7.5]
    )
    plot_function_residual(
        x_=[E, E, E, unfolded_total.index],
        y_=[JUNO_akima*xsec_sv, hm_corrected, JUNO_exp*xsec_sv, juno_points*xsec_sv_dyb],
        label_=[r'Akima', r'HM corrected', r'exponential', r'no interp.'], styles=['c-', 'k:', 'm--', 'ko'],
        ylabel_=r'JUNO unosc. spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-7.5, 10]
    )

########################################################################################################################
# EF - HM comparison - exponential interpolation
########################################################################################################################
plot = False

# interpolating EF
ef_235 = react.eval_235u(E, which_input="EF")
ef_239 = react.eval_239pu(E, which_input="EF")
ef_238 = react.eval_238u(E, which_input="EF")
ef_241_exp = react.eval_241pu(E, which_input="EF")
ef_241_akima = Akima1DInterpolator(pu241_ef.index, pu241_ef["spectrum"])
JUNO_ef = inputs_json["fission_fractions"]["235U"] * ef_235 + inputs_json["fission_fractions"]["238U"] * ef_238 \
          + inputs_json["fission_fractions"]["239Pu"] * ef_239 + inputs_json["fission_fractions"]["241Pu"] * ef_241_exp

# interpolating HM
hm_235 = react.eval_235u(E, which_input="Huber")
hm_239 = react.eval_239pu(E, which_input="Huber")
hm_241 = react.eval_241pu(E, which_input="Huber")
hm_238 = react.eval_238u(E, which_input="Mueller")
JUNO_hm = inputs_json["fission_fractions"]["235U"] * hm_235 + inputs_json["fission_fractions"]["238U"] * hm_238 \
          + inputs_json["fission_fractions"]["239Pu"] * hm_239 + inputs_json["fission_fractions"]["241Pu"] * hm_241

if plot:
    ylabel = r'isotopic spectrum $\times\,\sigma_{\text{IBD}}$ [a.u.]'
    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[ef_235*xsec_sv, hm_235*xsec_sv], ylim2=[-5, 20],
        label_=[r'EF '+U5, r'HM '+U5], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[ef_238*xsec_sv, hm_238*xsec_sv], ylim2=[-10, 10],
        label_=[r'EF '+U8, r'HM '+U8], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[ef_239*xsec_sv, hm_239*xsec_sv], ylim2=[-15, 20],
        label_=[r'EF '+Pu9, r'HM '+Pu9], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[ef_241_exp*xsec_sv, hm_241*xsec_sv], ylim2=[-20, 15],
        label_=[r'EF '+Pu1, r'HM '+Pu1], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    plot_function_residual(
        x_=[E, E], y_=[ef_241_exp*xsec_sv, ef_241_akima(E, extrapolate=bool)*xsec_sv],
        label_=[r'EF '+Pu1+' exp', r'EF '+Pu1+' Akima'], styles=['r', 'y-.'], ylabel_=ylabel
    )

    plot_function_residual(
        x_=[E, E], y_=[JUNO_ef*xsec_sv, JUNO_hm*xsec_sv*0.95],
        label_=[r'EF', r'HM $\times$ 0.95'], styles=['r-', 'b-.'], ylabel_=r'JUNO unosc. spectrum [a.u.]',
        ylim2=[-7.5, 5]
    )

########################################################################################################################
# final comparison
########################################################################################################################


def ef_spectrum(x_):
    y_ = inputs_json["fission_fractions"]["235U"] * react.eval_235u(x_, which_input="EF") +\
         inputs_json["fission_fractions"]["238U"] * react.eval_238u(x_, which_input="EF") +\
         inputs_json["fission_fractions"]["239Pu"] * react.eval_239pu(x_, which_input="EF") + \
         inputs_json["fission_fractions"]["241Pu"] * react.eval_241pu(x_, which_input="EF")
    return y_


def dyb_spectrum(x_):
    y_ = react.eval_total(x_, which_input='DYB') + df_235 * react.eval_235u(x_, which_input="DYB") \
         + df_239 * react.eval_239pu(x_, which_input='DYB_combo') \
         + df_238 * react.eval_238u(x_, which_input="Mueller") \
         + (df_241 - 0.183 * df_239) * react.eval_241pu(x_, which_input="Huber")
    return y_


def dyb_ef_spectrum(x_, opt_=''):

    e_thr = 8.65
    scale_factor = dyb_spectrum(e_thr) / ef_spectrum(e_thr)
    if opt_ == "s":
        y_ = np.where(x_ < e_thr, dyb_spectrum(x_), ef_spectrum(x_)*scale_factor)
    else:
        y_ = np.where(x_ < e_thr, dyb_spectrum(x_), ef_spectrum(x_))

    return y_


plot = False
E2 = np.arange(8.5, 12.01, 0.005)
E_t = 8.65
if plot:
    ax1, ax2 = plot_function_residual(
        x_=[E, E, E, E, E],
        y_=[JUNO_exp*xsec_sv, JUNO_ef*xsec_sv, JUNO_hm*xsec_sv, flux_hm * xsec_sv, flux_hm * xsec_sv * dyb_correction],
        label_=[r'exp DYB', r'exp EF', r'exp HM', 'HM exp-pol', 'HM corrected'],
        styles=['k-', 'r--', 'b-.', 'g-.', 'y-.'], ylabel_=r'JUNO unosc. spectrum [a.u.]', ylim2=[-12, 20]
    )
    ax2.get_legend().remove()
    ax2.text(7, 13, r"(X - exp DYB)/(exp DYB)", fontsize=13)

    plot_function(
        x_=[E, E, E, E], y_=[dyb_correction, JUNO_exp/JUNO_hm, JUNO_exp/JUNO_ef, JUNO_ef/JUNO_hm], ylim=[0.8, 1.2],
        label_=[r'input DYB/HM (2016)', 'DYB(2021)/HM(2011)', 'DYB(2021)/EF(2019)', 'EF(2019)/HM(2011)'],
        styles=['k-', 'b-.', 'r--', 'g-.'], ylabel_=r'JUNO spectra ratios'
    )

    plot_function(
        x_=[unfolded_total.index, E, E],
        y_=[juno_points*xsec_sv_dyb, JUNO_ef*xsec_sv, dyb_ef_spectrum(E)*xsec_sv],
        label_=[r"DYB", r"EF - exp", r"DYB+EF"],
        styles=['ko', 'r-', 'b--'], ylabel_=r'JUNO spectrum [a.u.]'
    )

    xs_e2 = react.eval_xs(E2, which_xs="SV_CI")
    ax = plot_function(
        x_=[unfolded_total.index.values[-1], E2, E2, E2, E2],
        y_=[juno_points.values[-1] * react.eval_xs(E_t, which_xs="SV_CI"), dyb_spectrum(E2) * xs_e2,
            ef_spectrum(E2) * xs_e2,
            dyb_ef_spectrum(E2) * xs_e2, dyb_ef_spectrum(E2, opt_="s") * xs_e2],
        label_=[r"DYB", "DYB - exp", r"EF - exp", r"DYB+EF", r"DYB+EF s", r"DYB+EF t"],
        styles=['ko', 'k-', 'r-', 'b--', 'g--'], ylabel_=r'JUNO spectrum [a.u.]'
    )
    ax.legend(loc="upper right")


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
