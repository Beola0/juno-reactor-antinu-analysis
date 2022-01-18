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
from reactor import ReactorSpectrum


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


########################################################################################################################
# read and plot tabulated spectra
########################################################################################################################
time_start = time.perf_counter_ns()
plot = False
ylabel = r'isotopic spectrum [N$_{\nu}$/fission/MeV]'
# Mueller
u235_m = pd.read_csv("Inputs/spectra/u235_mueller.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
u238_m = pd.read_csv("Inputs/spectra/u238_mueller.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
pu239_m = pd.read_csv("Inputs/spectra/pu239_mueller.csv", sep=",",
                            names=["bin_center", "isotopic_spectrum"], header=0)
pu241_m = pd.read_csv("Inputs/spectra/pu241_mueller.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
if plot:
    plot_function(
        x_=[u235_m["bin_center"], u238_m["bin_center"], pu239_m["bin_center"], pu241_m["bin_center"]],
        y_=[u235_m["isotopic_spectrum"], u238_m["isotopic_spectrum"], pu239_m["isotopic_spectrum"], pu241_m["isotopic_spectrum"]],
        label_=[r'M '+U5, r'M '+U8, r'M '+Pu9, r'M '+Pu1],
        styles=['b^', 'rs', 'gp', 'mH'], ylabel_=ylabel, xlim=None, ylim=None
    )

# Huber
u235_h = pd.read_csv("Inputs/spectra/u235_huber.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
pu239_h = pd.read_csv("Inputs/spectra/pu239_huber.csv", sep=",",
                            names=["bin_center", "isotopic_spectrum"], header=0)
pu241_h = pd.read_csv("Inputs/spectra/pu241_huber.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
if plot:
    plot_function(
        x_=[u235_h["bin_center"], pu239_h["bin_center"], pu241_h["bin_center"]],
        y_=[u235_h["isotopic_spectrum"], pu239_h["isotopic_spectrum"], pu241_h["isotopic_spectrum"]],
        label_=[r'H '+U5, r'H '+Pu9, r'H '+Pu1],
        styles=['b^', 'gp', 'mH'], ylabel_=ylabel, xlim=None, ylim=None
    )

# EF
u235_ef = pd.read_csv("Inputs/spectra/u235_EF.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
u238_ef = pd.read_csv("Inputs/spectra/u238_EF.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
pu239_ef = pd.read_csv("Inputs/spectra/pu239_EF.csv", sep=",",
                            names=["bin_center", "isotopic_spectrum"], header=0)
pu241_ef = pd.read_csv("Inputs/spectra/pu241_EF.csv", sep=",",
                                names=["bin_center", "isotopic_spectrum"], header=0)
if plot:
    ax = plot_function(
        x_=[u235_ef["bin_center"], u238_ef["bin_center"], pu239_ef["bin_center"], pu241_ef["bin_center"]],
        y_=[u235_ef["isotopic_spectrum"], u238_ef["isotopic_spectrum"], pu239_ef["isotopic_spectrum"], pu241_ef["isotopic_spectrum"]],
        label_=[r'EF '+U5, r'EF '+U8, r'EF '+Pu9, r'EF '+Pu1],
        styles=['b^', 'rs', 'gp', 'mH'], ylabel_=ylabel, xlim=None, ylim=None
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

# DYB
unfolded_spectrum = pd.read_csv("Inputs/spectra/total_unfolded_DYB.csv", sep=",",
                                names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=0)
unfolded_u235 = pd.read_csv("Inputs/spectra/u235_unfolded_DYB.csv", sep=",",
                            names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=0)
unfolded_pu_combo = pd.read_csv("Inputs/spectra/pu_combo_unfolded_DYB.csv", sep=",",
                                names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=0)
if plot:
    plot_function(
        x_=[unfolded_u235["bin_center"], unfolded_pu_combo["bin_center"], unfolded_spectrum["bin_center"]],
        y_=[unfolded_u235["isotopic_spectrum"], unfolded_pu_combo["isotopic_spectrum"], unfolded_spectrum["isotopic_spectrum"]],
        label_=[r'DYB '+U5, r'DYB Pu-comb', r'DYB tot'],
        styles=['b^', 'rh', 'ko'], ylabel_=ylabel, xlim=None, ylim=None
    )


########################################################################################################################
# compare different interpolation (and extrapolation) methods
########################################################################################################################
plot = False
E = np.arange(1.81, 10.01, 0.005)  # in MeV
# E = np.arange(1.925, 8.65, 0.01)  # in MeV (for DYB reactor model)

react = ReactorSpectrum(inputs_json)
flux_hm = react.isotopic_spectrum_hubermueller(E, plot_this=False)  # with exp-polynomial formula
xsec_sv = react.cross_section_sv(E)  # SV IBD cross section from common inputs

# HM - exp-pol parametrization
u235_hm = react.isotopic_spectrum_exp(E, params_u235)
pu239_hm = react.isotopic_spectrum_exp(E, params_pu239)
u238_hm = react.isotopic_spectrum_exp(E, params_u238)
pu241_hm = react.isotopic_spectrum_exp(E, params_pu241)

# with DYB binning (based on HM)
u238_dyb = react.isotopic_spectrum_exp(unfolded_spectrum["bin_center"], params_u238)
pu241_dyb = react.isotopic_spectrum_exp(unfolded_spectrum["bin_center"], params_pu241)
xsec_sv_dyb = react.cross_section_sv(unfolded_spectrum["bin_center"])

# interpolating DYB spectra + JUNO spectra
juno_points = unfolded_spectrum["isotopic_spectrum"] + df_235 * unfolded_u235["isotopic_spectrum"] \
              + df_239 * unfolded_pu_combo["isotopic_spectrum"] + df_238 * u238_dyb + (df_241 - 0.183 * df_239) * pu241_dyb

s_total_lin = interp1d(unfolded_spectrum["bin_center"], unfolded_spectrum["isotopic_spectrum"], kind='linear', fill_value="extrapolate")
s_235_lin = interp1d(unfolded_u235["bin_center"], unfolded_u235["isotopic_spectrum"], kind='linear', fill_value="extrapolate")
s_combo_lin = interp1d(unfolded_pu_combo["bin_center"], unfolded_pu_combo["isotopic_spectrum"], kind='linear', fill_value="extrapolate")
JUNO_lin = s_total_lin(E) + df_235 * s_235_lin(E) + df_239 * s_combo_lin(E) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_quad = interp1d(unfolded_spectrum["bin_center"], unfolded_spectrum["isotopic_spectrum"], kind='quadratic', fill_value="extrapolate")
s_235_quad = interp1d(unfolded_u235["bin_center"], unfolded_u235["isotopic_spectrum"], kind='quadratic', fill_value="extrapolate")
s_combo_quad = interp1d(unfolded_pu_combo["bin_center"], unfolded_pu_combo["isotopic_spectrum"], kind='quadratic', fill_value="extrapolate")
JUNO_quad = s_total_quad(E) + df_235 * s_235_quad(E) + df_239 * s_combo_quad(E) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_cub = interp1d(unfolded_spectrum["bin_center"], unfolded_spectrum["isotopic_spectrum"], kind='cubic', fill_value="extrapolate")
s_235_cub = interp1d(unfolded_u235["bin_center"], unfolded_u235["isotopic_spectrum"], kind='cubic', fill_value="extrapolate")
s_combo_cub = interp1d(unfolded_pu_combo["bin_center"], unfolded_pu_combo["isotopic_spectrum"], kind='cubic', fill_value="extrapolate")
JUNO_cub = s_total_cub(E) + df_235 * s_235_cub(E) + df_239 * s_combo_cub(E) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_akima = Akima1DInterpolator(unfolded_spectrum["bin_center"], unfolded_spectrum["isotopic_spectrum"])
s_235_akima = Akima1DInterpolator(unfolded_u235["bin_center"], unfolded_u235["isotopic_spectrum"])
s_combo_akima = Akima1DInterpolator(unfolded_pu_combo["bin_center"], unfolded_pu_combo["isotopic_spectrum"])
JUNO_akima = s_total_akima(E, extrapolate=bool) + df_235 * s_235_akima(E, extrapolate=bool) + df_239 * s_combo_akima(E, extrapolate=bool) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

s_total_exp = interp1d(unfolded_spectrum["bin_center"], unfolded_spectrum["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
s_235_exp = interp1d(unfolded_spectrum["bin_center"], unfolded_u235["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
s_combo_exp = interp1d(unfolded_spectrum["bin_center"], unfolded_pu_combo["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
JUNO_exp = np.exp(s_total_exp(E)) + df_235 * np.exp(s_235_exp(E)) + df_239 * np.exp(s_combo_exp(E)) \
                                + df_238 * u238_hm + (df_241 - 0.183 * df_239) * pu241_hm

dyb_correction = react.get_dybfluxbump_ratio(E)
hm_corrected = flux_hm * xsec_sv * dyb_correction

### PLOTS
if plot:
    plot_function(
        x_=[E, E, E, unfolded_spectrum["bin_center"]], y_=[s_total_lin(E), s_total_quad(E), s_total_cub(E), unfolded_spectrum["isotopic_spectrum"]],
        label_=[r'linear', r'quadratic', r'cubic', r'DYB'], styles=['r', 'b--', 'g:', 'ko'],
        ylabel_=ylabel, xlim=None, ylim=None
    )

    plot_function_residual(
        x_=[E, E], y_=[np.exp(s_total_exp(E))*xsec_sv, JUNO_exp*xsec_sv],
        label_=[r'DYB total', r'JUNO total'], styles=['r', 'b--'], ylabel_=r'tot spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-3, 3]
    )

    plot_function_residual(
        x_=[E, E, E, unfolded_spectrum["bin_center"]],
        y_=[s_total_quad(E)*xsec_sv, s_total_cub(E)*xsec_sv, s_total_lin(E)*xsec_sv, unfolded_spectrum["isotopic_spectrum"]*xsec_sv_dyb],
        label_=[r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['b--', 'g:', 'r', 'ko'],
        ylabel_=r'DYB tot spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-20, 30]
    )

    plot_function_residual(
        x_=[E, E, E, unfolded_spectrum["bin_center"]],
        y_=[JUNO_quad*xsec_sv, JUNO_cub*xsec_sv, JUNO_lin*xsec_sv, juno_points*xsec_sv_dyb],
        label_=[r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['b--', 'g:', 'r', 'ko'],
        ylabel_=r'JUNO unosc. spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-20, 30]
    )

    plot_function_residual(
        x_=[E, E, E, E, unfolded_spectrum["bin_center"]],
        y_=[JUNO_akima*xsec_sv, JUNO_quad*xsec_sv, JUNO_cub*xsec_sv, JUNO_lin*xsec_sv, juno_points*xsec_sv_dyb],
        label_=[r'Akima', r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['c-', 'b--', 'g:', 'r', 'ko'],
        ylabel_=r'JUNO unosc. spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-20, 30]
    )

    plot_function_residual(
        x_=[E, E, E, unfolded_spectrum["bin_center"]],
        y_=[JUNO_akima*xsec_sv, hm_corrected, JUNO_exp*xsec_sv, juno_points*xsec_sv_dyb],
        label_=[r'Akima', r'HM corrected', r'exponential', r'no interp.'], styles=['c-', 'k:', 'm--', 'ko'],
        ylabel_=r'JUNO unosc. spectrum [a.u.]', xlim=None, ylim=None, ylim2=[-20, 30]
    )


########################################################################################################################
# EF - HM comparison - exponential interpolation
########################################################################################################################
plot = False
# interpolating EF
ef_235 = interp1d(u235_ef["bin_center"], u235_ef["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
ef_239 = interp1d(pu239_ef["bin_center"], pu239_ef["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
ef_238 = interp1d(u238_ef["bin_center"], u238_ef["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
ef_241_exp = interp1d(pu241_ef["bin_center"], pu241_ef["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
ef_241_akima = Akima1DInterpolator(pu241_ef["bin_center"], pu241_ef["isotopic_spectrum"])
JUNO_ef = inputs_json["fission_fractions"]["235U"] * np.exp(ef_235(E)) + inputs_json["fission_fractions"]["238U"] * np.exp(ef_238(E)) \
          + inputs_json["fission_fractions"]["239Pu"] * np.exp(ef_239(E)) + inputs_json["fission_fractions"]["241Pu"] * np.exp(ef_241_exp(E))

# interpolating HM
hm_235 = interp1d(u235_h["bin_center"], u235_h["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
hm_239 = interp1d(pu239_h["bin_center"], pu239_h["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
hm_241 = interp1d(pu241_h["bin_center"], pu241_h["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
hm_238 = interp1d(u238_m["bin_center"], u238_m["isotopic_spectrum"].apply(lambda x: math.log(x)), kind='linear', fill_value="extrapolate")
JUNO_hm = inputs_json["fission_fractions"]["235U"] * np.exp(hm_235(E)) + inputs_json["fission_fractions"]["238U"] * np.exp(hm_238(E)) \
          + inputs_json["fission_fractions"]["239Pu"] * np.exp(hm_239(E)) + inputs_json["fission_fractions"]["241Pu"] * np.exp(hm_241(E))

if plot:
    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[np.exp(ef_235(E))*xsec_sv, np.exp(hm_235(E))*xsec_sv], ylim2=[-10, 20],
        label_=[r'EF '+U5, r'HM '+U5], styles=['r', 'b-.'], ylabel_=r'isotopic spectrum $\times\,\sigma_{\text{IBD}}$ [a.u.]'
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[np.exp(ef_238(E))*xsec_sv, np.exp(hm_238(E))*xsec_sv], ylim2=[-20, 20],
        label_=[r'EF '+U8, r'HM '+U8], styles=['r', 'b-.'], ylabel_=r'isotopic spectrum $\times\,\sigma_{\text{IBD}}$ [a.u.]'
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[np.exp(ef_239(E))*xsec_sv, np.exp(hm_239(E))*xsec_sv], ylim2=[-20, 20],
        label_=[r'EF '+Pu9, r'HM '+Pu9], styles=['r', 'b-.'], ylabel_=r'isotopic spectrum $\times\,\sigma_{\text{IBD}}$ [a.u.]'
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax1, ax2 = plot_function_residual(
        x_=[E, E], y_=[np.exp(ef_241_exp(E))*xsec_sv, np.exp(hm_241(E))*xsec_sv], ylim2=[-20, 20],
        label_=[r'EF '+Pu1, r'HM '+Pu1], styles=['r', 'b-.'], ylabel_=r'isotopic spectrum $\times\,\sigma_{\text{IBD}}$ [a.u.]'
    )
    ax1.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax2.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    plot_function_residual(
        x_=[E, E], y_=[np.exp(ef_241_exp(E))*xsec_sv, ef_241_akima(E, extrapolate=bool)*xsec_sv],
        label_=[r'EF '+Pu1+' exp', r'EF '+Pu1+' Akima'], styles=['r', 'y-.'], ylabel_=r'isotopic spectrum $\times\,\sigma_{\text{IBD}}$ [a.u.]'
    )

    plot_function_residual(
        x_=[E, E], y_=[JUNO_ef*xsec_sv, JUNO_hm*xsec_sv*0.95],
        label_=[r'EF', r'HM $\times$ 0.95'], styles=['r-', 'b-.'], ylabel_=r'JUNO unosc. spectrum [a.u.]',
        # ylim2=[-12, 20]
    )

########################################################################################################################
# final comparison
########################################################################################################################


def ef_spectrum(x_):
    y_ = inputs_json["fission_fractions"]["235U"] * np.exp(ef_235(x_)) +\
         inputs_json["fission_fractions"]["238U"] * np.exp(ef_238(x_)) +\
         inputs_json["fission_fractions"]["239Pu"] * np.exp(ef_239(x_)) + \
         inputs_json["fission_fractions"]["241Pu"] * np.exp(ef_241_exp(x_))
    return y_


def dyb_spectrum(x_):
    u8 = react.isotopic_spectrum_exp(x_, params_u238)
    pu1 = react.isotopic_spectrum_exp(x_, params_pu241)
    y_ = np.exp(s_total_exp(x_)) + df_235 * np.exp(s_235_exp(x_)) + df_239 * np.exp(s_combo_exp(x_)) \
         + df_238 * u8 + (df_241 - 0.183 * df_239) * pu1
    return y_


def dyb_ef_spectrum(x_, opt_=''):

    e_thr = 8.65
    scale_factor = dyb_spectrum(8.65) / ef_spectrum(8.65)
    if opt_ == "s":
        y_ = np.where(x_ < e_thr, dyb_spectrum(x_), ef_spectrum(x_)*scale_factor)
    else:
        y_ = np.where(x_ < e_thr, dyb_spectrum(x_), ef_spectrum(x_))

    return y_


plot = True
E2 = np.arange(8.5, 12.01, 0.005)
E_t = 8.65
if plot:
    ax1, ax2 = plot_function_residual(
        x_=[E, E, E, E, E], y_=[JUNO_exp*xsec_sv, JUNO_ef*xsec_sv, JUNO_hm*xsec_sv, flux_hm * xsec_sv, flux_hm * xsec_sv * dyb_correction],
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
        x_=[unfolded_spectrum["bin_center"], E, E],
        y_=[juno_points*xsec_sv_dyb, JUNO_ef*xsec_sv, dyb_ef_spectrum(E)*xsec_sv],
        label_=[r"DYB", r"EF - exp", r"DYB+EF"],
        styles=['ko', 'r-', 'b--'], ylabel_=r'JUNO spectrum [a.u.]'
    )

    ax = plot_function(
        x_=[unfolded_spectrum["bin_center"].values[-1], E2, E2, E2, E2],
        y_=[juno_points.values[-1] * react.cross_section_sv(E_t), dyb_spectrum(E2) * react.cross_section_sv(E2),
            ef_spectrum(E2) * react.cross_section_sv(E2),
            dyb_ef_spectrum(E2) * react.cross_section_sv(E2), dyb_ef_spectrum(E2, opt_="s") * react.cross_section_sv(E2)],
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
