import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import json
from scipy.interpolate import interp1d, Akima1DInterpolator, UnivariateSpline
from scipy import integrate
import latex
from antinu_spectrum.plot import plot_function, plot_function_residual
from antinu_spectrum.reactor import UnoscillatedReactorSpectrum
from antinu_spectrum.oscillation import OscillationProbability


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

# Haag
u238_haag = react.get_238u_haag()

# Kopeikin
u235_k = react.get_235u_kopeikin()
u238_k = react.get_238u_kopeikin()

# DYB+PP
u235_dyb_pp = react.get_235u_dyb_pp()
pu239_dyb_pp = react.get_239pu_dyb_pp()

xlim = [-0.2, 10.3]
ylim = [-0.1, 3.1]

if plot:
    ylabel = r'isotopic spectrum [N$_{\nu}$/MeV/fission]'
    ax = plot_function(
        x_=[u235_m.index, u238_m.index, pu239_m.index, pu241_m.index],
        y_=[u235_m["spectrum"], u238_m["spectrum"], pu239_m["spectrum"], pu241_m["spectrum"]],
        label_=[r'M '+U5, r'M '+U8, r'M '+Pu9, r'M '+Pu1],
        styles=['b^', 'rs', 'gp', 'mH'], ylabel_=ylabel, xlim=xlim, ylim=ylim
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ax = plot_function(
        x_=[u235_h.index, pu239_h.index, pu241_h.index],
        y_=[u235_h["spectrum"], pu239_h["spectrum"], pu241_h["spectrum"]],
        label_=[r'H '+U5, r'H '+Pu9, r'H '+Pu1],
        styles=['b^', 'gp', 'mH'], ylabel_=ylabel, xlim=xlim, ylim=ylim
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ax = plot_function(
        x_=[u235_ef.index, u238_ef.index, pu239_ef.index, pu241_ef.index],
        y_=[u235_ef["spectrum"], u238_ef["spectrum"], pu239_ef["spectrum"], pu241_ef["spectrum"]],
        label_=[r'EF '+U5, r'EF '+U8, r'EF '+Pu9, r'EF '+Pu1],
        styles=['b^', 'rs', 'gp', 'mH'], ylabel_=ylabel, xlim=xlim, ylim=ylim
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ax = plot_function(
        x_=[unfolded_u235.index, unfolded_pu_combo.index, unfolded_total.index],
        y_=[unfolded_u235["spectrum"], unfolded_pu_combo["spectrum"], unfolded_total["spectrum"]],
        label_=[r'DYB '+U5, r'DYB Pu-combo', r'DYB tot'],
        styles=['b^', 'rh', 'ko'], ylabel_=ylabel, xlim=xlim, ylim=ylim
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ylabel_pp = r'reactor spectrum [cm$^2$/MeV/fission]'
    ax = plot_function(
        x_=[unfolded_u235.index, unfolded_pu_combo.index, unfolded_total.index],
        y_=[unfolded_u235["IBD_spectrum"], unfolded_pu_combo["IBD_spectrum"], unfolded_total["IBD_spectrum"]],
        label_=[r'DYB '+U5, r'DYB Pu-combo', r'DYB tot'],
        styles=['b^', 'rh', 'ko'], ylabel_=ylabel_pp, xlim=xlim, ylim=None
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ax = plot_function(
        x_=[u235_dyb_pp.index, pu239_dyb_pp.index],
        y_=[u235_dyb_pp["spectrum"], pu239_dyb_pp["spectrum"]],
        label_=[r'DYB-PP '+U5, r'DYB-PP '+Pu9],
        styles=['b^', 'gp'], ylabel_=ylabel_pp, xlim=xlim, ylim=None
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ax = plot_function(
        x_=[u238_haag.index], y_=[u238_haag["spectrum"]],
        label_=[r'Haag ' + U8],
        styles=['rs'], ylabel_=ylabel, xlim=xlim, ylim=ylim
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ax = plot_function(
        x_=[u235_k.index, u238_k.index], y_=[u235_k["spectrum"], u238_k["spectrum"]],
        label_=[r'Kopeikin ' + U5, r'Kopeikin ' + U8],
        styles=['b^', 'rs'], ylabel_=ylabel, xlim=xlim, ylim=ylim
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')

    ax = plot_function(
        x_=[u238_ef.index, u238_m.index, u238_haag.index, u238_k.index],
        y_=[u238_ef["spectrum"], u238_m["spectrum"], u238_haag["spectrum"], u238_k["spectrum"]],
        label_=[r'EF '+U8, r'M '+U8, r'Haag ' + U8, r'Kopeikin ' + U8],
        styles=['ko', 'go', 'ro', 'yo'], ylabel_=ylabel, xlim=xlim, ylim=ylim
    )
    ax.axvline(1.806, 0, 1, color='k', linestyle=':')
    # axins = inset_axes(ax, width="30%", height="20%", bbox_to_anchor=(.55, .2, 1.2, 2),
    #                    bbox_transform=ax.transAxes, loc=3)
    # axins.tick_params(labelleft=True, labelbottom=True)
    # axins.plot(u238_haag.index, u238_haag["spectrum"], "ro", markersize=4)
    # axins.plot(u238_haag.index, u238_k["spectrum"].iloc[4:-2], "yo", markersize=3)
    # # axins.set_ylim(1.05, 1.06)
    # axins.grid(alpha=0.65)
    # axins.xaxis.set_major_locator(plticker.MultipleLocator(base=2.0))
    # axins.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))
    plot_function(
        x_=[u238_haag.index, u238_haag.index],
        y_=[u238_haag["spectrum"], u238_k["spectrum"].iloc[4:-2]],
        label_=[r'Haag ' + U8, r'Kopeikin ' + U8],
        styles=['ro', 'yo'], ylabel_=ylabel, xlim=None, ylim=None
    )

    plot_function(
        x_=[u238_haag.index],
        y_=[u238_haag["spectrum"]/u238_k["spectrum"].iloc[4:-2]],
        # y_=[(u238_k["spectrum"].iloc[4:-2]-u238_haag["spectrum"])/u238_haag["spectrum"]],
        label_=[r'Haag/Kopeikin ' + U8],
        styles=['ro'], ylabel_=r'ratio', xlim=None, ylim=[1.05, 1.058]
    )
    # axins = inset_axes(ax, width="70%", height="40%", borderpad=3)
    # axins.tick_params(labelleft=True, labelbottom=True)
    # axins.plot(u238_haag.index, u238_haag["spectrum"] / u238_k["spectrum"].iloc[4:-2], "ro")
    # axins.set_ylim(1.05, 1.06)
    # axins.grid(alpha=0.65)
    # axins.xaxis.set_major_locator(plticker.MultipleLocator(base=2.0))
    # axins.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))


########################################################################################################################
# cross section comparison
########################################################################################################################
plot = False
E = np.arange(1.81, 10.01, 0.005)  # in MeV

xs_dyb = unfolded_total["IBD_spectrum"]/unfolded_total["spectrum"]
xs_dyb_5 = unfolded_u235["IBD_spectrum"]/unfolded_u235["spectrum"]
xs_dyb_pu = unfolded_pu_combo["IBD_spectrum"]/unfolded_pu_combo["spectrum"]

if plot:
    ylabel = r'$\sigma_{\textrm{{\small{IBD}}}}$ [cm$^2$/proton]'
    # ylabel = r'$\sigma_{\textrm{{\small{IBD}}}} \times N_P$ [cm$^2$]'
    ax = plot_function_residual(
        x_=[E, E, E],
        y_=[react.eval_xs(E, which_xs="SV_CI", bool_protons=False),
            react.eval_xs(E, which_xs="VB_CI", bool_protons=False),
            react.eval_xs(E, which_xs="SV_approx", bool_protons=False)],
        label_=[r'SV$_\textrm{{\small{ci}}}$', r'VB$_\textrm{{\small{ci}}}$', r'SV$_\textrm{{\small{approc}}}$'], styles=['k-', 'r--', 'b-.'],
        ylabel_=ylabel, ylim2=[-2, 4]
    )
    ax[1].legend(loc='upper right')

    plot_function_residual(
        x_=[unfolded_total.index, unfolded_u235.index, unfolded_pu_combo.index],
        y_=[xs_dyb, xs_dyb_5, xs_dyb_pu], styles=['b.', 'r.', 'g.'],
        label_=[r's total', r's u235', r's pu\_combo'], ylabel_=ylabel,
    )

    ax = plot_function_residual(
        x_=[unfolded_total.index, unfolded_total.index, unfolded_u235.index, unfolded_pu_combo.index],
        y_=[react.eval_xs(unfolded_total.index, which_xs="VB_CI", bool_protons=False), xs_dyb, xs_dyb_5, xs_dyb_pu],
        styles=['k-', 'b.', 'r.', 'g.'],
        label_=[r'VB\_CI', r's total', r's u235', r's pu\_combo'], ylabel_=ylabel,
    )
    ax[1].legend(loc='lower left')


########################################################################################################################
# compare different interpolation (and extrapolation) methods
########################################################################################################################
plot = False
E = np.arange(1.81, 10.01, 0.005)  # in MeV - extrapolating where needed

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

xsec_sv = react.eval_xs(E, which_xs="SV_approx", bool_protons=False)  # SV IBD cross section from common inputs
xsec_sv_dyb = react.eval_xs(unfolded_total.index, which_xs="SV_approx", bool_protons=False)  # for DYB binning

# with DYB binning (based on HM)
u238_dyb = react.eval_238u(unfolded_total.index, which_input='Mueller')
pu241_dyb = react.eval_241pu(unfolded_total.index, which_input='Huber')

# interpolating DYB spectra + JUNO spectra
juno_points = unfolded_total["spectrum"] + df_235 * unfolded_u235["spectrum"] \
              + df_239 * unfolded_pu_combo["spectrum"] + df_238 * u238_dyb + (df_241 - 0.183 * df_239) * pu241_dyb

s_total_lin = interp1d(unfolded_total.index, unfolded_total["spectrum"], kind='linear', fill_value="extrapolate")
s_total_quad = interp1d(unfolded_total.index, unfolded_total["spectrum"], kind='quadratic', fill_value="extrapolate")
s_total_cub = interp1d(unfolded_total.index, unfolded_total["spectrum"], kind='cubic', fill_value="extrapolate")
s_total_akima = Akima1DInterpolator(unfolded_total.index, unfolded_total["spectrum"])

s_total_exp = react.eval_total(E, which_input='DYB')
JUNO_exp = react.reactor_model_dyb(E, dyb_input)

react.set_fission_fractions(0.564, 0.304, 0.076, 0.056)  # DYB fission fractions
flux_hm_dyb = react.reactor_model_std(E, std_hm)  # with tabulated data
dyb_correction = react.get_dybfluxbump_ratio(E)
hm_corrected = flux_hm_dyb * xsec_sv * dyb_correction
react.set_fission_fractions(0.58, 0.30, 0.07, 0.05)  # back to JUNO fission fractions

### PLOTS
if plot:
    ylabel = r'isotopic spectrum [N$_{\nu}$/MeV/fission]'
    ylabel2 = r'DYB total [cm$^2$/MeV/fission]'
    plot_function(
        x_=[E, E, E, unfolded_total.index], y_=[s_total_lin(E), s_total_quad(E), s_total_cub(E), unfolded_total["spectrum"]],
        label_=[r'linear', r'quadratic', r'cubic', r'DYB'], styles=['r', 'b--', 'g:', 'ko'],
        ylabel_=ylabel, xlim=None, ylim=None
    )
    ax = plot_function_residual(
        x_=[E, E], y_=[s_total_exp*xsec_sv, JUNO_exp*xsec_sv],
        label_=[r'DYB total', r'JUNO total'], styles=['r', 'b--'],
        ylabel_=r'DYB model [cm$^2$/MeV/fission]',
        xlim=None, ylim=None, ylim2=[-1.2, 1.3]
    )
    ax[0].axvline(8.65, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[0].axvline(1.925, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(8.65, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(1.925, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    plot_function_residual(
        x_=[E, E, E, unfolded_total.index],
        y_=[s_total_quad(E)*xsec_sv, s_total_cub(E)*xsec_sv, s_total_lin(E)*xsec_sv, unfolded_total["spectrum"]*xsec_sv_dyb],
        label_=[r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['b--', 'g:', 'r', 'ko'],
        ylabel_=ylabel2, xlim=None, ylim=None, ylim2=[-2.5, 7.5]
    )
    plot_function_residual(
        x_=[E, E, E, E, unfolded_total.index],
        y_=[s_total_akima(E, extrapolate=bool)*xsec_sv, s_total_quad(E)*xsec_sv, s_total_cub(E)*xsec_sv, s_total_lin(E)*xsec_sv, unfolded_total["spectrum"]*xsec_sv_dyb],
        label_=[r'Akima', r'quadratic', r'cubic', r'linear', r'no interp.'], styles=['c-', 'b--', 'g:', 'r', 'ko'],
        ylabel_=ylabel2, xlim=None, ylim=None, ylim2=[-2.5, 7.5]
    )
    plot_function_residual(
        x_=[E, E, E, unfolded_total.index],
        y_=[s_total_akima(E, extrapolate=bool)*xsec_sv, hm_corrected, s_total_exp*xsec_sv, unfolded_total["spectrum"]*xsec_sv_dyb],
        label_=[r'Akima', r'HM corrected', r'exponential', r'no interp.'], styles=['c-', 'k:', 'm--', 'ko'],
        ylabel_=ylabel2, xlim=None, ylim=None, ylim2=[-7.5, 10]
    )
    ax = plot_function(
        x_=[E],
        y_=[react.reactor_model_std(E, std_hm)*xsec_sv],
        label_=[r'HM'], styles=['b'],
        ylabel_=r'vanilla model [cm$^2$/MeV/fission]', xlim=None, ylim=None
    )
    ax.axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax.axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

########################################################################################################################
# EF - HM comparison - exponential interpolation
########################################################################################################################
plot = False

std_ef = {
    '235U': 'EF',
    '238U': 'EF',
    '239Pu': 'EF',
    '241Pu': 'EF'
}

# interpolating EF
ef_235 = react.eval_235u(E, which_input="EF")
ef_239 = react.eval_239pu(E, which_input="EF")
ef_238 = react.eval_238u(E, which_input="EF")
ef_241_exp = react.eval_241pu(E, which_input="EF")
ef_241_akima = Akima1DInterpolator(pu241_ef.index, pu241_ef["spectrum"])
JUNO_ef = react.reactor_model_std(E, std_ef)

# interpolating HM
hm_235 = react.eval_235u(E, which_input="Huber")
hm_239 = react.eval_239pu(E, which_input="Huber")
hm_241 = react.eval_241pu(E, which_input="Huber")
hm_238 = react.eval_238u(E, which_input="Mueller")
JUNO_hm = react.reactor_model_std(E, std_hm)

if plot:
    ylabel = r'isotopic spectrum $\times\,\sigma_{\textrm{{\small{IBD}}}}$ [a.u.]'
    ax = plot_function_residual(
        x_=[E, E], y_=[ef_235*xsec_sv, hm_235*xsec_sv], ylim2=[-5, 20],
        label_=[r'EF '+U5, r'HM '+U5], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax[0].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[0].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax = plot_function_residual(
        x_=[E, E], y_=[ef_238*xsec_sv, hm_238*xsec_sv], ylim2=[-10, 10],
        label_=[r'EF '+U8, r'HM '+U8], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax[0].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[0].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax = plot_function_residual(
        x_=[E, E], y_=[ef_239*xsec_sv, hm_239*xsec_sv], ylim2=[-15, 20],
        label_=[r'EF '+Pu9, r'HM '+Pu9], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax[0].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[0].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    ax = plot_function_residual(
        x_=[E, E], y_=[ef_241_exp*xsec_sv, hm_241*xsec_sv], ylim2=[-20, 15],
        label_=[r'EF '+Pu1, r'HM '+Pu1], styles=['r', 'b-.'], ylabel_=ylabel
    )
    ax[0].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[0].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)

    plot_function_residual(
        x_=[E, E], y_=[ef_241_exp*xsec_sv, ef_241_akima(E, extrapolate=bool)*xsec_sv],
        label_=[r'EF '+Pu1+' exp', r'EF '+Pu1+' Akima'], styles=['r', 'y-.'], ylabel_=ylabel
    )

    ax = plot_function_residual(
        x_=[E, E], y_=[JUNO_ef*xsec_sv, JUNO_hm*xsec_sv*0.95],
        label_=[r'EF', r'HM $\times$ 0.95'], styles=['r-', 'b-.'], ylabel_=r'JUNO unosc. spectrum [a.u.]',
        ylim2=[-7.5, 5]
    )
    ax[0].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[0].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(8, 0, 1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(2, 0, 1, color='k', linestyle='--', linewidth=1)


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


plot = True
react.set_fission_fractions(0.58, 0.30, 0.07, 0.05)
flux_hm = react.reactor_model_std(E, std_hm)
E2 = np.arange(8.5, 12.01, 0.005)
E_t = 8.65

u5 = react.eval_235u(E, which_input="Huber")*xsec_sv
u8 = react.eval_238u(E, which_input="Mueller")*xsec_sv
p9 = react.eval_239pu(E, which_input="Huber")*xsec_sv
p1 = react.eval_241pu(E, which_input="Huber")*xsec_sv
if plot:
    plot_function(
        x_=[E, E, E, E],
        y_=[JUNO_ef*xsec_sv, JUNO_hm*xsec_sv, flux_hm * xsec_sv * dyb_correction, JUNO_exp*xsec_sv],
        label_=[r'EF vanilla', r'HM vanilla', 'HM vanilla+corrections', r'DYB model'], y_sci=True,
        styles=['r--', 'b-.', 'g-.', 'k-'],
        ylabel_=r'JUNO reactor spectrum [a.u.]'  # ,ylim2=[-12, 20]
    )
    # ax[1].get_legend().remove()
    # ax[1].text(7, 13, r"(X - DYB)/(DYB)", fontsize=13)

    ax = plot_function(
        x_=[E],  # E, E, E],
        y_=[dyb_correction],  # JUNO_exp/JUNO_hm, JUNO_exp/JUNO_ef], # JUNO_ef/JUNO_hm],
        label_=[r'CI 2016: DYB/HM', 'DYB(2021)/HM(2011)', 'DYB(2021)/EF(2019)', 'EF(2019)/HM(2011)'], y_sci=True,
        styles=['k-', 'b-.', 'r--', 'g-.'], ylabel_=r'ratios', ylim=[0.8, 1.2],
    )
    ax.legend(loc='upper left')
    ax.axhline(1, 0, 1, color='k', linewidth=0.5)

    plot_function(
        x_=[unfolded_total.index, E, E],
        y_=[juno_points*xsec_sv_dyb, JUNO_ef*xsec_sv, dyb_ef_spectrum(E)*xsec_sv],
        label_=[r"DYB", r"EF - exp", r"DYB+EF"], y_sci=True,
        styles=['ko', 'r-', 'b--'], ylabel_=r'JUNO spectrum [a.u.]'
    )

    xs_e2 = react.eval_xs(E2, which_xs="SV_approx")
    ax = plot_function(
        x_=[unfolded_total.index.values[-1], E2, E2, E2, E2],
        y_=[juno_points.values[-1] * react.eval_xs(E_t, which_xs="SV_approx"), dyb_spectrum(E2) * xs_e2,
            ef_spectrum(E2) * xs_e2,
            dyb_ef_spectrum(E2) * xs_e2, dyb_ef_spectrum(E2, opt_="s") * xs_e2],
        label_=[r"DYB", "DYB - exp", r"EF - exp", r"DYB+EF", r"DYB+EF s", r"DYB+EF t"], y_sci=True,
        styles=['ko', 'k-', 'r-', 'b--', 'g--'], ylabel_=r'JUNO spectrum [a.u.]'
    )
    ax.legend(loc="upper right")

    ax = plot_function_residual(
        x_=[E, E, E, E],
        y_=[react.eval_238u(E, which_input="Mueller")*xsec_sv, react.eval_238u(E, which_input="EF")*xsec_sv,
            react.eval_238u(E, which_input="Haag")*xsec_sv, react.eval_238u(E, which_input="Kopeikin")*xsec_sv],
        label_=[r"M " + U8, r"EF " + U8, r"Haag " + U8, r"Kpk " + U8],
        styles=['g--', 'k--', 'r-.', 'y-.'], ylabel_=r'isotopic spectrum $\times \sigma_{\textrm{{\small{IBD}}}}$ [a.u.]'
    )
    ax[1].legend(loc="upper right")
    ax[1].get_legend().remove()
    ax[1].text(0.8, 0.8, r"(X - M)/M", fontsize=13, transform=ax[1].transAxes)

    ax = plot_function(
        x_=[E, E, E, E],
        y_=[u5/integrate.simps(u5, E), u8/integrate.simps(u8, E),
            p9/integrate.simps(p9, E), p1/integrate.simps(p1, E)],
        label_=[r"H " + U5, r"M " + U8, r"H " + Pu9, r"H " + Pu1], styles=['k--', 'r--', 'b-.', 'g:'], y_sci=True,
        ylabel_=r'S$_{\textrm{{\small{iso}}}} \times \sigma_{\textrm{{\small{IBD}}}}$ [cm$^2$/MeV/fission]'
    )

    plot_function(
        x_=[E, E, E, E], y_=[u5, u8, p9, p1],
        label_=[r"H " + U5, r"M " + U8, r"H " + Pu9, r"H " + Pu1], styles=['k--', 'r--', 'b-.', 'g:'], y_sci=True,
        ylabel_=r'S$_{\textrm{{\small{iso}}}} \times \sigma_{\textrm{{\small{IBD}}}}$ [cm$^2$/MeV/fission]'
    )


########################################################################################################################
# oscillated spectrum
########################################################################################################################
plot = False
dyb_std = {
    'total': 'DYB',
    '235U': 'DYB',
    '238U': 'Mueller',
    '239Pu': 'DYB_combo',
    '241Pu': 'Huber'
}

vanilla_hm = {
    '235U': 'Huber',
    '238U': 'Mueller',
    '239Pu': 'Huber',
    '241Pu': 'Huber'
}

vanilla_haag = {
    '235U': 'Huber',
    '238U': 'Haag',
    '239Pu': 'Huber',
    '241Pu': 'Huber'
}

dyb_haag = {
    'total': 'DYB',
    '235U': 'DYB',
    '238U': 'Haag',
    '239Pu': 'DYB_combo',
    '241Pu': 'Huber'
}

dyb_k = {
    'total': 'DYB',
    '235U': 'DYB',
    '238U': 'Kopeikin',
    '239Pu': 'DYB_combo',
    '241Pu': 'Huber'
}

vanilla_k = {
    '235U': 'Huber',
    '238U': 'Kopeikin',
    '239Pu': 'Huber',
    '241Pu': 'Huber'
}

E_s = np.arange(1.81, 10, 0.005)
s_std = react.unoscillated_reactor_spectrum(E_s, dyb_std, which_xs="SV_approx", pu_combo=True)
s_haag = react.unoscillated_reactor_spectrum(E_s, dyb_haag, which_xs="SV_approx", pu_combo=True)
s_k = react.unoscillated_reactor_spectrum(E_s, dyb_k, which_xs="SV_approx", pu_combo=True)
# react.set_fission_fractions(0.564, 0.304, 0.076, 0.056)  # DYB fission fractions
s_vanilla = react.unoscillated_reactor_spectrum(E_s, vanilla_hm, which_xs="SV_approx", pu_combo=True)
s_vanilla_haag = react.unoscillated_reactor_spectrum(E_s, vanilla_haag, which_xs="SV_approx", pu_combo=True)
s_vanilla_k = react.unoscillated_reactor_spectrum(E_s, vanilla_k, which_xs="SV_approx", pu_combo=True)

norm_std = integrate.simps(s_std, E_s)
norm_haag = integrate.simps(s_haag, E_s)
norm_k = integrate.simps(s_k, E_s)
diff = (norm_haag-norm_std)/norm_std*100.
diff2 = (norm_k-norm_std)/norm_std*100.
print("\nDYB based model")
print("Difference in normalization w/ or w/o Haag spectrum of 238U")
print("{:.6e} ; {:.6e} ; {:.4f}%".format(norm_std, norm_haag, diff))
print("Difference in normalization w/ or w/o Kopeikin spectrum of 238U")
print("{:.6e} ; {:.6e} ; {:.4f}%".format(norm_std, norm_k, diff2))

norm_std = integrate.simps(s_vanilla, E_s)
norm_haag = integrate.simps(s_vanilla_haag, E_s)
norm_k = integrate.simps(s_vanilla_k, E_s)
diff = (norm_haag-norm_std)/norm_std*100.
diff2 = (norm_k-norm_std)/norm_std*100.
print("\nVanilla model")
print("Difference in normalization w/ or w/o Haag spectrum of 238U")
print("{:.6e} ; {:.6e} ; {:.4f}%".format(norm_std, norm_haag, diff))
print("Difference in normalization w/ or w/o Kopeikin spectrum of 238U")
print("{:.6e} ; {:.6e} ; {:.4f}%".format(norm_std, norm_k, diff2))

# xs = react.eval_xs(E_s, which_xs="SV_CI")

prob = OscillationProbability(inputs_json)
prob_N_v, prob_I_v = prob.eval_vacuum_prob(plot_this=False)
prob_N_ve, prob_I_ve = prob.eval_vacuum_prob(E_s, plot_this=False)

if plot:
    ax = plot_function_residual(
        x_=[E_s, E_s, E_s], y_=[s_std, s_haag, s_k], label_=[r'DYB+M', r'DYB+Haag', 'DYB+Kpk'],
        styles=['b-', 'r--', 'g-.'], ylabel_=r'Unoscillated spectrum [N$_{\nu}$/MeV/s]', y2_sci=True
    )
    ax[1].legend(loc='upper left')

    ax = plot_function_residual(
        x_=[E_s, E_s, E_s], y_=[s_vanilla, s_vanilla_haag, s_vanilla_k], label_=[r'H+M', r'H+Haag', r'H+Kpk'],
        styles=['b-', 'r--', 'g-.'], ylabel_=r'Unoscillated spectrum [N$_{\nu}$/MeV/s]'
    )
    ax[1].legend(loc='lower left')

    plot_function_residual(
        x_=[E_s, E_s, E_s],
        y_=[s_std*prob_N_ve, s_haag*prob_N_ve, s_k*prob_N_ve],
        label_=[r'DYB+M', r'DYB+Haag', 'DYB+Kpk'], styles=['b-', 'r--', 'g-.'],
        ylabel_=r"Oscillated spectrum [N$_{\nu}$/MeV/s]", y2_sci=True
    )


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
