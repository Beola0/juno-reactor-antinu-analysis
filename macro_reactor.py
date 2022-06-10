import numpy as np
import matplotlib.pyplot as plt
import time
import json
import latex
from antinu_spectrum.plot import plot_function
from antinu_spectrum.reactor import UnoscillatedReactorSpectrum


### MAIN ###
time_start = time.perf_counter_ns()

f = open('data/nominal_inputs.json')
# f = open('data/nominal_inputs.json')
# f = open('data/YB_inputs.json')
inputs_json = json.load(f)

# E = np.arange(1.806, 10.01, 0.01)  # in MeV
# E = np.arange(1., 12.01, 0.01)  # in MeV
E = np.arange(1.925, 8.65, 0.01)  # in MeV (for DYB reactor model)

dyb_input = {
    'total': 'DYB',
    '235U': 'DYB',
    '238U': 'HM_parametric',
    '239Pu': 'DYB_combo',
    '241Pu': 'HM_parametric'
}

std_vogel = {
    '235U': 'V_parametric',
    '238U': 'V_parametric',
    '239Pu': 'V_parametric',
    '241Pu': 'V_parametric'
}

std_hm = {
    '235U': 'HM_parametric',
    '238U': 'HM_parametric',
    '239Pu': 'HM_parametric',
    '241Pu': 'HM_parametric'
}

### REACTOR SPECTRUM
react = UnoscillatedReactorSpectrum(inputs_json)
flux_v = react.reactor_model_std(E, std_vogel)
flux_hm = react.reactor_model_std(E, std_hm)
flux_dyb = react.reactor_model_dyb(E, dyb_input)

iso_spectrum_v = react.reactor_spectrum(E, std_vogel)
iso_spectrum_hm = react.reactor_spectrum(E, std_hm)
iso_spectrum_dyb = react.reactor_spectrum(E, dyb_input)

reactor_flux_v = react.reactor_flux(E, std_vogel)
reactor_flux_hm = react.reactor_flux(E, std_hm)
reactor_flux_dyb = react.reactor_flux(E, dyb_input)

reactor_spectrum_v = react.unoscillated_reactor_spectrum(E, std_vogel, which_xs='SV_CI')
reactor_spectrum_hm = react.unoscillated_reactor_spectrum(E, std_hm, which_xs='SV_CI')
reactor_spectrum_dyb = react.unoscillated_reactor_spectrum(E, dyb_input, which_xs='SV_CI')

# nominal = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', plot_this=False)
# nominal_snf = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', bool_snf=True, plot_this=False)
# nominal_noneq = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', bool_noneq=True, plot_this=False)
# nominal_snf_noneq = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='HM', bool_snf=True, bool_noneq=True, plot_this=False)
# nominal_hm = react.antinu_spectrum_no_osc(E, which_xs='SV_CI', which_isospectrum='DYB', plot_this=False)

xsec_sv = react.eval_xs(E, which_xs="SV_CI", bool_protons=False)

u235 = react.eval_235u(E, which_input='HM_parametric')
pu239 = react.eval_239pu(E, which_input='HM_parametric')
u238 = react.eval_238u(E, which_input='HM_parametric')
pu241 = react.eval_241pu(E, which_input='HM_parametric')

dyb_correction = react.get_dybfluxbump_ratio(E)
hm_corrected = reactor_spectrum_hm * dyb_correction

ylabel0 = r'isotopic spectrum [$\textrm{N}_{\nu}$/fission/MeV]'
ylabel1 = r'reactor spectrum [$\textrm{N}_{\nu}$/s/MeV]'
ylabel2 = r'reactor flux [$\textrm{N}_{\nu}$/s/MeV/cm$^2$]'
ylabel3 = r'$S_{\bar{\nu}}$ [N$_{\nu}$/MeV/s]'

ax = plot_function(x_=[E, E, E, E, E],# E, E],
                   y_=[flux_hm, react.get_f235u() * u235, react.get_f239pu() * pu239, react.get_f238u() * u238,
                       react.get_f241pu() * pu241], # xsec_sv*2.e8, reactor_spectrum_hm*2500.],
                   label_=[r'total', r'$^{235}$U', r'$^{239}$Pu', r'$^{238}$U', r'$^{241}$Pu',
                           r'$\sigma_{\textrm{{\small{IBD}}}}$', r'$\bar{\nu}$ spectrum'],
                   styles=['m', 'b--', 'r-.', 'g:', 'y', 'c', 'k'],
                   ylabel_=r'$S^{\textrm{{\small{iso}}}}(E_{\nu})$ [N$_{\nu}$/MeV/fission]',
                   xlim=None, ylim=None, y_sci=True)
ax.get_legend().remove()
ax1 = ax.twinx()
ax1.set_ylabel(r'$\sigma_{\textrm{{\small{IBD}}}}$ [cm$^2$]', color='c')
ax1.set_ylim(-0.25e-42, 6.e-42)
ax1.plot(E, xsec_sv, 'c-', linewidth=1)
ax1.tick_params(axis='y', labelcolor='c')
# # ax2 = ax.twinx()
ax1.plot(E, xsec_sv*flux_hm*30., 'k-', linewidth=1)

plot_function(x_=[E, E, E], y_=[flux_v, flux_hm, flux_dyb],
              label_=[r'Vogel', r'HM', r'DYB'], styles=['g-', 'r--', 'k'],
              ylabel_=ylabel0, xlim=None, ylim=None, y_sci=True)

plot_function(x_=[E, E, E], y_=[iso_spectrum_v, iso_spectrum_hm, iso_spectrum_dyb],
              label_=[r'Vogel', r'HM', r'DYB'], styles=['g-', 'r--', 'k'],
              ylabel_=ylabel1, xlim=None, ylim=None, y_sci=True)

plot_function(x_=[E, E, E], y_=[reactor_flux_v, reactor_flux_hm, reactor_flux_dyb],
              label_=[r'Vogel', r'HM', r'DYB'], styles=['g-', 'r--', 'k'],
              ylabel_=ylabel2, xlim=None, ylim=None, y_sci=True)

plot_function(x_=[E, E, E], y_=[reactor_spectrum_v, reactor_spectrum_hm, reactor_spectrum_dyb],
              label_=[r'Vogel', r'HM', r'DYB'], styles=['g-', 'r--', 'k'],
              ylabel_=ylabel3, xlim=None, ylim=None, y_sci=True)

# plot_function(x_=[E, E, E, E], y_=[nominal, nominal_snf, nominal_noneq, nominal_snf_noneq],
#               label_=[r'nominal', r'SNF', r'NonEq', r'SNF+NonEq'], styles=['k', 'b:', 'r-', 'g--'],
#               ylabel_=ylabel, xlim=None, ylim=None)
#
# plot_function(x_=[E, E, E, E],
#               y_=[nominal, (nominal_snf-nominal)/nominal*100., (nominal_noneq-nominal)/nominal*100.,
#                   (nominal_snf_noneq-nominal)/nominal*100.],
#               label_=[r'nominal', r'SNF', r'NonEq', r'SNF+NonEq'], styles=['k', 'b:', 'r-', 'g--'],
#               ylabel_=r'(corr - nom)/nom [\%]', xlim=None, ylim=None)

plot_function(x_=[E, E, E], y_=[reactor_spectrum_hm, hm_corrected, reactor_spectrum_dyb],
              label_=[r'HM', r'HM - corrected', r'DYB'], styles=['g:', 'b--', 'k'],
              ylabel_=ylabel3, xlim=None, ylim=None, y_sci=True)


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
