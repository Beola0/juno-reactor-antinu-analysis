import numpy as np
import matplotlib.pyplot as plt
import time
import json
import latex
from antinu_spectrum.plot import plot_function
from antinu_spectrum.spectrum import OscillatedSpectrum

std_hm = {
    '235U': 'Huber',
    '238U': 'Mueller',
    '239Pu': 'Huber',
    '241Pu': 'Huber'
}

### MAIN ###
time_start = time.perf_counter_ns()

f = open('data/nufit_inputs.json')
# f = open('data/nominal_inputs.json')
# f = open('data/YB_inputs.json')
inputs_json = json.load(f)

E = np.arange(1.806, 10.01, 0.01)  # in MeV
# E = np.arange(1., 12.01, 0.01)  # in MeV
# E = np.arange(1.925, 8.65, 0.01)  # in MeV (for DYB)

spectrum = OscillatedSpectrum(inputs_json)
s_un = spectrum.unoscillated_reactor_spectrum(E, which_xs='SV_approx', which_inputs_=std_hm, bool_snf=False,
                                              bool_noneq=False, pu_combo=True)

s_N = s_un * spectrum.eval_matter_prob_no(E)
s_I = s_un * spectrum.eval_matter_prob_io(E)

# spectrum.set_b(0.)
# spectrum.set_a(2.5/100.)
s_N_resol = spectrum.gaussian_smearing_abc(s_N, E, E-0.78)
# spectrum.set_a(2.75/100.)
# s_N_resol2 = spectrum.gaussian_smearing_abc(s_N, E, E-0.78)
# spectrum.set_a(3/100.)
# s_N_resol3 = spectrum.gaussian_smearing_abc(s_N, E, E-0.78)


ylabel = r'$N(\bar{\nu})$ [a.u.]'
plot_function(x_=[E, E], y_=[s_N, s_I],
              label_=[r'NO', r'IO'], styles=['b-', 'r-'],
              ylabel_=ylabel,
              xlabel_=r'$E_{\nu}$ [MeV]', xlim=None, ylim=None, y_sci=True)

plot_function(x_=[E, E-0.78],#, E-0.78, E-0.78],
              y_=[s_N, s_N_resol],#, s_N_resol2, s_N_resol3],
              label_=[r'NO', r'NO with Eres'], styles=['k-', 'r-'],
              ylabel_=ylabel,
              xlabel_=r'$E$ [MeV]', xlim=[1.5,9.5], ylim=[0, 1.85e-4], y_sci=True)


### OSCILLATED SPECTRUM
# spectrum = OscillatedSpectrum(inputs_json)
# s_N, s_I = spectrum.osc_spectrum(E, plot_this=True, which_inputs_=std_hm, which_xsec='SV_approx', plot_un=False, runtime=True, matter=False)
# s_N_m, s_I_m = spectrum.osc_spectrum(E, plot_this=True, which_inputs_=std_hm, which_xsec='SV_approx', plot_un=False, runtime=True, matter=True)
#
# sres_N, sres_I = spectrum.resol_spectrum(E-0.78, which_inputs_=std_hm, which_xsec='SV_approx', matter=False, plot_this=False, runtime=True)
# sres_N_m, sres_I_m = spectrum.resol_spectrum(E-0.78, which_inputs_=std_hm, which_xsec='SV_approx', matter=True, plot_this=False, runtime=True)

# s_N_v = spectrum.osc_spectrum_no(E, plot_this=False, which_inputs_=std_hm, which_xsec='SV_approx', matter=True)
# s_N_hm = spectrum.osc_spectrum_no(E, plot_this=False, which_inputs_=std_hm, which_xsec='SV_approx', matter=True)
# s_N_dyb = spectrum.osc_spectrum_no(E, plot_this=False, which_inputs_=std_hm, which_xsec='SV_approx', matter=True)
# s_N_v_res = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_inputs_=std_hm, which_xsec='SV_approx', matter=True)
# s_N_hm_res = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_inputs_=std_hm, which_xsec='SV_approx', matter=True)
# s_N_dyb_res = spectrum.resol_spectrum_no(E-0.78, plot_this=False, which_inputs_=std_hm, which_xsec='SV_approx', matter=True)

# ylabel = r'$S(\bar{\nu})$ [N$_{\nu}$/s/MeV]'
# plot_function(x_=[E, E], y_=[s_N, s_N_m],
#               label_=[r'NO - vacuum', r'NO - matter'], styles=['k-', 'g--'],
#               ylabel_=ylabel,
#               xlabel_=r'$E_{\nu}$ [MeV]', xlim=None, ylim=None, y_sci=True)
#
# plot_function(x_=[E-0.78, E-0.78], y_=[sres_N, sres_N_m],
#               label_=[r'NO - vacuum', r'NO - matter'], styles=['k-', 'g--'],
#               ylabel_=ylabel,
#               xlabel_=r'$E_{\textrm{{\small{vis}}}}$ [MeV]', xlim=None, ylim=None, y_sci=True)

# plot_function(x_=[E, E, E], y_=[s_N_dyb, s_N_hm, s_N_v],
#               label_=[r'NO - DYB', r'NO - HM', r'NO - V'], styles=['k', 'r--', 'g-.'],
#               ylabel_=ylabel,
#               xlabel_=r'$E_{\nu}$ [MeV]', xlim=None, ylim=None)
#
# plot_function(x_=[E-0.78, E-0.78, E-0.78], y_=[s_N_dyb_res, s_N_hm_res, s_N_v_res],
#               label_=[r'NO - DYB', r'NO - HM', r'NO - V'], styles=['k', 'r--', 'g-.'],
#               ylabel_=ylabel,
#               xlabel_=r'$E_{\textrm{{\small{vis}}}}$ [MeV]', xlim=None, ylim=None)

# aa = spectrum.osc_spectrum_no(E, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=True,
#                               plot_this=False, plot_un=False, plot_singles=True, runtime=False)
# cc = spectrum.resol_spectrum_no(E-0.78, which_isospectrum='DYB', bool_snf=False, bool_noneq=False, matter=True,
#                                 plot_this=False, plot_singles=True, runtime=False)

elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
