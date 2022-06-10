import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.ticker as plticker
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import json
import math
import pandas as pd
from scipy import integrate, stats, interpolate
import latex
from antinu_spectrum.plot import plot_function, plot_function_residual
from antinu_spectrum.reactor import UnoscillatedReactorSpectrum
# from antinu_spectrum.oscillation import OscillationProbability


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

f235_dyb = 0.564
f239_dyb = 0.304
f238_dyb = 0.076
f241_dyb = 0.056

react = UnoscillatedReactorSpectrum(inputs_json)
const = 6.241509e21
w_th = 4.6  # GW - Taishan core
L = 52.77e5  # TS core - baseline in cm
Np = react.eval_n_protons()
eff = 0.822
BT_conversion = 20/540. * 1000
BT_conversion_sec = 20/540./86400 * 1000

time_start = time.perf_counter_ns()
react.verbose = False


########################################################################################################################
# fission fractions vs burnup
########################################################################################################################
T_burnup = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]  # as burnup, in MW d/ton_U
T_days = [x/BT_conversion for x in T_burnup]
T_sec = [x*86400 for x in T_days]

fiss_f = pd.read_csv(path_ff+"/1cycle_fission_fractions_CPC2017.csv", sep=',', skiprows=1, header=None,
                     names=['burnup', 'days', 'f_235', 'f_239', 'f_238', 'f_241'])
fiss_f['sum'] = fiss_f.iloc[:, 2:6].apply(np.sum, axis=1)
for iso_ in ['f_235', 'f_239', 'f_238', 'f_241']:
    fiss_f[iso_] = fiss_f[iso_] / fiss_f['sum']

f_235 = interpolate.interp1d(fiss_f['burnup'], fiss_f['f_235'])
f_239 = interpolate.interp1d(fiss_f['burnup'], fiss_f['f_239'])
f_238 = interpolate.interp1d(fiss_f['burnup'], fiss_f['f_238'])
f_241 = interpolate.interp1d(fiss_f['burnup'], fiss_f['f_241'])

average_235 = np.zeros(len(T_burnup))
average_239 = np.zeros(len(T_burnup))
average_238 = np.zeros(len(T_burnup))
average_241 = np.zeros(len(T_burnup))
for t_ in np.arange(len(T_burnup)):
    x = np.arange(0., T_burnup[t_], 5)
    average_235[t_] = integrate.simpson(f_235(x), x) / T_burnup[t_]
    average_239[t_] = integrate.simpson(f_239(x), x) / T_burnup[t_]
    average_238[t_] = integrate.simpson(f_238(x), x) / T_burnup[t_]
    average_241[t_] = integrate.simpson(f_241(x), x) / T_burnup[t_]

avg_energy = average_235*e_235 + average_239*e_239 + average_238*e_238 + average_241*e_241

data = {'T_burnup': T_burnup, 'T_day': T_days, 'avg_235': average_235, 'avg_239': average_239, 'avg_238': average_238,
        'avg_241': average_241, 'avg_eperfission': avg_energy}
average_ff = pd.DataFrame(data)

########################################################################################################################
# spectrum with mean fission fractions
########################################################################################################################
plot = True

energy = np.arange(1.875, 9.005, 0.005)
xs = react.eval_xs(energy, which_xs="SV_approx", bool_protons=False)

xsf = np.zeros(len(average_ff.index))
xsf_hm = np.zeros(len(average_ff.index))
spectra = []
spectra_t = []
spectra_hm = []
for i_ in np.arange(len(average_ff)):
    react.set_fission_fractions(average_ff['avg_235'].iloc[i_], average_ff['avg_239'].iloc[i_],
                                average_ff['avg_238'].iloc[i_], average_ff['avg_241'].iloc[i_])
    flux = react.reactor_model_dyb(energy, input_dyb)
    flux_hm = react.reactor_model_std(energy, std_hm)
    spectra.append(flux*xs)
    spectra_t.append(Np*eff/(4*math.pi*L*L)*w_th*const/average_ff['avg_eperfission'].iloc[i_]*flux*xs*T_sec[i_])
    spectra_hm.append(flux_hm * xs)
    xsf[i_] = integrate.simps(flux*xs, energy)
    xsf_hm[i_] = integrate.simps(flux_hm * xs, energy)

if plot:
    labels = []
    for i_ in np.arange(len(average_ff)):
        labels.append(r'T$_{\textrm{{\small{DAQ}}}}$ = %.f d' % average_ff['T_day'].iloc[i_])
    styles = ['navy', 'blue', 'blueviolet', 'magenta', 'hotpink', 'coral', 'red', 'brown']

    ax = plot_function_residual(
        x_=[energy, energy, energy, energy, energy, energy, energy, energy], y_=spectra, styles=styles, label_=labels,
        ylabel_=r'spectrum [cm$^2$/MeV/fission]', #ylim=[-0.05e-43, 2.05e-43], ylim2=[-20, 0.25]
    )
    ax[0].text(0.1, 0.08, r'DYB model', transform=ax[0].transAxes)
    ax[1].get_legend().remove()

    ax = plot_function(
        x_=[energy, energy, energy, energy, energy, energy, energy, energy], y_=spectra_t, styles=styles, label_=labels,
        ylabel_=r'spectrum [N$_{\nu}$/MeV]', y_sci=True #ylim=[-0.05e-43, 2.05e-43], ylim2=[-20, 0.25]
    )
    # ax.text(0.15, 0.06, r'DYB model', transform=ax.transAxes)
    ax.text(0.5, 0.85, r'\noindent average\\spectra', transform=ax.transAxes)
    # ax[1].get_legend().remove()

########################################################################################################################
# integrated spectrum
########################################################################################################################


def mean_energy_per_fission(t_daq):

    return f_235(t_daq)*e_235 + f_239(t_daq)*e_239 + f_238(t_daq)*e_238 + f_241(t_daq)*e_241


# def eval_spectrum_time(energy, t_daq)


plot = True

energy = np.arange(1.875, 9.005, 0.005)
xs = react.eval_xs(energy, which_xs="SV_approx", bool_protons=False)

int_spectra = []
s_total = react.eval_total(energy, which_input=input_dyb['total'])
s_235u = react.eval_235u(energy, which_input=input_dyb['235U'])
s_239pu = react.eval_239pu(energy, which_input=input_dyb['239Pu'])
s_238u = react.eval_238u(energy, which_input=input_dyb['238U'])
s_241pu = react.eval_241pu(energy, which_input=input_dyb['241Pu'])
for t_ in np.arange(len(average_ff)):

    tt = np.arange(0., T_burnup[t_], 5)
    s_appo = np.zeros((len(tt), len(energy)))
    # s_appo2 = np.zeros((len(tt), len(energy)))
    s_final = np.zeros((len(energy)))

    for tt_ in np.arange(len(tt)):
        df_235 = f_235(tt[tt_]) - f235_dyb
        df_239 = f_239(tt[tt_]) - f239_dyb
        df_238 = f_238(tt[tt_]) - f238_dyb
        df_241 = f_241(tt[tt_]) - f241_dyb
        appo = (s_total + s_235u*df_235 + s_239pu*df_239 + s_238u*df_238 + s_241pu*(df_241 - 0.183 * df_239))*xs
        # s_appo[tt_] = appo
        s_appo[tt_] = appo / mean_energy_per_fission(tt[tt_])

    for ee_ in np.arange(len(energy)):
        s_final[ee_] = integrate.simps(s_appo[:, ee_], tt)

    int_spectra.append(Np * eff / (4*math.pi*L*L) * w_th * const * s_final / BT_conversion_sec)

if plot:
    labels = []
    for i_ in np.arange(len(average_ff)):
        labels.append(r'T$_{\textrm{{\small{DAQ}}}}$ = %.f d' % average_ff['T_day'].iloc[i_])
    styles = ['navy', 'blue', 'blueviolet', 'magenta', 'hotpink', 'coral', 'red', 'brown']

    ax = plot_function(
        x_=[energy, energy, energy, energy, energy, energy, energy, energy], y_=int_spectra, styles=styles, label_=labels,
        ylabel_=r'spectrum [N$_{\nu}$/MeV]', y_sci=True #ylim=[-0.05e-43, 2.05e-43], ylim2=[-20, 0.25]
    )
    ax.text(0.5, 0.85, r'\noindent integrated\\spectra', transform=ax.transAxes)
    # ax[1].get_legend().remove()

########################################################################################################################
# comparison
########################################################################################################################

labels = []
for i_ in np.arange(len(average_ff)):
    labels.append(r'T$_{\textrm{{\small{DAQ}}}}$ = %.f d' % average_ff['T_day'].iloc[i_])
styles = ['navy', 'blue', 'blueviolet', 'magenta', 'hotpink', 'coral', 'red', 'brown']

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=[12, 9], sharex=True, sharey='row', constrained_layout=True,
                         gridspec_kw={'height_ratios': [2, 1, 2, 1]})
# fig_diff, axes_diff = plt.subplots(nrows=2, ncols=4, figsize=[12, 9], sharex=True, sharey=True, constrained_layout=False)

ylim = [-100, 4100]
ylim_diff = [-0.01, 0.25]

for i_, ax in enumerate(axes[0].flat):
    ax.plot(energy, int_spectra[i_], color=styles[i_], linestyle='-')
    ax.plot(energy, spectra_t[i_], color=styles[i_], linestyle='--')
    ax.set_title(labels[i_])
    ax.set_ylim(ylim)
    ax.grid(alpha=0.65)

for i_, ax in enumerate(axes[1].flat):
    ax.plot(energy, (spectra_t[i_]-int_spectra[i_])/int_spectra[i_]*100., color=styles[i_])
    ax.set_ylim(ylim_diff)
    ax.grid(alpha=0.65)

for i_, ax in enumerate(axes[2].flat):
    ax.plot(energy, int_spectra[i_+4], color=styles[i_+4], linestyle='-')
    ax.plot(energy, spectra_t[i_+4], color=styles[i_+4], linestyle='--')
    ax.set_title(labels[i_+4])
    ax.set_ylim(ylim)
    ax.grid(alpha=0.65)

for i_, ax in enumerate(axes[3].flat):
    ax.plot(energy, (spectra_t[i_+4]-int_spectra[i_+4])/int_spectra[i_+4]*100., color=styles[i_+4])
    ax.set_ylim(ylim_diff)
    ax.grid(alpha=0.65)


axes[0, 0].set_ylabel(r'spectrum [N$_{\nu}$/MeV]')
axes[1, 0].set_ylabel(r'diff [\%]')
axes[2, 0].set_ylabel(r'spectrum [N$_{\nu}$/MeV]')
axes[3, 0].set_ylabel(r'diff [\%]')
for ax in axes[1]:
    ax.set_xlabel(r'$E_{\nu}$ [MeV]')
for ax in axes[3]:
    ax.set_xlabel(r'$E_{\nu}$ [MeV]')

print(r'\nNumber of events')
print(r'T_daq [d]; integrate spectrum; average spectrum; relative diff [%]')
for t_ in np.arange(len(average_ff)):
    ibd_int = integrate.simpson(int_spectra[t_], energy)
    ibd_avg = integrate.simpson(spectra_t[t_], energy)
    print(r'%.0f; %.2f IBDs; %.2f IBDs; %.2f '
          % (average_ff['T_day'].iloc[t_], ibd_int, ibd_avg, (ibd_avg-ibd_int)/ibd_int*100.))


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
