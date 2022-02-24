import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import json
import os
import sys
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

path = '/Users/beatricejelmini/Desktop/JUNO/JUNO_codes/JUNO_ReactorNeutrinosAnalysis/Inputs/spectra'

f = open('Inputs/nominal_inputs.json')
inputs_json = json.load(f)
react = UnoscillatedReactorSpectrum(inputs_json)


########################################################################################################################
# Huber 241Pu - covariance matrix
########################################################################################################################
huber_241pu = react.get_241pu_huber()
plot = True

xs = react.eval_xs(huber_241pu.index, bool_protons=False, which_xs='SV_CI')

N_h = len(huber_241pu.index)
stat_uncertainty = np.maximum(abs(huber_241pu["neg_stat"]), huber_241pu["pos_stat"])
bias_uncertainty = np.maximum(abs(huber_241pu["neg_bias"]), huber_241pu["pos_bias"])
z_uncertainty = np.maximum(abs(huber_241pu["neg_z"]), huber_241pu["pos_z"])
wm_uncertainty = np.maximum(abs(huber_241pu["neg_wm"]), huber_241pu["pos_wm"])
norm_uncertainty = np.maximum(abs(huber_241pu["neg_norm"]), huber_241pu["pos_norm"])

stat_cov = np.zeros((N_h, N_h))
stat_cov_xs = np.zeros((N_h, N_h))
stat_cov_rel = np.zeros((N_h, N_h))
for i_ in np.arange(N_h):
    stat_cov[i_][i_] = stat_uncertainty.iloc[i_] * stat_uncertainty.iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_]
    stat_cov_xs[i_][i_] = stat_uncertainty.iloc[i_] * stat_uncertainty.iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_] * xs[i_] * xs[i_]
    stat_cov_rel[i_][i_] = stat_uncertainty.iloc[i_] * stat_uncertainty.iloc[i_]

bias_cov = np.zeros((N_h, N_h))
bias_cov_xs = np.zeros((N_h, N_h))
bias_cov_rel = np.zeros((N_h, N_h))
for i_ in np.arange(N_h):
    bias_cov[i_][i_] = bias_uncertainty.iloc[i_] * bias_uncertainty.iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_]
    bias_cov_xs[i_][i_] = bias_uncertainty.iloc[i_] * bias_uncertainty.iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_] * xs[i_] * xs[i_]
    bias_cov_rel[i_][i_] = bias_uncertainty.iloc[i_] * bias_uncertainty.iloc[i_]

z_cov = np.zeros((N_h, N_h))
z_cov_xs = np.zeros((N_h, N_h))
z_cov_rel = np.zeros((N_h, N_h))
for i_ in np.arange(N_h):
    for j_ in np.arange(N_h):
        z_cov[i_][j_] = z_uncertainty.iloc[i_] * z_uncertainty.iloc[j_] * \
                        huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[j_]
        z_cov_xs[i_][j_] = z_uncertainty.iloc[i_] * z_uncertainty.iloc[j_] * \
                        huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[j_] * xs[i_] * xs[j_]
        z_cov_rel[i_][j_] = z_uncertainty.iloc[i_] * z_uncertainty.iloc[j_]

wm_cov = np.zeros((N_h, N_h))
wm_cov_xs = np.zeros((N_h, N_h))
wm_cov_rel = np.zeros((N_h, N_h))
for i_ in np.arange(N_h):
    for j_ in np.arange(N_h):
        wm_cov[i_][j_] = wm_uncertainty.iloc[i_] * wm_uncertainty.iloc[j_] * \
                         huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[j_]
        wm_cov_xs[i_][j_] = wm_uncertainty.iloc[i_] * wm_uncertainty.iloc[j_] * \
                         huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[j_] * xs[i_] * xs[j_]
        wm_cov_rel[i_][j_] = wm_uncertainty.iloc[i_] * wm_uncertainty.iloc[j_]

norm_cov = np.zeros((N_h, N_h))
norm_cov_xs = np.zeros((N_h, N_h))
norm_cov_rel = np.zeros((N_h, N_h))
for i_ in np.arange(N_h):
    for j_ in np.arange(N_h):
        norm_cov[i_][j_] = norm_uncertainty.iloc[i_] * norm_uncertainty.iloc[j_] * \
                           huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[j_]
        norm_cov_xs[i_][j_] = norm_uncertainty.iloc[i_] * norm_uncertainty.iloc[j_] * \
                           huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[j_] * xs[i_] * xs[j_]
        norm_cov_rel[i_][j_] = norm_uncertainty.iloc[i_] * norm_uncertainty.iloc[j_]

total_241pu = stat_cov + bias_cov + z_cov + wm_cov + norm_cov
total_241pu_xs = stat_cov_xs + bias_cov_xs + z_cov_xs + wm_cov_xs + norm_cov_xs
total_241pu_rel = stat_cov_rel + bias_cov_rel + z_cov_rel + wm_cov_rel + norm_cov_rel

if plot:
    matrices = [stat_cov, bias_cov, z_cov, wm_cov, norm_cov, total_241pu]
    matrices_xs = [stat_cov_xs, bias_cov_xs, z_cov_xs, wm_cov_xs, norm_cov_xs, total_241pu_xs]
    matrices_rel = [stat_cov_rel, bias_cov_rel, z_cov_rel, wm_cov_rel, norm_cov_rel, total_241pu_rel]
    titles = ["stat", "bias", "Z", "WM", "norm", "total"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(0, total_241pu.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    for i_, ax in enumerate(axes.flat):
        ax.imshow(matrices[i_], origin='lower', cmap=cmap, norm=normalizer)
        ax.set_title(titles[i_])
        if i_ == 3 or i_ == 4 or i_ == 5:
            ax.set_xlabel("bin")
        if i_ == 0 or i_ == 3:
            ax.set_ylabel("bin")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        label=Pu1+r' covariance [(N$_{\nu}$/\si{MeV}/fission)$^2$]')
    cbar.formatter.set_powerlimits((0, 0))

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(0, total_241pu_xs.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    for i_, ax in enumerate(axes.flat):
        ax.imshow(matrices_xs[i_], origin='lower', cmap=cmap, norm=normalizer)
        ax.set_title(titles[i_])
        if i_ == 3 or i_ == 4 or i_ == 5:
            ax.set_xlabel("bin")
        if i_ == 0 or i_ == 3:
            ax.set_ylabel("bin")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        label=Pu1+r' covariance [(\si{\centi\meter\squared}/\si{MeV}/fission)$^2$]')
    cbar.formatter.set_powerlimits((0, 0))

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(0, total_241pu_rel.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    for i_, ax in enumerate(axes.flat):
        ax.imshow(matrices_rel[i_], origin='lower', cmap=cmap, norm=normalizer)
        ax.set_title(titles[i_])
        if i_ == 3 or i_ == 4 or i_ == 5:
            ax.set_xlabel("bin")
        if i_ == 0 or i_ == 3:
            ax.set_ylabel("bin")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=Pu1+r' relative uncert. "[\%$^2$]"')
    cbar.formatter.set_powerlimits((0, 0))


########################################################################################################################
# Mueller 238U - covariance matrix
########################################################################################################################
mueller_238u = react.get_238u_mueller()
plot = True

xs = react.eval_xs(mueller_238u.index, bool_protons=False, which_xs='SV_CI')

N_m = len(mueller_238u.index)
nucleardb_cov = np.zeros((N_m, N_m))
nucleardb_cov_xs = np.zeros((N_m, N_m))
nucleardb_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    nucleardb_cov[i_][i_] = mueller_238u["nuclear_db"].iloc[i_] * mueller_238u["nuclear_db"].iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_]
    nucleardb_cov_xs[i_][i_] = mueller_238u["nuclear_db"].iloc[i_] * mueller_238u["nuclear_db"].iloc[i_] * \
                            huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_] * xs[i_] * xs[i_]
    nucleardb_cov_rel[i_][i_] = mueller_238u["nuclear_db"].iloc[i_] * mueller_238u["nuclear_db"].iloc[i_]

forbid_cov = np.zeros((N_m, N_m))
forbid_cov_xs = np.zeros((N_m, N_m))
forbid_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    forbid_cov[i_][i_] = mueller_238u["forbidden_treatment"].iloc[i_] * mueller_238u["forbidden_treatment"].iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_]
    forbid_cov_xs[i_][i_] = mueller_238u["forbidden_treatment"].iloc[i_] * mueller_238u["forbidden_treatment"].iloc[i_] * \
                         huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_] * xs[i_] * xs[i_]
    forbid_cov_rel[i_][i_] = mueller_238u["forbidden_treatment"].iloc[i_] * mueller_238u["forbidden_treatment"].iloc[i_]

corr_cov = np.zeros((N_m, N_m))
corr_cov_xs = np.zeros((N_m, N_m))
corr_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    corr_cov[i_][i_] = mueller_238u["corrections"].iloc[i_] * mueller_238u["corrections"].iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_]
    corr_cov_xs[i_][i_] = mueller_238u["corrections"].iloc[i_] * mueller_238u["corrections"].iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_] * xs[i_] * xs[i_]
    corr_cov_rel[i_][i_] = mueller_238u["corrections"].iloc[i_] * mueller_238u["corrections"].iloc[i_]

missing_cov = np.zeros((N_m, N_m))
missing_cov_xs = np.zeros((N_m, N_m))
missing_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    missing_cov[i_][i_] = mueller_238u["missing_info"].iloc[i_] * mueller_238u["missing_info"].iloc[i_] * \
                       huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_]
    missing_cov_xs[i_][i_] = mueller_238u["missing_info"].iloc[i_] * mueller_238u["missing_info"].iloc[i_] * \
                          huber_241pu["spectrum"].iloc[i_] * huber_241pu["spectrum"].iloc[i_] * xs[i_] * xs[i_]
    missing_cov_rel[i_][i_] = mueller_238u["missing_info"].iloc[i_] * mueller_238u["missing_info"].iloc[i_]

total_238u = nucleardb_cov + forbid_cov + corr_cov + missing_cov
total_238u_xs = nucleardb_cov_xs + forbid_cov_xs + corr_cov_xs + missing_cov_xs
total_238u_rel = nucleardb_cov_rel + forbid_cov_rel + corr_cov_rel + missing_cov_rel

if plot:
    matrices = [nucleardb_cov, forbid_cov, corr_cov, missing_cov, total_238u]
    matrices_xs = [nucleardb_cov_xs, forbid_cov_xs, corr_cov_xs, missing_cov_xs, total_238u_xs]
    matrices_rel = [nucleardb_cov_rel, forbid_cov_rel, corr_cov_rel, missing_cov_rel, total_238u_rel]
    titles = ["nuclear db", "forbidden", "corrections", "missing info", "total"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(0, total_238u.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    for i_, ax in enumerate(axes.flat):
        try:
            ax.imshow(matrices[i_], origin='lower', cmap=cmap, norm=normalizer)
        except IndexError:
            continue
        ax.set_title(titles[i_])
        if i_ == 3 or i_ == 4 or i_ == 5:
            ax.set_xlabel("bin")
        if i_ == 0 or i_ == 3:
            ax.set_ylabel("bin")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=U8+r' covariance [(N$_{\nu}$/\si{MeV}/fission)$^2$]')
    cbar.formatter.set_powerlimits((0, 0))
    axes[-1, -1].axis('off')

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(0, total_238u_xs.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    for i_, ax in enumerate(axes.flat):
        try:
            ax.imshow(matrices_xs[i_], origin='lower', cmap=cmap, norm=normalizer)
        except IndexError:
            continue
        ax.set_title(titles[i_])
        if i_ == 3 or i_ == 4 or i_ == 5:
            ax.set_xlabel("bin")
        if i_ == 0 or i_ == 3:
            ax.set_ylabel("bin")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=U8+r' covariance [(\si{\centi\meter\squared}/\si{MeV}/fission)$^2$]')
    cbar.formatter.set_powerlimits((0, 0))
    axes[-1, -1].axis('off')

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(0, total_238u_rel.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    for i_, ax in enumerate(axes.flat):
        try:
            ax.imshow(matrices_rel[i_], origin='lower', cmap=cmap, norm=normalizer)
        except IndexError:
            continue
        ax.set_title(titles[i_])
        if i_ == 3 or i_ == 4 or i_ == 5:
            ax.set_xlabel("bin")
        if i_ == 0 or i_ == 3:
            ax.set_ylabel("bin")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=U8+r' relative uncert. "[\%$^2$]"')
    cbar.formatter.set_powerlimits((0, 0))
    axes[-1, -1].axis('off')

########################################################################################################################
# DYB/DYB+PP - covariance matrix from unfolding
########################################################################################################################
covariance_dyb = np.loadtxt(path+'/CovMatrix_unfolding_DYB.txt')
N_dyb = 75
corr_dyb = np.zeros((N_dyb, N_dyb))
for i in np.arange(0, N_dyb, 1):
    for j in np.arange(0, N_dyb, 1):
        corr_dyb[i, j] = covariance_dyb[i, j]/np.sqrt(covariance_dyb[i, i])/np.sqrt(covariance_dyb[j, j])

full_covariance_dyb_pp = np.loadtxt(path+'/CovMatrix_unfolding_DYB_PROSPECT.txt')
full_covariance_dyb_pp = full_covariance_dyb_pp*1.e-86
N = 50
corr_dyb_pp = np.zeros((N, N))
for i in np.arange(0, N, 1):
    for j in np.arange(0, N, 1):
        corr_dyb_pp[i, j] = full_covariance_dyb_pp[i, j]/np.sqrt(full_covariance_dyb_pp[i, i])/np.sqrt(full_covariance_dyb_pp[j, j])


plt.figure()
plt.imshow(full_covariance_dyb_pp, origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(corr_dyb_pp, origin='lower', vmin=-1, vmax=1)
plt.colorbar()

plt.figure()
plt.imshow(covariance_dyb, origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(corr_dyb, origin='lower', vmin=-1, vmax=1)
plt.colorbar()

########################################################################################################################
# DYB/DYB+PP - covariance matrix from unfolding
########################################################################################################################
total = np.block([
    [covariance_dyb, np.zeros((N_dyb, N_m+N_h))],
    [np.zeros((N_m, N_dyb)), total_238u_xs, np.zeros((N_m, N_h))],
    [np.zeros((N_h, N_dyb+N_m)), total_241pu_xs]
])

plt.figure(constrained_layout=True)
plt.imshow(total, origin='lower')
plt.colorbar(label=r'covariance [(\si{\centi\meter\squared}/\si{MeV}/fission)$^2$]')

# ax.pcolormesh

plt.ion()
plt.show()
