import numpy as np
from scipy import interpolate, stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import time
import json
import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')
import latex
from plot import plot_function, plot_function_residual, plot_matrix
from reactor import UnoscillatedReactorSpectrum
from numpy import linalg


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
time_start = time.perf_counter_ns()


########################################################################################################################
# Huber 241Pu - covariance matrix
########################################################################################################################
huber_241pu = react.get_241pu_huber()
plot = False

xs = react.eval_xs(huber_241pu.index, bool_protons=False, which_xs='SV_approx')

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
        ax.set_title(titles[i_], fontsize=16)
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

    plot_matrix(
        m_=total_241pu_xs, xlabel_='bin', ylabel_='bin', title_=Pu1+r' total covariance',
        label_=r'Covariance [(\si{\centi\meter\squared}/\si{MeV}/fission)$^2$]', figsize_=[6.1, 5.],
        cmap_='nipy_spectral'
    )

    plot_function(
        x_=[huber_241pu.index], y_=[huber_241pu['spectrum']], label_=[r'H '+Pu1],
        ylabel_=r'$S_{241}$ [N$_{\nu}$/\si{MeV}/fission]', styles=['bo']
    )
    plot_function(
        x_=[huber_241pu.index], y_=[huber_241pu['spectrum']*xs], label_=[r'H '+Pu1],
        ylabel_=r'$S_{241}$ [\si{\centi\meter\squared}/\si{MeV}/fission]', styles=['bo']
    )
    plot_function(
        x_=[huber_241pu.index], y_=[np.sqrt(np.diagonal(total_241pu_xs))/(huber_241pu['spectrum']*xs)], label_=[r'H '+Pu1],
        ylabel_=r'$S_{241}$ relative uncert.', styles=['bo']
    )


########################################################################################################################
# Mueller 238U - covariance matrix
########################################################################################################################
mueller_238u = react.get_238u_mueller()
# EF evaluated in Mueller bin centers
EF_238u = react.eval_238u(mueller_238u.index, which_input='EF')
plot = False

xs = react.eval_xs(mueller_238u.index, bool_protons=False, which_xs='SV_approx')

N_m = len(mueller_238u.index)
nucleardb_cov = np.zeros((N_m, N_m))
nucleardb_cov_xs = np.zeros((N_m, N_m))
nucleardb_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    nucleardb_cov[i_][i_] = mueller_238u["nuclear_db"].iloc[i_] * mueller_238u["nuclear_db"].iloc[i_] * \
                       EF_238u[i_] * EF_238u[i_]
    nucleardb_cov_xs[i_][i_] = mueller_238u["nuclear_db"].iloc[i_] * mueller_238u["nuclear_db"].iloc[i_] * \
                            EF_238u[i_] * EF_238u[i_] * xs[i_] * xs[i_]
    nucleardb_cov_rel[i_][i_] = mueller_238u["nuclear_db"].iloc[i_] * mueller_238u["nuclear_db"].iloc[i_]

forbid_cov = np.zeros((N_m, N_m))
forbid_cov_xs = np.zeros((N_m, N_m))
forbid_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    forbid_cov[i_][i_] = mueller_238u["forbidden_treatment"].iloc[i_] * mueller_238u["forbidden_treatment"].iloc[i_] * \
                       EF_238u[i_] * EF_238u[i_]
    forbid_cov_xs[i_][i_] = mueller_238u["forbidden_treatment"].iloc[i_] * mueller_238u["forbidden_treatment"].iloc[i_] * \
                         EF_238u[i_] * EF_238u[i_] * xs[i_] * xs[i_]
    forbid_cov_rel[i_][i_] = mueller_238u["forbidden_treatment"].iloc[i_] * mueller_238u["forbidden_treatment"].iloc[i_]

corr_cov = np.zeros((N_m, N_m))
corr_cov_xs = np.zeros((N_m, N_m))
corr_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    corr_cov[i_][i_] = mueller_238u["corrections"].iloc[i_] * mueller_238u["corrections"].iloc[i_] * \
                       EF_238u[i_] * EF_238u[i_]
    corr_cov_xs[i_][i_] = mueller_238u["corrections"].iloc[i_] * mueller_238u["corrections"].iloc[i_] * \
                       EF_238u[i_] * EF_238u[i_] * xs[i_] * xs[i_]
    corr_cov_rel[i_][i_] = mueller_238u["corrections"].iloc[i_] * mueller_238u["corrections"].iloc[i_]

missing_cov = np.zeros((N_m, N_m))
missing_cov_xs = np.zeros((N_m, N_m))
missing_cov_rel = np.zeros((N_m, N_m))
for i_ in np.arange(N_m):
    missing_cov[i_][i_] = mueller_238u["missing_info"].iloc[i_] * mueller_238u["missing_info"].iloc[i_] * \
                       EF_238u[i_] * EF_238u[i_]
    missing_cov_xs[i_][i_] = mueller_238u["missing_info"].iloc[i_] * mueller_238u["missing_info"].iloc[i_] * \
                          EF_238u[i_] * EF_238u[i_] * xs[i_] * xs[i_]
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

    plot_matrix(
        m_=total_238u_xs, xlabel_='bin', ylabel_='bin', title_=U8+r' total covariance',
        label_=r'Covariance [(\si{\centi\meter\squared}/\si{MeV}/fission)$^2$]', figsize_=[6.1, 5.],
        cmap_='nipy_spectral'
    )

    plot_function(
        x_=[mueller_238u.index], y_=[EF_238u], label_=[r'EF '+U8],
        ylabel_=r'$S_{238}$ [N$_{\nu}$/\si{MeV}/fission]', styles=['bo']
    )
    plot_function(
        x_=[mueller_238u.index], y_=[EF_238u*xs], label_=[r'EF '+U8],
        ylabel_=r'$S_{238}$ [\si{\centi\meter\squared}/\si{MeV}/fission]', styles=['bo']
    )


########################################################################################################################
# DYB/DYB+PP - covariance matrix from unfolding
########################################################################################################################
plot = False

covariance_dyb = np.loadtxt(path+'/CovMatrix_unfolding_DYB.txt')
N_dyb = covariance_dyb.shape[0]
corr_dyb = np.zeros((N_dyb, N_dyb))
for i in np.arange(0, N_dyb, 1):
    for j in np.arange(0, N_dyb, 1):
        corr_dyb[i, j] = covariance_dyb[i, j]/np.sqrt(covariance_dyb[i, i])/np.sqrt(covariance_dyb[j, j])

block_rows = np.split(covariance_dyb, 3, 0)
u_u, u_pu, u_tot = np.split(block_rows[0], 3, 1)
pu_u, pu_pu, pu_tot = np.split(block_rows[1], 3, 1)
tot_u, tot_pu, tot_tot = np.split(block_rows[2], 3, 1)

reordered_cov_dyb = np.block([
    [tot_tot, tot_u, tot_pu],
    [u_tot, u_u, u_pu],
    [pu_tot, pu_u, pu_pu]
])

full_covariance_dyb_pp = np.loadtxt(path+'/CovMatrix_unfolding_DYB_PROSPECT.txt')
full_covariance_dyb_pp = full_covariance_dyb_pp*1.e-86
N = full_covariance_dyb_pp.shape[0]
corr_dyb_pp = np.zeros((N, N))
for i in np.arange(0, N, 1):
    for j in np.arange(0, N, 1):
        corr_dyb_pp[i, j] = full_covariance_dyb_pp[i, j]/np.sqrt(full_covariance_dyb_pp[i, i])/np.sqrt(full_covariance_dyb_pp[j, j])

if plot:
    plot_matrix(
        m_=covariance_dyb, xlabel_='bin', ylabel_='bin', title_=r'DYB - covariance',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.1, 5.]
    )
    plot_matrix(
        m_=corr_dyb, xlabel_='bin', ylabel_='bin', title_=r'DYB - correlation', min_=-1, max_=1,
        label_=r'Correlation', figsize_=[6.1, 5.]
    )
    plot_matrix(
        m_=reordered_cov_dyb, xlabel_='bin', ylabel_='bin', title_=r'DYB - covariance',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.1, 5.]
    )

    plot_matrix(
        m_=full_covariance_dyb_pp, xlabel_='bin', ylabel_='bin', title_=r'DYB+PP - covariance',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.1, 5.]
    )
    plot_matrix(
        m_=corr_dyb_pp, xlabel_='bin', ylabel_='bin', title_=r'DYB+PP - correlation', min_=-1, max_=1,
        label_=r'Correlation', figsize_=[6.1, 5.]
    )


########################################################################################################################
# total covariance
########################################################################################################################
plot = False

# interpolating 238U and 241Pu matrices
energy = react.get_235u_dyb().index
xs = react.eval_xs(energy, bool_protons=False, which_xs='SV_approx')

f = interpolate.interp2d(mueller_238u.index, mueller_238u.index, total_238u, kind='linear')
new_total_238u = f(energy, energy)

f = interpolate.interp2d(huber_241pu.index, huber_241pu.index, np.log(total_241pu), kind='linear')
new_total_241pu = np.exp(f(energy, energy))

for i_ in np.arange(len(energy)):
    for j_ in np.arange(len(energy)):
        new_total_238u[i_, j_] = new_total_238u[i_, j_] * xs[i_] * xs[j_]
        new_total_241pu[i_, j_] = new_total_241pu[i_, j_] * xs[i_] * xs[j_]


# using covariance matrix for DYB with new order
total = np.block([
    [reordered_cov_dyb, np.zeros((N_dyb, N_m+N_h))],
    [np.zeros((N_m, N_dyb)), new_total_238u, np.zeros((N_m, N_h))],
    [np.zeros((N_h, N_dyb+N_m)), new_total_241pu]
])

if plot:
    plot_matrix(
        m_=total, xlabel_='bin', ylabel_='bin', title_=r'total',  # cmap_='nipy_spectral',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.15, 5.]
    )

    plot_matrix(
        m_=new_total_238u, xlabel_='bin', ylabel_='bin', title_=U8+r' total interpolated', cmap_='nipy_spectral',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.15, 5.]
    )
    plot_matrix(
        m_=new_total_241pu, xlabel_='bin', ylabel_='bin', title_=Pu1+r' total interpolated', cmap_='nipy_spectral',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.15, 5.]
    )


########################################################################################################################
# 241Pu toyMC
########################################################################################################################
plot = False

N_samples = 1000
huber_241pu = react.get_241pu_huber()
xs = react.eval_xs(huber_241pu.index, bool_protons=False, which_xs='SV_approx')
c = linalg.cholesky(total_241pu_xs)
N = len(huber_241pu.index)
spectrum = huber_241pu['spectrum']*xs

samples = stats.multivariate_normal.rvs(mean=spectrum, cov=total_241pu_xs, size=N_samples)

V = np.zeros((N, N))
for j in np.arange(0, N):
    for k in np.arange(0, N):
        V[j, k] = ((samples[:, j] - spectrum.iloc[j]) * (samples[:, k] - spectrum.iloc[k])).sum()
V = V/N_samples

if plot:
    plot_matrix(
        m_=c, xlabel_='bin', ylabel_='bin', title_=r'total', origin_='upper',
        label_=r'Cholesky dec. for 241Pu', figsize_=[6.15, 5.]
    )

    plt.figure()
    for i_ in np.arange(N_samples):
        plt.plot(huber_241pu.index, samples[i_])
    plt.plot(huber_241pu.index, huber_241pu['spectrum']*xs, 'ko', markersize=4)
    plt.grid(alpha=0.65)

    plot_matrix(
        m_=V, xlabel_='bin', ylabel_='bin', title_=r'from ToyMC - %i samples' % N_samples, origin_='lower',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.15, 5.], cmap_='nipy_spectral'
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(0, total_241pu_xs.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    axes[0].imshow(total_241pu_xs, origin='lower', cmap=cmap, norm=normalizer)
    axes[0].set_xlabel("bin")
    axes[0].set_ylabel("bin")
    axes[0].set_title("Analytical")
    axes[1].imshow(V, origin='lower', cmap=cmap, norm=normalizer)
    axes[1].set_xlabel("bin")
    axes[1].set_title(r'from ToyMC - %i samples' % N_samples)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=r' covariance matrix [(\si{\centi\meter\squared}/MeV/fission)$^2$]')
    cbar.formatter.set_powerlimits((0, 0))


########################################################################################################################
# Transformation matrix R and spectrum (125x1)
########################################################################################################################
plot = False

model_inputs = {
    'total': 'DYB',
    '235U': 'DYB',
    '238U': 'EF',
    '239Pu': 'DYB_combo',
    '241Pu': 'Huber'
}
energy = react.get_total_dyb().index.to_numpy()
xs = react.eval_xs(energy, bool_protons=False, which_xs='SV_approx')

input_spectra = react.eval_total(energy, which_input='DYB') * xs
input_spectra = np.append(input_spectra, react.eval_235u(energy, which_input='DYB') * xs)
input_spectra = np.append(input_spectra, react.eval_239pu(energy, which_input='DYB_combo') * xs)
input_spectra = np.append(input_spectra, react.eval_238u(energy, which_input='EF') * xs)
input_spectra = np.append(input_spectra, react.eval_241pu(energy, which_input='Huber') * xs)

df_235 = 0.016
df_239 = -0.004
df_238 = -0.006
df_241 = -0.006

id_25 = np.identity(25)
R = np.block(
    [id_25, df_235*id_25, df_239*id_25, df_238*id_25, (df_241-0.183*df_239)*id_25]
)

# expected reactor spectra w/o oscillations
juno_matrix = np.dot(R, input_spectra)
juno_formula = react.reactor_model_dyb(energy, model_inputs) * xs

if plot:
    ylabel = r'spectrum [\si{\centi\meter\squared}/MeV/fission]'
    ax = plot_function(
        x_=[np.arange(125)], y_=[input_spectra], styles=['bo'], label_=['spectrum'], ylabel_=ylabel,
        base_major=10.0, base_minor=5.0, xlabel_=r'bin'
    )
    ax.get_legend().remove()

    plot_matrix(
        m_=R, xlabel_='bin', ylabel_='bin', title_=r'', origin_='upper',
        label_=r'Transformation matrix', figsize_=[10., 4.], cmap_='nipy_spectral'
    )

    plot_function_residual(
        x_=[energy, energy], y_=[juno_formula, juno_matrix], label_=[r'formula', r'matrix'],
        styles=['b-', 'r--'], ylabel_=ylabel, y2_sci=True
    )


########################################################################################################################
# Obtain covariance matrix of final spectrum
########################################################################################################################
plot = True

# method 1: analytical approach
R_t = R.transpose()
cov_analytical = linalg.multi_dot([R, total, R_t])

# method 2: toyMC
N_smp = 10000
M = len(energy)

input_samples = stats.multivariate_normal.rvs(mean=input_spectra, cov=total, size=N_smp)
juno_samples = np.zeros((N_smp, M))
for i_ in np.arange(N_smp):
    juno_samples[i_] = np.dot(R, input_samples[i_])

cov_toyMC = np.zeros((M, M))
for j in np.arange(M):
    for k in np.arange(M):
        cov_toyMC[j, k] = ((juno_samples[:, j] - juno_matrix[j]) * (juno_samples[:, k] - juno_matrix[k])).sum()
cov_toyMC = cov_toyMC/N_smp

if plot:
    plot_matrix(
        m_=cov_analytical, xlabel_='bin', ylabel_='bin', title_=r'Covariance - analytical', origin_='lower',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.15, 5.], cmap_='nipy_spectral'
    )
    plot_function(
        x_=[energy], y_=[np.sqrt(np.diagonal(cov_analytical))/juno_matrix], label_=[r'spectral uncertainty'],
        ylabel_=r'relative uncert. - matrix approach', styles=['bo'], ylim=[0, 0.2]
    )

    # plt.figure(figsize=[7., 4.], constrained_layout=True)
    # for i_ in np.arange(N_smp):
    #     plt.plot(np.arange(len(input_spectra)), input_samples[i_])
    # plt.plot(np.arange(len(input_spectra)), input_spectra, 'ko', markersize=4)
    # plt.grid(alpha=0.65)
    #
    # plt.figure(constrained_layout=True)
    # for i_ in np.arange(N_smp):
    #     plt.plot(energy, juno_samples[i_])
    # plt.plot(energy, juno_matrix, 'ko', markersize=4)
    # plt.grid(alpha=0.65)

    plot_matrix(
        m_=cov_toyMC, xlabel_='bin', ylabel_='bin', title_=r'Covariance - toyMC (%s samples)' % N_smp, origin_='lower',
        label_=r'Covariance [(\si{\centi\meter\squared}/MeV/fission)$^2$]', figsize_=[6.15, 5.], cmap_='nipy_spectral'
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[9, 6], sharex=True, sharey=True, constrained_layout=True)
    cmap = cm.get_cmap('nipy_spectral')
    normalizer = Normalize(cov_analytical.min(), cov_analytical.max())
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    axes[0].imshow(cov_analytical, origin='lower', cmap=cmap, norm=normalizer)
    axes[0].set_xlabel("bin")
    axes[0].set_ylabel("bin")
    axes[0].set_title("Analytical")
    axes[1].imshow(cov_toyMC, origin='lower', cmap=cmap, norm=normalizer)
    axes[1].set_xlabel("bin")
    axes[1].set_title(r'From ToyMC - %i samples' % N_smp)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label=r' covariance matrix [(\si{\centi\meter\squared}/MeV/fission)$^2$]')
    cbar.formatter.set_powerlimits((0, 0))


# ax.pcolormesh

elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
