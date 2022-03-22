import numpy as np
from scipy import interpolate, stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib.colors import Normalize
import time
import json
import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')
sys.path.insert(0, cwd + '/Inputs/cov_matrices')
import latex
from plot import plot_function, plot_function_residual, plot_matrix, plot_two_matrices, plot_six_matrices
from reactor import UnoscillatedReactorSpectrum
from numpy import linalg
import matplotlib.colors as colors
import generate_covariance_matrices
import pandas as pd


########################################################################################################################
# useful stuff
########################################################################################################################

U5 = r'$^{235}$U'
U8 = r'$^{238}$U'
Pu9 = r'$^{239}$Pu'
Pu1 = r'$^{241}$Pu'

path = '/Users/beatricejelmini/Desktop/JUNO/JUNO_codes/JUNO_ReactorNeutrinosAnalysis/Inputs/cov_matrices'

label_covariance = r'Covariance [(N$_{\nu}$/MeV/fission)$^2$]'
label_covariance_xs = r'Covariance [(cm$^2$/MeV/fission)$^2$]'

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

stat_cov = generate_covariance_matrices.generate_from_huber("241", uncertainty_='stat')
stat_cov_xs = generate_covariance_matrices.generate_from_huber("241", uncertainty_='stat', scale_=xs)
stat_cov_rel = generate_covariance_matrices.generate_from_huber("241", uncertainty_='stat', absolute_=False)

bias_cov = generate_covariance_matrices.generate_from_huber("241", uncertainty_='bias')
bias_cov_xs = generate_covariance_matrices.generate_from_huber("241", uncertainty_='bias', scale_=xs)
bias_cov_rel = generate_covariance_matrices.generate_from_huber("241", uncertainty_='bias', absolute_=False)

z_cov = generate_covariance_matrices.generate_from_huber("241", uncertainty_='z')
z_cov_xs = generate_covariance_matrices.generate_from_huber("241", uncertainty_='z', scale_=xs)
z_cov_rel = generate_covariance_matrices.generate_from_huber("241", uncertainty_='z', absolute_=False)

wm_cov = generate_covariance_matrices.generate_from_huber("241", uncertainty_='wm')
wm_cov_xs = generate_covariance_matrices.generate_from_huber("241", uncertainty_='wm', scale_=xs)
wm_cov_rel = generate_covariance_matrices.generate_from_huber("241", uncertainty_='wm', absolute_=False)

norm_cov = generate_covariance_matrices.generate_from_huber("241", uncertainty_='norm')
norm_cov_xs = generate_covariance_matrices.generate_from_huber("241", uncertainty_='norm', scale_=xs)
norm_cov_rel = generate_covariance_matrices.generate_from_huber("241", uncertainty_='norm', absolute_=False)

total_241pu = generate_covariance_matrices.generate_from_huber("241", uncertainty_='total')
total_241pu_xs = generate_covariance_matrices.generate_from_huber("241", uncertainty_='total', scale_=xs)
total_241pu_rel = generate_covariance_matrices.generate_from_huber("241", uncertainty_='total', absolute_=False)

if plot:
    matrices = [stat_cov, bias_cov, z_cov, wm_cov, norm_cov, total_241pu]
    matrices_xs = [stat_cov_xs, bias_cov_xs, z_cov_xs, wm_cov_xs, norm_cov_xs, total_241pu_xs]
    matrices_rel = [stat_cov_rel, bias_cov_rel, z_cov_rel, wm_cov_rel, norm_cov_rel, total_241pu_rel]
    titles = ["stat", "bias", "Z", "WM", "norm", "total"]

    plot_six_matrices(
        matrices, xlabel_='bin', ylabel_='bin', label_=Pu1+r' ' + label_covariance,  origin_='lower',
        fig_length=9, fig_height=6, titles_=titles, sci_mode_=True, shrink=.9, mode_='powerlaw',
        constrained_layout_=True
    )

    plot_six_matrices(
        matrices_xs, xlabel_='bin', ylabel_='bin', label_=Pu1+r' ' + label_covariance_xs, origin_='lower',
        fig_length=9, fig_height=6, titles_=titles, sci_mode_=True, shrink=0.9, mode_='powerlaw',
        constrained_layout_=True
    )

    plot_six_matrices(
        matrices_rel, xlabel_='bin', ylabel_='bin', label_=Pu1+r' relative uncert. "[\%$^2$]"', origin_='lower',
        fig_length=9, fig_height=6, titles_=titles, sci_mode_=True, shrink=0.9, mode_='powerlaw',
        constrained_layout_=True
    )

    plot_matrix(
        m_=total_241pu_xs, xlabel_='bin', ylabel_='bin', title_=Pu1+r' total covariance',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5, shrink=1
    )

    plot_function(
        x_=[huber_241pu.index], y_=[huber_241pu['spectrum']], label_=[r'H '+Pu1],
        ylabel_=r'$S_{241}$ [N$_{\nu}$/MeV/fission]', styles=['bo']
    )
    plot_function(
        x_=[huber_241pu.index], y_=[huber_241pu['spectrum']*xs], label_=[r'H '+Pu1],
        ylabel_=r'$S_{241}$ [cm$^2$/MeV/fission]', styles=['bo']
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
nucleardb_cov = generate_covariance_matrices.generate_from_ef("238", uncertainty_='nuclear_db', absolute_=True)
nucleardb_cov_xs = generate_covariance_matrices.generate_from_ef("238", uncertainty_='nuclear_db', scale_=xs)
nucleardb_cov_rel = generate_covariance_matrices.generate_from_ef("238", uncertainty_='nuclear_db', absolute_=False)

forbid_cov = generate_covariance_matrices.generate_from_ef("238", uncertainty_='forbidden_treatment', absolute_=True)
forbid_cov_xs = generate_covariance_matrices.generate_from_ef("238", uncertainty_='forbidden_treatment', scale_=xs)
forbid_cov_rel = generate_covariance_matrices.generate_from_ef("238", uncertainty_='forbidden_treatment', absolute_=False)

corr_cov = generate_covariance_matrices.generate_from_ef("238", uncertainty_='corrections', absolute_=True)
corr_cov_xs = generate_covariance_matrices.generate_from_ef("238", uncertainty_='corrections', scale_=xs)
corr_cov_rel = generate_covariance_matrices.generate_from_ef("238", uncertainty_='corrections', absolute_=False)

missing_cov = generate_covariance_matrices.generate_from_ef("238", uncertainty_='missing_info', absolute_=True)
missing_cov_xs = generate_covariance_matrices.generate_from_ef("238", uncertainty_='missing_info', scale_=xs)
missing_cov_rel = generate_covariance_matrices.generate_from_ef("238", uncertainty_='missing_info', absolute_=False)

total_238u = nucleardb_cov + forbid_cov + corr_cov + missing_cov
total_238u_xs = nucleardb_cov_xs + forbid_cov_xs + corr_cov_xs + missing_cov_xs
total_238u_rel = nucleardb_cov_rel + forbid_cov_rel + corr_cov_rel + missing_cov_rel

if plot:
    matrices = [nucleardb_cov, forbid_cov, corr_cov, missing_cov, total_238u]
    matrices_xs = [nucleardb_cov_xs, forbid_cov_xs, corr_cov_xs, missing_cov_xs, total_238u_xs]
    matrices_rel = [nucleardb_cov_rel, forbid_cov_rel, corr_cov_rel, missing_cov_rel, total_238u_rel]
    titles = ["nuclear db", "forbidden", "corrections", "missing info", "total"]

    plot_six_matrices(
        matrices, xlabel_='bin', ylabel_='bin', label_=U8+r' ' + label_covariance, origin_='lower',
        fig_length=9, fig_height=6, titles_=titles, sci_mode_=True, shrink=0.9, mode_='powerlaw',
        constrained_layout_=True
    )

    plot_six_matrices(
        matrices_xs, xlabel_='bin', ylabel_='bin', label_=U8+r' ' + label_covariance_xs, origin_='lower',
        fig_length=9, fig_height=6, titles_=titles, sci_mode_=True, shrink=0.9, mode_='powerlaw',
        constrained_layout_=True
    )

    plot_six_matrices(
        matrices_rel, xlabel_='bin', ylabel_='bin', label_=U8+r' relative uncert. "[\%$^2$]"', origin_='lower',
        fig_length=9, fig_height=6, titles_=titles, sci_mode_=True, shrink=0.9, mode_='powerlaw',
        constrained_layout_=True
    )

    plot_matrix(
        m_=total_238u_xs, xlabel_='bin', ylabel_='bin', title_=U8+r' total covariance',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5, shrink=1,
    )

    plot_function(
        x_=[mueller_238u.index], y_=[EF_238u], label_=[r'EF '+U8],
        ylabel_=r'$S_{238}$ [N$_{\nu}$/MeV/fission]', styles=['bo']
    )
    plot_function(
        x_=[mueller_238u.index], y_=[EF_238u*xs], label_=[r'EF '+U8],
        ylabel_=r'$S_{238}$ [cm$^2$/MeV/fission]', styles=['bo']
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
        label_=label_covariance_xs, fig_length=6.15, fig_height=5, shrink=1, mode_='powerlaw'
    )
    plot_matrix(
        m_=corr_dyb, xlabel_='bin', ylabel_='bin', title_=r'DYB - correlation', min_=-1, max_=1,
        label_=r'Correlation', fig_length=6.15, fig_height=5, shrink=1, mode_='centered'
    )
    plot_matrix(
        m_=reordered_cov_dyb, xlabel_='bin', ylabel_='bin', title_=r'DYB - covariance',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5, shrink=1, mode_='powerlaw'
    )

    plot_matrix(
        m_=full_covariance_dyb_pp, xlabel_='bin', ylabel_='bin', title_=r'DYB+PP - covariance',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5, shrink=1, mode_='powerlaw'
    )
    plot_matrix(
        m_=corr_dyb_pp, xlabel_='bin', ylabel_='bin', title_=r'DYB+PP - correlation', min_=-1, max_=1,
        label_=r'Correlation', fig_length=6.15, fig_height=5, shrink=1, mode_='powerlaw'
    )


########################################################################################################################
# reshape of covariance matrix through toyMC
########################################################################################################################
plot = False

N_samples = 100
new_bins = react.get_235u_dyb().index
new_cov_238u, nn = generate_covariance_matrices.reshape_cov_matrix(new_bins, mueller_238u.index.to_numpy(),
                                                                   EF_238u, total_238u, n_samples_=N_samples)
new_cov_241pu, nn2 = generate_covariance_matrices.reshape_cov_matrix(new_bins, huber_241pu.index.to_numpy(),
                                                                     huber_241pu["spectrum"].to_numpy(), total_241pu,
                                                                     n_samples_=N_samples)

# new_bins_2 = np.arange(1.925, 8.125, 0.125)
new_bins_2 = np.arange(2, 8.25, 0.125)
new_cov_238u_2, nn_2 = generate_covariance_matrices.reshape_cov_matrix(new_bins_2, mueller_238u.index.to_numpy(),
                                                                       EF_238u, total_238u, n_samples_=N_samples)
new_cov_241pu_2, nn2_2 = generate_covariance_matrices.reshape_cov_matrix(new_bins_2, huber_241pu.index.to_numpy(),
                                                                         huber_241pu["spectrum"].to_numpy(),
                                                                         total_241pu, n_samples_=N_samples)

xs_reshape = react.eval_xs(new_bins, which_xs='SV_approx', bool_protons=False)

nnn = len(new_bins)
for i in np.arange(nnn):
    for j in np.arange(nnn):
        new_cov_238u[i, j] = new_cov_238u[i, j] * xs_reshape[i] * xs_reshape[j]
        new_cov_241pu[i, j] = new_cov_241pu[i, j] * xs_reshape[i] * xs_reshape[j]

xs_reshape_2 = react.eval_xs(new_bins_2, which_xs='SV_approx', bool_protons=False)
new_cov_238u_2_xs = np.zeros((len(new_bins_2), len(new_bins_2)))
new_cov_241pu_2_xs = np.zeros((len(new_bins_2), len(new_bins_2)))
for i in np.arange(len(new_bins_2)):
    for j in np.arange(len(new_bins_2)):
        new_cov_238u_2_xs[i, j] = new_cov_238u_2[i, j] * xs_reshape_2[i] * xs_reshape_2[j]
        new_cov_241pu_2_xs[i, j] = new_cov_241pu_2[i, j] * xs_reshape_2[i] * xs_reshape_2[j]

#
# plt.figure(constrained_layout=True)
# for i_ in np.arange(N_smpls):
#     plt.plot(old_binning, pure_samples[i_])
# plt.plot(old_binning, central_values, 'ko', markersize=4)
# plt.grid(alpha=0.65)
# #
#
# plt.figure(constrained_layout=True)
# for i_ in np.arange(N_smpls):
#     plt.plot(new_binning, reshaped_samples[i_])
# plt.plot(old_binning, central_values, 'ko', markersize=4)
# plt.plot(new_binning, new_central_values, 'ro', markersize=4)
# plt.grid(alpha=0.65)
if plot:
    plot_matrix(
        m_=new_cov_238u, xlabel_='bin', ylabel_='bin', title_=U8+r' reshaped cov matrix',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_matrix(
        m_=new_cov_241pu, xlabel_='bin', ylabel_='bin', title_=Pu1+r' reshaped cov matrix',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_two_matrices(
        [total_238u_xs, new_cov_238u], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=["Input", r'Output from ToyMC - %i samples' % N_samples],
        constrained_layout_=True, shrink=1
    )

    plot_two_matrices(
        [total_241pu_xs, new_cov_241pu], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=["Input", r'Output from ToyMC - %i samples' % N_samples],
        constrained_layout_=True, shrink=1
    )

    plot_matrix(
        m_=new_cov_238u_2_xs, xlabel_='bin', ylabel_='bin', title_=U8+r' reshaped cov matrix',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_matrix(
        m_=new_cov_241pu_2_xs, xlabel_='bin', ylabel_='bin', title_=Pu1+r' reshaped cov matrix',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_two_matrices(
        [total_238u_xs, new_cov_238u_2_xs], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=["Input", r'Output from ToyMC - %i samples' % N_samples],
        constrained_layout_=True, shrink=1
    )

    plot_two_matrices(
        [total_241pu_xs, new_cov_241pu_2_xs], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=["Input", r'Output from ToyMC - %i samples' % N_samples],
        constrained_layout_=True, shrink=1
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

# multiply by cross section
for i_ in np.arange(len(energy)):
    for j_ in np.arange(len(energy)):
        new_total_238u[i_, j_] = new_total_238u[i_, j_] * xs[i_] * xs[j_]
        new_total_241pu[i_, j_] = new_total_241pu[i_, j_] * xs[i_] * xs[j_]


# using covariance matrix for DYB with new order
total = np.block([
    [reordered_cov_dyb, np.zeros((N_dyb, N_m+N_h))],
    [np.zeros((N_m, N_dyb)), new_cov_238u, np.zeros((N_m, N_h))],
    [np.zeros((N_h, N_dyb+N_m)), new_cov_241pu]
])
np.save(path+'/total_covariance_125x125.npy', total)

if plot:
    plot_matrix(
        m_=total, xlabel_='bin', ylabel_='bin', title_=r'total', mode_='powerlaw',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_matrix(
        m_=new_total_238u, xlabel_='bin', ylabel_='bin', title_=U8+r' total interpolated',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )
    plot_matrix(
        m_=new_total_241pu, xlabel_='bin', ylabel_='bin', title_=Pu1+r' total interpolated',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_two_matrices(
        [total_238u_xs, new_total_238u], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=["Input", r'Output from interpolation'],
        constrained_layout_=True, shrink=1
    )

    plot_two_matrices(
        [total_241pu_xs, new_total_241pu], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=["Input", r'Output from interpolation'],
        constrained_layout_=True, shrink=1
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

V = generate_covariance_matrices.evaluate_cov_matrix_from_samples(samples, spectrum.to_numpy())

if plot:
    plot_matrix(
        m_=c, xlabel_='bin', ylabel_='bin', title_=r'total', origin_='upper',
        label_=r'Cholesky dec. for 241Pu', fig_length=6.15, fig_height=5
    )

    plt.figure()
    for i_ in np.arange(N_samples):
        plt.plot(huber_241pu.index, samples[i_])
    plt.plot(huber_241pu.index, huber_241pu['spectrum']*xs, 'ko', markersize=4)
    plt.grid(alpha=0.65)

    plot_matrix(
        m_=V, xlabel_='bin', ylabel_='bin', title_=r'from ToyMC - %i samples' % N_samples, origin_='lower',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_two_matrices(
        [total_241pu_xs, V], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=[r'Analytical', r'from ToyMC - %i samples' % N_samples],
        constrained_layout_=True, shrink=1
    )


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
R_2 = react.get_transformation_matrix()  # same as R, :D

# expected reactor spectra w/o oscillations
juno_matrix = np.dot(R, input_spectra)
juno_formula = react.reactor_model_dyb(energy, model_inputs) * xs
juno_matrix2 = react.reactor_model_matrixform(energy, model_inputs, pu_combo=True)*xs  # same as juno_matrix, :D

E = np.arange(1.81, 9, 0.10)
f_appo = interpolate.interp1d(energy, np.log(juno_matrix2),
                              kind='linear', fill_value="extrapolate")
juno_matrix3 = np.exp(f_appo(E))
juno_matrix4 = react.reactor_model_matrixform(E, model_inputs, pu_combo=True) * react.eval_xs(E, bool_protons=False, which_xs='SV_approx')
# juno_matrix3 and juno_matrix4 are different!
ii = react.get_input_spectrum_array(model_inputs, xs_=False)
ii2 = react.get_input_spectrum_array(model_inputs, xs_=True)

if plot:
    ylabel = r'spectrum [cm$^2$/MeV/fission]'
    ax = plot_function(
        x_=[np.arange(125)], y_=[input_spectra], styles=['bo'], label_=['spectrum'], ylabel_=ylabel,
        base_major=10.0, base_minor=5.0, xlabel_=r'bin'
    )
    ax.get_legend().remove()

    plot_matrix(
        m_=R, xlabel_='bin', ylabel_='bin', title_=r'', origin_='upper', constrained_layout_=True,
        label_=r'Transformation matrix', fig_length=11, fig_height=4, sci_mode_=False, shrink=0.5, mode_='powerlaw'
    )

    plot_function_residual(
        x_=[energy, energy], y_=[juno_formula, juno_matrix], label_=[r'formula', r'matrix'],
        styles=['b-', 'r--'], ylabel_=ylabel, y2_sci=True
    )


########################################################################################################################
# Obtain covariance matrix of final reactor spectrum
########################################################################################################################
plot = False

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

cov_toyMC = generate_covariance_matrices.evaluate_cov_matrix_from_samples(juno_samples, juno_matrix)

if plot:
    plot_matrix(
        m_=cov_analytical, xlabel_='bin', ylabel_='bin', title_=r'Covariance - analytical', origin_='lower',
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
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
        label_=label_covariance_xs, fig_length=6.15, fig_height=5
    )

    plot_two_matrices(
        [cov_analytical, cov_toyMC], xlabel_='bin', ylabel_='bin', label_=label_covariance_xs, origin_='lower',
        fig_length=12.3, fig_height=5, titles_=[r'Analytical', r'from ToyMC - %i samples' % N_smp],
        constrained_layout_=True, shrink=1
    )


elapsed_time = time.perf_counter_ns() - time_start
elapsed_time = elapsed_time * 10 ** (-9)  # in seconds
mins = int(elapsed_time / 60.)
sec = elapsed_time - 60. * mins
print('\nelapsed time: ' + str(mins) + ' mins ' + str(sec) + ' s')

plt.ion()
plt.show()
