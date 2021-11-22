import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import math
from scipy import integrate

from reactor import ReactorSpectrum
from oscillation import OscillationProbability
from detector_response import DetectorResponse

# TODO:
# - remove parts for plotting --> improved, DONE
# - adapt change of baselines and powers --> add methods, read from file
# - initialise with .json file --> DONE
# - include sum over more reactors in single method
# - use DetectorResponse as parent class? --> move a b c in Detector Response --> DONE


class OscillatedSpectrum(OscillationProbability, ReactorSpectrum, DetectorResponse):

    def __init__(self, inputs_json_):
        ReactorSpectrum.__init__(self, inputs_json_)
        OscillationProbability.__init__(self, inputs_json_)
        DetectorResponse.__init__(self, inputs_json_)

        self.baseline = 52.5  # [km]

        self.osc_spect_N = 0.
        self.osc_spect_I = 0.
        self.resol_N = 0.
        self.resol_I = 0.
        self.sum_spectra_N = 0.
        self.sum_spectra_I = 0.
        self.sum_resol_N = 0.
        self.sum_resol_I = 0.

        self.baselines = []
        self.powers = []

    # TODO: need adjustments
    def set_L_P_distribution(self, baselines, powers):
        self.baselines = baselines
        self.powers = powers

    def osc_spectrum_N(self, nu_energy_, matter=True, which_xsec='SV', which_isospectrum='DYB',
                       bool_snf=True, bool_noneq=True, plot_this=False, plot_un=False):

        ReactorSpectrum.antinu_spectrum_no_osc(self, nu_energy_,
                                               which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                                               bool_snf=bool_snf, bool_noneq=bool_noneq)

        if matter:
            ylabel_ = r'$S_{\nu}$ [N$_{\nu}$/\si{s}/\si{\MeV}] (in matter)'
            print("\nevaluating oscillation probability in matter - N")
            prob = OscillationProbability.eval_matter_prob_N(self, nu_energy_)
        else:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]'
            print("\nevaluating oscillation probability in vacuum - N")
            prob = OscillationProbability.eval_vacuum_prob_N(self, nu_energy_)

        self.osc_spect_N = self.spectrum_unosc * prob

        if plot_this:
            if plot_un:
                plot_function(x_=[nu_energy_, nu_energy_], y_=[self.spectrum_unosc, self.osc_spect_N],
                              label_=[r'Unoscillated spectrum', r'NO'], colours=['k', 'b'],
                              ylabel_=ylabel_, xlim=[1.5, 10.5], ylim=None)
            else:
                plot_function(x_=[nu_energy_], y_=[self.osc_spect_N], label_=[r'NO'], colours=['b'],
                              ylabel_=ylabel_, xlim=[1.5, 10.5], ylim=None)

        return self.osc_spect_N

    def osc_spectrum_I(self, nu_energy_, matter=True, which_xsec='SV', which_isospectrum='DYB',
                       bool_snf=True, bool_noneq=True, plot_this=False, plot_un=False):

        ReactorSpectrum.antinu_spectrum_no_osc(self, nu_energy_,
                                               which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                                               bool_snf=bool_snf, bool_noneq=bool_noneq)
        if matter:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}] (in matter)'
            print("\nevaluating oscillation probability in matter - I")
            prob = OscillationProbability.eval_matter_prob_I(self, nu_energy_)
        else:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]'
            print("\nevaluating oscillation probability in vacuum - I")
            prob = OscillationProbability.eval_vacuum_prob_I(self, nu_energy_)

        self.osc_spect_I = self.spectrum_unosc * prob

        if plot_this:
            if plot_un:
                plot_function(x_=[nu_energy_, nu_energy_], y_=[self.spectrum_unosc, self.osc_spect_I],
                              label_=[r'Unoscillated spectrum', r'IO'], colours=['k', 'r'],
                              ylabel_=ylabel_, xlim=[1.5, 10.5], ylim=None)
            else:
                plot_function(x_=[nu_energy_], y_=[self.osc_spect_I], label_=[r'IO'], colours=['r'],
                              ylabel_=ylabel_, xlim=[1.5, 10.5], ylim=None)

        return self.osc_spect_I

    def osc_spectrum(self, nu_energy_, matter=True, which_xsec='SV', which_isospectrum='DYB',
                     bool_snf=True, bool_noneq=True, plot_this=False, plot_un=False):

        self.osc_spectrum_N(nu_energy_, matter=matter, which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                            bool_snf=bool_snf, bool_noneq=bool_noneq)
        self.osc_spectrum_I(nu_energy_, matter=matter, which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                            bool_snf=bool_snf, bool_noneq=bool_noneq)

        if matter:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}] (in matter)'
        else:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]'

        if plot_this:
            if plot_un:
                plot_function(x_=[nu_energy_, nu_energy_, nu_energy_],
                              y_=[self.spectrum_unosc, self.osc_spect_N, self.osc_spect_I],
                              label_=[r'Unoscillated spectrum', r'NO', r'IO'], colours=['k', 'b', 'r--'],
                              ylabel_=ylabel_, xlim=[1.5, 10.5], ylim=None)
            else:
                plot_function(x_=[nu_energy_, nu_energy_], y_=[self.osc_spect_N, self.osc_spect_I], label_=[r'NO', r'IO'],
                              colours=['b', 'r--'], ylabel_=ylabel_, xlim=[1.5, 10.5], ylim=None)

        return self.osc_spect_N, self.osc_spect_I

    ### oscillated spectrum with energy resolution (via numerical convolution)
    ### for further reference: https://arxiv.org/abs/1210.8141, eq. (2.12) and (2.14)
    ### see also the implementation of the numerical convolution in the class DetectorResponse

    def resol_spectrum_N(self, visible_energy_, matter=True, which_xsec='SV', which_isospectrum='DYB',
                         bool_snf=True, bool_noneq=True, plot_this=False):

        # nu_energy = np.arange(1.806, 30.01, 0.01)
        nu_energy = np.arange(1.925, 8.65, 0.01)
        dep_energy = nu_energy - 0.78

        self.osc_spectrum_N(nu_energy, matter=matter,
                            which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                            bool_snf=bool_snf, bool_noneq=bool_noneq)

        print('adding experimental resolution via numerical convolution, it might take some time.')
        self.resol_N = DetectorResponse.gaussian_smearing_abc(self, self.osc_spect_N, dep_energy, visible_energy_)

        if matter:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}] (in matter)'
        else:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]'

        if plot_this:
            plot_function(x_=[visible_energy_], y_=[self.resol_N], label_=[r'NO'], colours=['b'],
                          ylabel_=ylabel_, xlabel_=r'$E_{\text{vis}}$ [\si{MeV}]',
                          xlim=[1.5-1., 10.5-1.], ylim=None)

        return self.resol_N

    def resol_spectrum_I(self, visible_energy_, matter=True, which_xsec='SV', which_isospectrum='DYB',
                         bool_snf=True, bool_noneq=True, plot_this=False):

        nu_energy = np.arange(1.925, 8.65, 0.01)
        dep_energy = nu_energy - 0.78

        self.osc_spectrum_I(nu_energy, matter=matter,
                            which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                            bool_snf=bool_snf, bool_noneq=bool_noneq)

        print('adding experimental resolution via numerical convolution, it might take some time.')
        self.resol_I = DetectorResponse.gaussian_smearing_abc(self, self.osc_spect_I, dep_energy, visible_energy_)

        if matter:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}] (in matter)'
        else:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]'

        if plot_this:
            plot_function(x_=[visible_energy_], y_=[self.resol_I], label_=[r'IO'], colours=['r'],
                          ylabel_=ylabel_, xlabel_=r'$E_{\text{vis}}$ [\si{MeV}]',
                          xlim=[1.5-1., 10.5-1.], ylim=None)

        return self.resol_I

    def resol_spectrum(self, visible_energy_, matter=True, which_xsec='SV', which_isospectrum='DYB',
                       bool_snf=True, bool_noneq=True, plot_this=False):

        self.resol_spectrum_N(visible_energy_, matter=matter,
                              which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                              bool_snf=bool_snf, bool_noneq=bool_noneq)
        self.resol_spectrum_I(visible_energy_, matter=matter,
                              which_xsec=which_xsec, which_isospectrum=which_isospectrum,
                              bool_snf=bool_snf, bool_noneq=bool_noneq)

        if matter:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}] (in matter)'
        else:
            ylabel_ = r'$S(\bar{\nu})$ [N$_{\nu}$/\si{s}/\si{\MeV}]'

        if plot_this:
            plot_function(x_=[visible_energy_, visible_energy_], y_=[self.resol_N, self.resol_I],
                          label_=[r'NO', r'IO'], colours=['b', 'r--'],
                          ylabel_=ylabel_, xlabel_=r'$E_{\text{vis}}$ [\si{MeV}]',
                          xlim=[1.5-1., 10.5-1.], ylim=None)

        return self.resol_N, self.resol_I


###  TODO:   REVISION NEEDED
    def sum(self, baselines, powers, E, normalize=False, plot_sum=False, plot_baselines=False):

        if len(baselines) != len(powers):
            print('Error: length of baselines array is different from length of powers array.')
            return -1

        N_cores = len(baselines)
        self.sum_spectra_N = np.zeros(len(E))
        self.sum_spectra_I = np.zeros(len(E))

        loc = plticker.MultipleLocator(base=2.0)
        loc1 = plticker.MultipleLocator(base=0.5)

        if plot_baselines:
            fig_b = plt.figure(figsize=[10.5, 6.5])
            # fig_b = plt.figure()
            ax_b = fig_b.add_subplot(111)
            fig_b.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.10)

        for n_ in np.arange(0, N_cores):
            self.baseline = baselines[n_]
            self.osc_spectrum(E, 0, normalize=True)
            self.sum_spectra_N = self.sum_spectra_N \
                                 + self.norm_osc_spect_N * powers[n_] / math.pow(baselines[n_], 2)
            self.sum_spectra_I = self.sum_spectra_I \
                                 + self.norm_osc_spect_I * powers[n_] / math.pow(baselines[n_], 2)
            if plot_baselines:
                ax_b.plot(E, self.norm_osc_spect_N, linewidth=1., label=r'L = %.2f \si{\km}' % baselines[n_])

        if plot_baselines:
            # ax_b.set_title(r'Antineutrino spectra at different baselines (NO)')
            ax_b.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax_b.set_xlim(1.5, 10.5)
            ax_b.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax_b.set_ylim(-0.01, 0.61)
            ax_b.xaxis.set_major_locator(loc)
            ax_b.xaxis.set_minor_locator(loc1)
            ax_b.tick_params('both', direction='out', which='both')
            ax_b.legend()
            ax_b.grid(alpha=0.65)
            # fig_b.savefig('SpectrumPlots/baselines.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/baselines.pdf')

        if normalize:
            norm_N = integrate.simps(self.sum_spectra_N, E)
            norm_I = integrate.simps(self.sum_spectra_I, E)
            self.sum_spectra_N = self.sum_spectra_N / norm_N
            self.sum_spectra_I = self.sum_spectra_I / norm_I

        if plot_sum:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.12, right=0.96, top=0.95)
            ax.plot(E, self.sum_spectra_N, 'b', linewidth=1., label='NO')
            ax.plot(E, self.sum_spectra_I, 'r--', linewidth=1., label='IO')
            ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax.set_xlim(1.5, 10.5)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax.set_ylim(-5.e-5, 4.e-3)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            if normalize:
                ax.ticklabel_format(axis='y', style='plain')
                ax.set_ylim(-0.005, 0.305)
            # ax.set_title(r'Antineutrino spectra with true baseline distribution')
            ax.legend()
            ax.grid(alpha=0.65)
            # fig.savefig('SpectrumPlots/sum.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/sum.pdf')

        # set baseline to default ideal value
        self.baseline = 52.5

        return self.sum_spectra_N, self.sum_spectra_I

    def sum_resol(self, baselines, powers, E, a, b, normalize=False, plot_sum=False, plot_baselines=False):

        if len(baselines) != len(powers):
            print('Error: length of baselines array is different from length of powers array.')
            return -1

        N_cores = len(baselines)
        self.sum_resol_N = np.zeros(len(E))
        self.sum_resol_I = np.zeros(len(E))

        loc = plticker.MultipleLocator(base=2.0)
        loc1 = plticker.MultipleLocator(base=0.5)

        if plot_baselines:
            fig_b = plt.figure(figsize=[10.5, 6.5])
            ax_b = fig_b.add_subplot(111)
            fig_b.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.10)

        for n_ in np.arange(0, N_cores):
            self.baseline = baselines[n_]
            self.resol_spectrum(E, a, b, 0, normalize=True)
            self.sum_resol_N = self.sum_resol_N + self.resol_N * powers[n_] / math.pow(baselines[n_], 2)
            self.sum_resol_I = self.sum_resol_I + self.resol_I * powers[n_] / math.pow(baselines[n_], 2)

            if plot_baselines:
                ax_b.plot(E, self.resol_N, linewidth=1., label=r'L = %.2f \si{\km}' % baselines[n_])

        if plot_baselines:
            # ax_b.set_title(r'Antineutrino spectra at different baselines (NO)' + '\nwith energy resolution')
            ax_b.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
            ax_b.set_xlim(0.5, 9.5)
            ax_b.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax_b.set_ylim(-0.01, 0.61)
            ax_b.xaxis.set_major_locator(loc)
            ax_b.xaxis.set_minor_locator(loc1)
            ax_b.tick_params('both', direction='out', which='both')
            ax_b.legend()
            ax_b.grid(alpha=0.65)
            # fig_b.savefig('SpectrumPlots/resol_baselines.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/resol_baselines.pdf')

        if normalize:
            norm_N = integrate.simps(self.sum_resol_N, E - 0.8)
            norm_I = integrate.simps(self.sum_resol_I, E - 0.8)
            self.sum_resol_N = self.sum_resol_N / norm_N
            self.sum_resol_I = self.sum_resol_I / norm_I

        if plot_sum:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.12, right=0.96, top=0.95)
            ax.plot(E, self.sum_resol_N, 'b', linewidth=1., label='NO')
            ax.plot(E, self.sum_resol_I, 'r--', linewidth=1., label='IO')
            ax.set_xlabel(r'$E_{\text{vis}}$ [\si{MeV}]')
            ax.set_xlim(0.5,9.5)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax.set_ylim(-5.e-5, 4.e-3)
            if normalize:
                ax.ticklabel_format(axis='y', style='plain')
                ax.set_ylim(-0.005, 0.305)
            # ax.set_title(
            #     r'Antineutrino spectra with true baseline distribution' + '\nwith energy resolution (\SI{3}{\percent} at \SI{1}{\MeV})')
            ax.legend()
            ax.grid(alpha=0.65)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # fig.savefig('SpectrumPlots/resol_sum.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/resol_sum.pdf')

        # set baseline to default ideal value
        self.baseline = 52.5

        return self.sum_resol_N, self.sum_resol_I

    def eval_NO(self, E, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # NO: 3l = 31
        self.sin2_13_N = t13
        self.deltam_3l_N = m3l  # [eV^2]

        ReactorSpectrum.unosc_spectrum(self, E)
        OscillationProbability.eval_prob(self, E, 1)

        self.norm_osc_spect_N = self.norm_spectrum_un * self.prob_E_N

        if self.norm_bool:
            norm = integrate.simps(self.norm_osc_spect_N, E)
            self.norm_osc_spect_N = self.norm_osc_spect_N / norm

        return self.norm_osc_spect_N

    def eval_IO(self, E, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # IO: 3l = 32
        self.sin2_13_I = t13
        self.deltam_3l_I = m3l  # [eV^2]

        ReactorSpectrum.unosc_spectrum(self, E)
        OscillationProbability.eval_prob(self, E, 1)

        self.norm_osc_spect_I = self.norm_spectrum_un * self.prob_E_I

        if self.norm_bool:
            norm = integrate.simps(self.norm_osc_spect_I, E)
            self.norm_osc_spect_I = self.norm_osc_spect_I / norm

        return self.norm_osc_spect_I

    def eval_resol_NO(self, E_fin, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # NO: 3l = 31
        self.sin2_13_N = t13
        self.deltam_3l_N = m3l  # [eV^2]

        E_nu = np.arange(1.806, 30.01, 0.01)
        E_dep = E_nu - 0.8

        ReactorSpectrum.unosc_spectrum(self, E_nu)
        OscillationProbability.eval_prob(self, E_nu, 1)

        self.norm_osc_spect_N = self.norm_spectrum_un * self.prob_E_N

        det_response = DetectorResponse()
        self.resol_N = det_response.gaussian_smearing(self.norm_osc_spect_N, E_dep, E_fin, a=self.a, b=self.b)

        if self.norm_bool:
            norm = integrate.simps(self.resol_N, E_fin)
            self.resol_N = self.resol_N / norm

        return self.resol_N

    def eval_resol_IO(self, E_fin, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # IO: 3l = 32
        self.sin2_13_I = t13
        self.deltam_3l_I = m3l  # [eV^2]

        E_nu = np.arange(1.806, 30.01, 0.01)
        E_dep = E_nu - 0.8

        ReactorSpectrum.unosc_spectrum(self, E_nu)
        OscillationProbability.eval_prob(self, E_nu, -1)

        self.norm_osc_spect_I = self.norm_spectrum_un * self.prob_E_I

        det_response = DetectorResponse()
        self.resol_I = det_response.gaussian_smearing(self.norm_osc_spect_I, E_dep, E_fin, a=self.a, b=self.b)

        if self.norm_bool:
            norm = integrate.simps(self.resol_I, E_fin)
            self.resol_I = self.resol_I / norm

        return self.resol_I

    def eval_sum_NO(self, E, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # NO: 3l = 31
        self.sin2_13_N = t13
        self.deltam_3l_N = m3l  # [eV^2]

        N_cores = len(self.baselines)
        self.sum_spectra_N = np.zeros(len(E))

        for n_ in np.arange(0, N_cores):
            self.baseline = self.baselines[n_]
            self.osc_spectrum(E, 1, normalize=True)
            self.sum_spectra_N = self.sum_spectra_N + self.norm_osc_spect_N * self.powers[n_] \
                                 / math.pow(self.baselines[n_], 2)

        if self.norm_bool:
            norm = integrate.simps(self.sum_spectra_N, E)
            self.sum_spectra_N = self.sum_spectra_N / norm

        # set baseline to default ideal value
        self.baseline = 52.5

        return self.sum_spectra_N

    def eval_sum_IO(self, E, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # IO: 3l = 32
        self.sin2_13_I = t13
        self.deltam_3l_I = m3l  # [eV^2]

        N_cores = len(self.baselines)
        self.sum_spectra_I = np.zeros(len(E))

        for n_ in np.arange(0, N_cores):
            self.baseline = self.baselines[n_]
            self.osc_spectrum(E, -1, normalize=True)
            self.sum_spectra_I = self.sum_spectra_I + self.norm_osc_spect_I * self.powers[n_] \
                                 / math.pow(self.baselines[n_], 2)

        if self.norm_bool:
            norm = integrate.simps(self.sum_spectra_I, E)
            self.sum_spectra_I = self.sum_spectra_I / norm

        # set baseline to default ideal value
        self.baseline = 52.5

        return self.sum_spectra_I

    def eval_sum_resol_NO(self, E_fin, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # NO: 3l = 31
        self.sin2_13_N = t13
        self.deltam_3l_N = m3l  # [eV^2]

        E_nu = np.arange(1.806, 30.01, 0.01)
        E_dep = E_nu - 0.8

        N_cores = len(self.baselines)
        self.sum_spectra_N = np.zeros(len(E_nu))

        det_response = DetectorResponse()

        for n_ in np.arange(0, N_cores):
            self.baseline = self.baselines[n_]
            self.osc_spectrum(E_nu, 1, normalize=True)
            self.sum_spectra_N = self.sum_spectra_N + self.norm_osc_spect_N * self.powers[n_] \
                                 / math.pow(self.baselines[n_], 2)

        self.sum_resol_N = det_response.numerical_det_response(self.sum_spectra_N, E_dep, E_fin, a=self.a, b=self.b)

        if self.norm_bool:
            norm = integrate.simps(self.sum_resol_N, E_fin)
            self.sum_resol_N = self.sum_resol_N / norm

        # set baseline to default ideal value
        self.baseline = 52.5

        return self.sum_resol_N

    def eval_sum_resol_IO(self, E_fin, t12, m21, t13, m3l):

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # IO: 3l = 32
        self.sin2_13_I = t13
        self.deltam_3l_I = m3l  # [eV^2]

        E_nu = np.arange(1.806, 30.01, 0.01)
        E_dep = E_nu - 0.8

        N_cores = len(self.baselines)
        self.sum_spectra_I = np.zeros(len(E_nu))

        det_response = DetectorResponse()

        for n_ in np.arange(0, N_cores):
            self.baseline = self.baselines[n_]
            self.osc_spectrum(E_nu, -1, normalize=True)
            self.sum_spectra_I = self.sum_spectra_I + self.norm_osc_spect_I * self.powers[n_] \
                                 / math.pow(self.baselines[n_], 2)

        self.sum_resol_I = det_response.gaussian_smearing(self.sum_spectra_I, E_dep, E_fin, a=self.a, b=self.b)

        if self.norm_bool:
            norm = integrate.simps(self.sum_resol_I, E_fin)
            self.sum_resol_I = self.sum_resol_I / norm

        # set baseline to default ideal value
        self.baseline = 52.5

        return self.sum_resol_I


def plot_function(x_, y_, label_, colours, ylabel_, xlabel_=r'$E_{\nu}$ [\si{MeV}]', xlim=None, ylim=None, logx=False):
    if len(x_) != len(y_):
        print("Error in plot_function: different lengths - skip plotting")
        return 1

    loc = plticker.MultipleLocator(base=2.0)
    loc1 = plticker.MultipleLocator(base=0.5)

    fig = plt.figure(figsize=[8, 5.5])
    fig.subplots_adjust(left=0.09, right=0.97, top=0.95)
    ax_ = fig.add_subplot(111)
    for i_ in np.arange(len(x_)):
        if not logx:
            ax_.plot(x_[i_], y_[i_], colours[i_], linewidth=1., label=label_[i_])
        else:
            ax_.semilogx(x_[i_], y_[i_], colours[i_], linewidth=1., label=label_[i_])

    ax_.grid(alpha=0.65)
    ax_.set_xlabel(xlabel_)
    ax_.set_ylabel(ylabel_)

    if xlim is not None:
        ax_.set_xlim(xlim)
    if ylim is not None:
        ax_.set_ylim(ylim)

    if not logx:
        ax_.xaxis.set_major_locator(loc)
        ax_.xaxis.set_minor_locator(loc1)
    ax_.tick_params('both', direction='out', which='both')
    ax_.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_.legend()
