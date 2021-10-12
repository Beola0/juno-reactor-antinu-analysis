import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
import sys

# TODO:
# - initialize class with .json file with fission fractions --> DONE
# - create spectrum from inputs, like DYB spectrum --> DONE
# - remove part for plotting
# - add methods to change fission fractions --> DONE
# - move IBD part to DetectorResponse (??)
# - add SNF and NonEq contributions --> DONE
# - add nuisances: for matter density, SNF and NonEq


class ReactorSpectrum:

    def __init__(self, inputs_json_):

        self.fiss_frac_235u = inputs_json_["fission_fractions"]["235U"]
        self.fiss_frac_239pu = inputs_json_["fission_fractions"]["239Pu"]
        self.fiss_frac_238u = inputs_json_["fission_fractions"]["238U"]
        self.fiss_frac_241pu = inputs_json_["fission_fractions"]["241Pu"]

        self.fiss_en_235u = inputs_json_["mean_fission_energy"]["235U"]
        self.fiss_en_239pu = inputs_json_["mean_fission_energy"]["239Pu"]
        self.fiss_en_238u = inputs_json_["mean_fission_energy"]["238U"]
        self.fiss_en_241pu = inputs_json_["mean_fission_energy"]["241Pu"]

        self.thermal_power = inputs_json_["thermal_power"]
        self.baseline = inputs_json_["baseline"]

        self.inputs_json = inputs_json_

        self.iso_spectrum = 0.
        self.react_spectrum = 0.
        self.react_flux = 0.
        self.x_sec = 0.
        self.x_sec_np = 0.
        self.spectrum_unosc = 0.
        self.proton_number = 0.
        self.snf = 0.
        self.noneq = 0.

        self.which_xsec = ''
        self.which_isospectrum = ''
        
    def set_fission_fractions(self, f235u_, f239pu_, f238u_, f241pu_):
        self.fiss_frac_235u = f235u_
        self.fiss_frac_239pu = f239pu_
        self.fiss_frac_238u = f238u_
        self.fiss_frac_241pu = f241pu_
        
    def set_f235u(self, f235u_):
        self.fiss_frac_235u = f235u_
        
    def set_f238u(self, f238u_):
        self.fiss_frac_238u = f238u_

    def set_f239pu(self, f239pu_):
        self.fiss_frac_239pu = f239pu_
        
    def set_f241pu(self, f241pu_):
        self.fiss_frac_241pu = f241pu_
        
    def get_f235u(self):
        return self.fiss_frac_235u
        
    def get_f238u(self):
        return self.fiss_frac_238u

    def get_f239pu(self):
        return self.fiss_frac_239pu
        
    def get_f241pu(self):
        return self.fiss_frac_241pu

    def set_fission_energies(self, e235u_, e239pu_, e238u_, e241pu_):
        self.fiss_en_235u = e235u_
        self.fiss_en_239pu = e239pu_
        self.fiss_en_238u = e238u_
        self.fiss_en_241pu = e241pu_

    def set_e235u(self, e235u_):
        self.fiss_en_235u = e235u_

    def set_e238u(self, e238u_):
        self.fiss_en_238u = e238u_

    def set_e239pu(self, e239pu_):
        self.fiss_en_239pu = e239pu_

    def set_e241pu(self, e241pu_):
        self.fiss_en_241pu = e241pu_

    def get_e235u(self):
        return self.fiss_en_235u

    def get_e238u(self):
        return self.fiss_en_238u

    def get_e239pu(self):
        return self.fiss_en_239pu

    def get_e241pu(self):
        return self.fiss_en_241pu

    def set_th_power(self, val_):
        self.thermal_power = val_

    def get_th_power(self):
        return self.thermal_power

    def set_baseline(self, val_):
        self.baseline = val_

    def get_baseline(self):
        return self.baseline

    @staticmethod
    def isotopic_spectrum_exp(x_, params_):
        n_ = len(params_)
        appo = 0.
        for j_ in np.arange(n_):
            appo = appo + params_[j_] * np.power(x_, j_)
        return np.exp(appo)

    ### from DYB arXiv:1607.05378 - common inputs
    def get_snf_ratio(self, nu_energy_):

        input_ = pd.read_csv("Inputs/SNF_FluxRatio.txt", sep="\t",
                             names=["nu_energy", "snf_ratio"], header=None)

        f_appo = interp1d(input_["nu_energy"], input_["snf_ratio"])
        self.snf = f_appo(nu_energy_)

        return self.snf

    ### from DYB arXiv:1607.05378 - common inputs
    def get_noneq_ratio(self, nu_energy_):

        input_ = pd.read_csv("Inputs/NonEq_FluxRatio.txt", sep="\t",
                             names=["nu_energy", "noneq_ratio"], header=None)

        f_appo = interp1d(input_["nu_energy"], input_["noneq_ratio"])
        self.noneq = f_appo(nu_energy_)

        return self.noneq

    def isotopic_spectrum_vogel(self, nu_energy_, plot_this=False):

        self.which_isospectrum = 'V'
        print("\nUsing Vogel isotopic spectra")

        ### params taken from Vogel, Engel, PRD 39-11 pp 3378, 1989
        ### exponential of a polynomial of second order
        params_u235 = [0.870, -0.160, -0.091]
        params_pu239 = [0.896, -0.239, -0.0981]
        params_u238 = [0.976, -0.162, -0.0790]
        params_pu241 = [0.793, -0.080, -0.1085]
        u235 = self.isotopic_spectrum_exp(nu_energy_, params_u235)
        pu239 = self.isotopic_spectrum_exp(nu_energy_, params_pu239)
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        self.iso_spectrum = self.fiss_frac_235u * u235 + self.fiss_frac_239pu * pu239 \
                            + self.fiss_frac_238u * u238 + self.fiss_frac_241pu * pu241

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.iso_spectrum, 'k', linewidth=1.5, label='Total')
            # ax.plot(nu_energy_, self.f235u * u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            # ax.plot(nu_energy_, self.f239pu * pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            # ax.plot(nu_energy_, self.f238u * u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            # ax.plot(nu_energy_, self.f241pu * pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.plot(nu_energy_, u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            ax.plot(nu_energy_, pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            ax.plot(nu_energy_, u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            ax.plot(nu_energy_, pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.legend()
            ax.grid(alpha=0.65)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (Vogel)')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return self.iso_spectrum

    def isotopic_spectrum_hubermueller(self, nu_energy_, plot_this=False):

        self.which_isospectrum = 'HM'
        print("\nUsing Huber+Mueller isotopic spectra")

        ### params taken from Mueller PRC 83 (2011) for 238U and Huber PRC 84 (2011) for others
        ### exponential of a polynomial of fifth order
        params_u235 = [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]
        params_pu239 = [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]
        params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
        params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]
        u235 = self.isotopic_spectrum_exp(nu_energy_, params_u235)
        pu239 = self.isotopic_spectrum_exp(nu_energy_, params_pu239)
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        self.iso_spectrum = self.fiss_frac_235u * u235 + self.fiss_frac_239pu * pu239 \
                            + self.fiss_frac_238u * u238 + self.fiss_frac_241pu * pu241

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.iso_spectrum, 'k', linewidth=1.5, label='Total')
            # ax.plot(nu_energy_, self.f235u * u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            # ax.plot(nu_energy_, self.f239pu * pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            # ax.plot(nu_energy_, self.f238u * u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            # ax.plot(nu_energy_, self.f241pu * pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.plot(nu_energy_, u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            ax.plot(nu_energy_, pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            ax.plot(nu_energy_, u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            ax.plot(nu_energy_, pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.legend()
            ax.grid(alpha=0.65)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (H+M)')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return self.iso_spectrum

    def isotopic_spectrum_DYB(self, nu_energy_, plot_this=False):

        self.which_isospectrum = 'DYB'
        print("\nUsing DYB isotopic spectra (default)")

        ### params taken from Mueller PRC 83 (2011) for 238U and Huber PRC 84 (2011) for 241Pu
        params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
        params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        f235_dyb = 0.564
        f239_dyb = 0.304
        f238_dyb = 0.076
        f241_dyb = 0.056

        df_235 = self.fiss_frac_235u - f235_dyb
        df_239 = self.fiss_frac_239pu - f239_dyb
        df_238 = self.fiss_frac_238u - f238_dyb
        df_241 = self.fiss_frac_241pu - f241_dyb

        ### unfolded spectra from DYB, arXiv:2102.04614
        unfolded_spectrum = pd.read_csv("Inputs/total_unfolded_DYB.txt", sep="\t",
                                        names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=0)
        unfolded_u235 = pd.read_csv("Inputs/u235_unfolded_DYB.txt", sep="\t",
                                    names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=0)
        unfolded_pu_combo = pd.read_csv("Inputs/pu_combo_unfolded_DYB.txt", sep="\t",
                                        names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=0)

        s_total = interp1d(unfolded_spectrum["bin_center"], unfolded_spectrum["isotopic_spectrum"])
        s_235 = interp1d(unfolded_u235["bin_center"], unfolded_u235["isotopic_spectrum"])
        s_combo = interp1d(unfolded_pu_combo["bin_center"], unfolded_pu_combo["isotopic_spectrum"])

        self.iso_spectrum = s_total(nu_energy_) + df_235 * s_235(nu_energy_) + df_239 * s_combo(nu_energy_) \
                            + df_238 * u238 + (df_241 - 0.183*df_239) * pu241

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.iso_spectrum, 'k', linewidth=1.5, label='Total')
            ax.legend()
            ax.grid(alpha=0.65)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (DYB)')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return self.iso_spectrum

    def reactor_spectrum(self, nu_energy_, which_isospectrum='HM', plot_this=False):

        const = 6.241509e21

        if self.which_isospectrum != which_isospectrum:
            if which_isospectrum == 'V':
                self.isotopic_spectrum_vogel(nu_energy_)
            elif which_isospectrum == 'HM':
                self.isotopic_spectrum_hubermueller(nu_energy_)
            elif which_isospectrum == 'DYB':
                self.isotopic_spectrum_DYB(nu_energy_)
            else:
                print("\nError: only 'V', 'VB' or 'DYB' are accepted values for which_isospectrum argument, "
                      "in reactor_spectrum function, ReactorSpectrum class.")
                sys.exit()

        en_per_fiss = self.fiss_frac_235u * self.fiss_en_235u + self.fiss_frac_239pu * self.fiss_en_239pu \
                      + self.fiss_frac_238u * self.fiss_en_238u + self.fiss_frac_241pu * self.fiss_en_241pu

        self.react_spectrum = self.thermal_power / en_per_fiss * self.iso_spectrum * const

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.react_spectrum, 'k', linewidth=1.5, label='Reactor spectrum')
            ax.legend()
            ax.grid(alpha=0.65)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\Phi_{\nu}$ [$\text{N}_{\nu}/\si{\s}/\si{\MeV}$]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return self.react_spectrum

    def reactor_flux_no_osc(self, nu_energy_, which_isospectrum='HM',
                            bool_snf=False, bool_noneq=False, plot_this=False):

        den = 4. * math.pi * np.power(self.baseline*1.e5, 2)

        if self.which_isospectrum != which_isospectrum:
            if which_isospectrum == 'V':
                self.reactor_spectrum(nu_energy_, which_isospectrum=which_isospectrum)
            elif which_isospectrum == 'HM':
                self.reactor_spectrum(nu_energy_, which_isospectrum=which_isospectrum)
            elif which_isospectrum == 'DYB':
                self.reactor_spectrum(nu_energy_, which_isospectrum=which_isospectrum)
            else:
                print("\nError: only 'V', 'VB' or 'DYB' are accepted values for which_isospectrum argument, "
                      "in reactor_flux_no_osc function, ReactorSpectrum class.")
                sys.exit()

        self.react_flux = self.react_spectrum / den

        if bool_snf:
            print("\nAdding SNF contribution")
            if not np.any(self.snf):
                print('Reading SNF from file')
                self.get_snf_ratio(nu_energy_)
            self.react_flux = self.react_flux + self.react_flux * self.snf

        if bool_noneq:
            print("\nAdding NonEq contribution")
            if not np.any(self.noneq):
                print('Reading NonEq from file')
                self.get_noneq_ratio(nu_energy_)
            self.react_flux = self.react_flux + self.noneq * self.react_flux

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(nu_energy_, self.react_flux, 'k', linewidth=1.5, label='Reactor flux')
            ax.legend()
            ax.grid(alpha=0.65)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\Phi_{\nu}$ [$\text{N}_{\nu}/\si{\s}/\si{\MeV}/\si{\centi\m\squared}$]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return self.react_flux

    ### TODO: move to DetectorResponse class

    def eval_n_protons(self):

        target_mass = self.inputs_json["detector"]["mass"] * 1000 * 1000.
        h_mass = self.inputs_json["detector"]["m_H"] * 1.660539066e-27
        h_fraction = self.inputs_json["detector"]["f_H"]
        h1_abundance = self.inputs_json["detector"]["alpha_H"]
        self.proton_number = target_mass * h_fraction * h1_abundance / h_mass

        return self.proton_number

    ### cross section from Strumia and Vissani - common inputs
    def cross_section_sv(self, nu_energy_):

        self.which_xsec = 'SV'
        print("\nUsing Strumia Vissani cross section from common inputs (default)")

        if self.proton_number == 0.:
            self.eval_n_protons()

        input_ = pd.read_csv("Inputs/IBDXsec_StrumiaVissani.txt", sep="\t",
                             names=["nu_energy", "cross_section"], header=None)

        f_appo = interp1d(input_["nu_energy"], input_["cross_section"])
        self.x_sec = f_appo(nu_energy_)
        self.x_sec_np = self.x_sec * self.proton_number

        return self.x_sec_np

    ### cross section from Vogel and Beacom - common inputs
    def cross_section_vb(self, nu_energy_):

        self.which_xsec = 'VB'
        print("\nUsing Vogel Beacom cross section from common inputs")

        if self.proton_number == 0.:
            self.eval_n_protons()

        input_ = pd.read_csv("Inputs/IBDXsec_VogelBeacom_DYB.txt", sep="\t",
                             names=["nu_energy", "cross_section"], header=None)

        f_appo = interp1d(input_["nu_energy"], input_["cross_section"])
        self.x_sec = f_appo(nu_energy_)
        self.x_sec_np = self.x_sec * self.proton_number

        return self.x_sec_np

    ### cross section from Strumia, Vissani, https://arxiv.org/abs/astro-ph/0302055, eq. (25)
    def cross_section(self, nu_energy_, plot_this=False):

        self.which_xsec = 'std'

        alpha = -0.07056
        beta = 0.02018
        gamma = -0.001953
        delta = 1.293  # MeV, mass(n)-mass(p)
        m_e = 0.511  # MeV
        const = 10 ** (-43)  # cm^2

        positron_energy = np.subtract(nu_energy_, delta)  # positron's energy

        appo = np.power(positron_energy, 2) - m_e ** 2
        p_e = np.sqrt(appo)  # positron's momentum

        appo_exp = alpha + beta * np.log(nu_energy_) + gamma * np.power(np.log(nu_energy_), 3)
        energy_exp = np.power(nu_energy_, appo_exp)

        self.x_sec = const * p_e * positron_energy * energy_exp

        if self.proton_number == 0.:
            self.eval_n_protons()

        self.x_sec_np = self.x_sec * self.proton_number

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(nu_energy_, self.x_sec*self.proton_number, 'k', linewidth=1.5, label='IBD cross section')
            ax.grid(alpha=0.65)
            ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\sigma_{\text{IBD}} \times N_P$ [\si{\centi\meter\squared}]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/cross_section.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/cross_section.pdf')

        return self.x_sec_np

    # TODO: add self.bool_snf and self.bool_noneq
    def antinu_spectrum_no_osc(self, nu_energy_, which_xsec='SV', which_isospectrum='HM',
                               bool_snf=False, bool_noneq=False, plot_this=False):

        if self.which_isospectrum != which_isospectrum or bool_snf or bool_noneq:
            if which_isospectrum == 'V':
                self.reactor_flux_no_osc(nu_energy_, which_isospectrum=which_isospectrum,
                                         bool_snf=bool_snf, bool_noneq=bool_noneq)
            elif which_isospectrum == 'HM':
                self.reactor_flux_no_osc(nu_energy_, which_isospectrum=which_isospectrum,
                                         bool_snf=bool_snf, bool_noneq=bool_noneq)
            elif which_isospectrum == 'DYB':
                self.reactor_flux_no_osc(nu_energy_, which_isospectrum=which_isospectrum,
                                         bool_snf=bool_snf, bool_noneq=bool_noneq)
            else:
                print("\nError: only 'V', 'VB' or 'DYB' are accepted values for which_isospectrum argument, "
                      "in antinu_spectrum_no_osc function, ReactorSpectrum class.")
                sys.exit()

        if self.which_xsec != which_xsec:
            if which_xsec == 'SV':
                self.cross_section_sv(nu_energy_)
            elif which_xsec == 'VB':
                self.cross_section_vb(nu_energy_)
            else:
                print("\nError: only 'SV' or 'VB' are accepted values for which_xsec argument, "
                      "in antinu_spectrum_nu_osc function, ReactorSpectrum class.")
                sys.exit()

        self.spectrum_unosc = self.react_flux * self.x_sec_np

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(nu_energy_, self.spectrum_unosc, 'k', linewidth=1.5, label='spectrum')  # not normalized spectrum
            ax.grid(alpha=0.65)
            ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$S_{\bar{\nu}}$ [N$_{\nu}$/\si{\MeV}/\si{s}]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # plt.savefig('SpectrumPlots/unoscillated_spectrum.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/unoscillated_spectrum.pdf')

        return self.spectrum_unosc

