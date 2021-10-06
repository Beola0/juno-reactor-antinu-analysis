import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.interpolate import interp1d
import math

# TODO:
# - initialize class with .json file with fission fractions --> DONE
# - create spectrum from inputs, like DYB spectrum
# - remove part for plotting
# - add methods to change fission fractions --> DONE
# - move IBD part to DetectorResponse


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

        self.tot_flux = 0.
        self.x_sec = 0.
        self.spectrum_unosc = 0.
        # self.norm_spectrum_un = 0.
        self.proton_number = 0.
        
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

    def isotopic_spectrum_vogel(self, nu_energy_, plot_this=False):

        ### params taken from Vogel, Engel, PRD 39-11 pp 3378, 1989
        params_u235 = [0.870, -0.160, -0.091]
        params_pu239 = [0.896, -0.239, -0.0981]
        params_u238 = [0.976, -0.162, -0.0790]
        params_pu241 = [0.793, -0.080, -0.1085]
        u235 = self.isotopic_spectrum_exp(nu_energy_, params_u235)
        pu239 = self.isotopic_spectrum_exp(nu_energy_, params_pu239)
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        appo = self.fiss_frac_235u * u235 + self.fiss_frac_239pu * pu239 \
               + self.fiss_frac_238u * u238 + self.fiss_frac_241pu * pu241

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, appo, 'k', linewidth=1.5, label='Total')
            # ax.plot(nu_energy_, self.f235u * u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            # ax.plot(nu_energy_, self.f239pu * pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            # ax.plot(nu_energy_, self.f238u * u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            # ax.plot(nu_energy_, self.f241pu * pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.plot(nu_energy_, u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            ax.plot(nu_energy_, pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            ax.plot(nu_energy_, u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            ax.plot(nu_energy_, pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.legend()
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (Vogel)')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return appo

    def isotopic_spectrum_hubermueller(self, nu_energy_, plot_this=False):

        ### params taken from Mueller PRC 83 (2011) for 238U and Huber PRC 84 (2011) for others
        params_u235 = [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]
        params_pu239 = [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]
        params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
        params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]
        u235 = self.isotopic_spectrum_exp(nu_energy_, params_u235)
        pu239 = self.isotopic_spectrum_exp(nu_energy_, params_pu239)
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        appo = self.fiss_frac_235u * u235 + self.fiss_frac_239pu * pu239 \
               + self.fiss_frac_238u * u238 + self.fiss_frac_241pu * pu241

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, appo, 'k', linewidth=1.5, label='Total')
            # ax.plot(nu_energy_, self.f235u * u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            # ax.plot(nu_energy_, self.f239pu * pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            # ax.plot(nu_energy_, self.f238u * u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            # ax.plot(nu_energy_, self.f241pu * pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.plot(nu_energy_, u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            ax.plot(nu_energy_, pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            ax.plot(nu_energy_, u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            ax.plot(nu_energy_, pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.legend()
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (H+M)')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return appo

    def reactor_spectrum(self, nu_energy_, plot_this=False):

        const = 6.241509e21

        tot_iso_spectrum = self.isotopic_spectrum_vogel(nu_energy_)

        en_per_fiss = self.fiss_frac_235u * self.fiss_en_235u + self.fiss_frac_239pu * self.fiss_en_239pu \
                      + self.fiss_frac_238u * self.fiss_en_238u + self.fiss_frac_241pu * self.fiss_en_241pu

        react_spect = self.thermal_power / en_per_fiss * tot_iso_spectrum * const

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, react_spect, 'k', linewidth=1.5, label='Reactor spectrum')
            ax.legend()
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\Phi_{\nu}$ [$\text{N}_{\nu}/\si{\s}/\si{\MeV}$]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return react_spect

    def reactor_flux_no_osc(self, nu_energy_, plot_this=False):

        den = 4. * math.pi * np.power(self.baseline*1.e5, 2)
        react_spect = self.reactor_spectrum(nu_energy_)

        react_flux = react_spect / den

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, react_flux, 'k', linewidth=1.5, label='Reactor flux')
            ax.legend()
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\Phi_{\nu}$ [$\text{N}_{\nu}/\si{\s}/\si{\MeV}/\si{\centi\m\squared}$]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return react_flux

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

        if self.proton_number == 0:
            self.eval_n_protons()

        input_ = pd.read_csv("Inputs/IBDXsec_StrumiaVissani.txt", sep="\t",
                             names=["nu_energy", "cross_section"], header=None)

        f_appo = interp1d(input_["nu_energy"], input_["cross_section"])
        self.x_sec = f_appo(nu_energy_)

        return self.x_sec * self.proton_number

    ### cross section from Vogel and Beacom - common inputs
    def cross_section_vb(self, nu_energy_):

        if self.proton_number == 0:
            self.eval_n_protons()

        input_ = pd.read_csv("Inputs/IBDXsec_VogelBeacom_DYB.txt", sep="\t",
                             names=["nu_energy", "cross_section"], header=None)

        f_appo = interp1d(input_["nu_energy"], input_["cross_section"])
        self.x_sec = f_appo(nu_energy_)

        return self.x_sec * self.proton_number

    ### Strumia, Vissani, https://arxiv.org/abs/astro-ph/0302055, eq. (25)
    def cross_section(self, nu_energy_, plot_this=False):

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

        if self.proton_number == 0:
            self.eval_n_protons()

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.09, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.x_sec*self.proton_number, 'k', linewidth=1.5, label='IBD cross section')
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\sigma_{\text{IBD}} \times N_P$ [\si{\centi\meter\squared}]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/cross_section.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/cross_section.pdf')

        return self.x_sec * self.proton_number

    def antinu_spectrum_no_osc(self, nu_energy_, plot_this=False):

        appo = self.reactor_flux_no_osc(nu_energy_)
        xsec = self.cross_section_sv(nu_energy_)

        self.spectrum_unosc = appo * xsec

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.12, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.spectrum_unosc, 'k', linewidth=1.5, label='spectrum')  # not normalized spectrum
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$S_{\bar{\nu}}$ [N$_{\nu}$/\si{\MeV}/\si{s}]')
            # ax.set_ylim(-0.015, 0.33)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/unoscillated_spectrum.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/unoscillated_spectrum.pdf')

        return self.spectrum_unosc

