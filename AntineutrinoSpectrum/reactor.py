import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from scipy.integrate import simps

# TODO:
# - initialize class with .json file with fission fractions --> DONE
# - create spectrum from inputs, like DYB spectrum
# - remove part for plotting
# - add methods to change fission fractions --> DONE


class ReactorSpectrum:

    def __init__(self, inputs_json_):

        self.f235u = inputs_json_["fission_fractions"]["235U"]
        self.f239pu = inputs_json_["fission_fractions"]["239Pu"]
        self.f238u = inputs_json_["fission_fractions"]["238U"]
        self.f241pu = inputs_json_["fission_fractions"]["241Pu"]

        self.tot_flux = 0.
        self.x_sec = 0.
        self.spectrum_un = 0.
        self.norm_spectrum_un = 0.
        
    def set_fission_fractions(self, f235u_, f239pu_, f238u_, f241pu_):
        self.f235u = f235u_
        self.f239pu = f239pu_
        self.f238u = f238u_
        self.f241pu = f241pu_
        
    def set_f235u(self, f235u_):
        self.f235u = f235u_
        
    def set_f238u(self, f238u_):
        self.f238u = f238u_

    def set_f239pu(self, f239pu_):
        self.f239pu = f239pu_
        
    def set_f241pu(self, f241pu_):
        self.f241pu = f241pu_
        
    def get_f235u(self):
        return self.f235u
        
    def get_f238u(self):
        return self.f238u

    def get_f239pu(self):
        return self.f239pu
        
    def get_f241pu(self):
        return self.f241pu

    @staticmethod
    def reactor_exp(x_, a_, b_, c_):
        res = np.exp(a_ - b_ * x_ - c_ * (x_ ** 2))
        return res

    def flux(self, nu_energy_, plot_this=False):
        """
        Evaluate the reactor antineutrino flux.

        For given input energies, it evaluates the total reactor flux and the single contributions of the four isotopes
        of the reactor fuel

        Parameters
        ----------
        nu_energy_ : numpy array
            Antineutrino energies.
        plot_this : bool, optional
            A flag used to plot the contributions of the four core isotopes to the flux and the total flux, as functions
            of energy (default is False).

        Returns
        -------
        tot_flux : numpy array
            Total flux

        References
        ----------
        http://inspirehep.net/record/25814/ and https://arxiv.org/abs/0807.3203, eq. (2)
        """
        u235 = self.reactor_exp(nu_energy_, 0.870, 0.160, 0.091)
        pu239 = self.reactor_exp(nu_energy_, 0.896, 0.239, 0.0981)
        u238 = self.reactor_exp(nu_energy_, 0.976, 0.162, 0.0790)
        pu241 = self.reactor_exp(nu_energy_, 0.793, 0.080, 0.1085)

        self.tot_flux = self.f235u * u235 + self.f239pu * pu239 + self.f238u * u238 + self.f241pu * pu241

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.tot_flux, 'k', linewidth=1.5, label='Total')
            ax.plot(nu_energy_, self.f235u * u235, 'b--', linewidth=1.5, label=r'$^{235}$U')
            ax.plot(nu_energy_, self.f239pu * pu239, 'r-.', linewidth=1.5, label=r'$^{239}$Pu')
            ax.plot(nu_energy_, self.f238u * u238, 'g:', linewidth=1.5, label=r'$^{238}$U')
            ax.plot(nu_energy_, self.f241pu * pu241, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.legend()
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [$\si{MeV}$]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\Phi_{\nu}$ [arb. unit]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/flux.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/flux.pdf')

        return self.tot_flux

    def cross_section(self, nu_energy_, plot_this=False):
        """
        Evaluate the IBD cross section.

        For given input energies, it evaluates the IBD cross section.

        Parameters
        ----------
        nu_energy_ : numpy array
            Antineutrino energies.
        plot_this : bool, optional
            A flag used to plot the IBD cross section as a function of energy (default is False)

        Returns
        -------
        x_sec : numpy array
            IBD cross section

        References
        ----------
        Strumia, Vissani, https://arxiv.org/abs/astro-ph/0302055, eq. (25)
        """
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

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.09, right=0.96, top=0.95)
            ax.plot(nu_energy_, self.x_sec, 'k', linewidth=1.5, label='IBD cross section')
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$\sigma_{\text{IBD}}$ [\si{\centi\meter\squared}]')
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/cross_section.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/cross_section.pdf')

        return self.x_sec

    def unosc_spectrum(self, nu_energy_, plot_this=False):
        """
        Evaluate the unoscillated reactor spectrum.

        The spectrum is obtained by multiplying the total reactor flux and the IBD cross section.

        Parameters
        ----------
        nu_energy_ : numpy array
            Antineutrino energies.
        plot_this : bool, optional
            A flag used to plot the unoscillated reactor spectrum as a function of the neutrino energy
            (default is False)

        Returns
        -------
        norm_spectrum_un : numpy array
            normalized unoscillated spectrum
        """
        self.flux(nu_energy_, plot_this=False)
        self.cross_section(nu_energy_, plot_this=False)

        self.spectrum_un = self.tot_flux * self.x_sec
        integral = simps(self.spectrum_un, nu_energy_)
        self.norm_spectrum_un = self.spectrum_un / integral

        if plot_this:
            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.12, right=0.96, top=0.95)
            # ax.plot(nu_energy_,self.spectrum_un,'b',linewidth=1,label='spectrum') # not normalized spectrum
            ax.plot(nu_energy_, self.norm_spectrum_un, 'k', linewidth=1.5, label='spectrum')  # normalized spectrum
            ax.grid(alpha=0.45)
            ax.set_xlabel(r'$E_{\nu}$ [\si{MeV}]')
            ax.set_xlim(1.5, 10.5)
            ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax.set_ylim(-0.015, 0.33)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(loc1)
            ax.tick_params('both', direction='out', which='both')
            # plt.savefig('SpectrumPlots/unoscillated_spectrum.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/unoscillated_spectrum.pdf')

        return self.norm_spectrum_un
