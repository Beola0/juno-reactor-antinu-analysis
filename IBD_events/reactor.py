import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps


def reactor_exp(x, a, b, c):
    res = np.exp(a - b * x - c * (x ** 2))
    return res


class ReactorSpectrum:

    def __init__(self):
        # fixed (for now) fission fractions
        self.f235u = 0.58
        self.f239pu = 0.30
        self.f238u = 0.07
        self.f241pu = 0.05

        # average thermal energy per fission
        # self.E235u = 201.7  # [MeV]
        # self.E239pu = 210.0
        # self.E238u = 205.0
        # self.E241pu = 212.4

        self.E235u = 202.36  # [MeV]
        self.E239pu = 211.12
        self.E238u = 205.99
        self.E241pu = 214.26

        self.tot_flux = 0
        self.x_sec = 0
        self.spectrum_un = 0
        self.norm_spectrum_un = 0

    ### this method evaluates the antineutrino flux
    ### for further reference: http://inspirehep.net/record/25814/ and https://arxiv.org/abs/0807.3203, eq. (2)
    def flux(self, E, th_power, plot_this=False):
        u235 = reactor_exp(E, 0.870, 0.160, 0.091)
        pu239 = reactor_exp(E, 0.896, 0.239, 0.0981)
        u238 = reactor_exp(E, 0.976, 0.162, 0.0790)
        pu241 = reactor_exp(E, 0.793, 0.080, 0.1085)

        tot_rate = self.f235u * self.E235u + self.f239pu * self.E239pu + self.f238u * self.E238u + self.f241pu * self.E241pu
        tot_flux = self.f235u * u235 + self.f239pu * pu239 + self.f238u * u238 + self.f241pu * pu241

        self.tot_flux = tot_flux / tot_rate * th_power * 6.24e21

        if plot_this:
            fig = plt.figure()
            # fig.suptitle('Reactor antineutrino flux')
            ax = fig.add_subplot(111)

            ax.plot(E, self.f235u * u235 / tot_rate * th_power * 6.24e21, 'b', linewidth=1.5, label=r'$^{235}$U')
            ax.plot(E, self.f239pu * pu239 / tot_rate * th_power * 6.24e21, 'r', linewidth=1.5, label=r'$^{239}$Pu')
            ax.plot(E, self.f238u * u238 / tot_rate * th_power * 6.24e21, 'g', linewidth=1.5, label=r'$^{238}$U')
            ax.plot(E, self.f241pu * pu241 / tot_rate * th_power * 6.24e21, 'y', linewidth=1.5, label=r'$^{241}$Pu')
            ax.plot(E, self.tot_flux, 'k', linewidth=1.5, label='total')

            ax.legend()
            ax.grid()
            ax.set_xlabel(r'$\text{E}_{\nu}$ [$\si{MeV}$]')
            ax.set_ylabel(r'$\Phi_{\nu}$ [arb. unit]')
            ax.set_title(r'Reactor antineutrino flux')

            plt.savefig('flux.pdf', format='pdf', transparent=True)
            print('\nThe plot has been saved in flux.pdf')

        return self.tot_flux

    ### this method evaluates the IBD cross section
    ### for further reference: Strumia, Vissani, https://arxiv.org/abs/astro-ph/0302055, eq. (25)
    def cross_section(self, E, plot_this=False):  # E is the neutrino's energy
        alpha = -0.07056
        beta = 0.02018
        gamma = -0.001953
        Delta = 1.293  # MeV, mass(n)-mass(p)
        m_e = 0.511  # MeV
        const = 10 ** (-43)  # cm^2

        E_e = np.subtract(E, Delta)  # positron's energy

        appo = np.power(E_e, 2) - m_e ** 2
        p_e = np.sqrt(appo)  # positron's momentum

        appo_exp = alpha + beta * np.log(E) + gamma * np.power(np.log(E), 3)
        E_exp = np.power(E, appo_exp)

        self.x_sec = const * p_e * E_e * E_exp

        if plot_this:
            fig = plt.figure()
            # fig.suptitle('IBD cross section')
            ax = fig.add_subplot(111)

            ax.plot(E, self.x_sec, 'k', linewidth=1.5, label='IBD cross section')
            ax.grid()
            ax.set_xlabel(r'$\text{E}_{\nu}$ [\si{MeV}]')
            ax.set_ylabel(r'$\sigma_{\text{IBD}}$ [\si{\centi\meter\squared}]')
            ax.set_title(r'IBD cross section')

            plt.savefig('cross_section.pdf', format='pdf', transparent=True)
            print('\nThe plot has been saved in cross_section.pdf')

        return self.x_sec

    ### this method combines the flux and the cross section to obtain the reactor's spectrum
    def unosc_spectrum(self, E, plot_this=False):
        self.flux(E, 1., plot_this=False)
        self.cross_section(E, plot_this=False)

        self.spectrum_un = self.tot_flux * self.x_sec
        integral = simps(self.spectrum_un, E)
        self.norm_spectrum_un = self.spectrum_un / integral

        if plot_this:
            fig = plt.figure()
            # fig.suptitle('Unoscillated reactor spectrum')
            ax = fig.add_subplot(111)

            # ax.plot(E,self.spectrum_un,'b',linewidth=1,label='spectrum') # not normalized spectrum
            ax.plot(E, self.norm_spectrum_un, 'k', linewidth=1.5, label='spectrum')  # normalized spectrum
            ax.grid()
            ax.set_xlabel(r'$\text{E}_{\nu}$ [\si{MeV}]')
            ax.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
            ax.set_title(r'Unoscillated reactor spectrum')

            plt.savefig('unoscillated_spectrum.pdf', format='pdf', transparent=True)
            print('\nThe plot has been saved in unoscillated_spectrum.pdf')

        return self.norm_spectrum_un
