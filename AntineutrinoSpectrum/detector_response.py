import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate

# TODO
# - add non linearity
# - initialize with json file? --> DONE
# - remove plotting parts


class DetectorResponse:

    def __init__(self, inputs_json_):

        self.smeared_spectrum = 0.

        self.a = inputs_json_["energy_resolution"]["a"]
        self.sigma_a = inputs_json_["energy_resolution"]["sigma_a"]
        self.b = inputs_json_["energy_resolution"]["b"]
        self.sigma_b = inputs_json_["energy_resolution"]["sigma_b"]
        self.c = inputs_json_["energy_resolution"]["c"]
        self.sigma_c = inputs_json_["energy_resolution"]["sigma_c"]

    @staticmethod
    def gaussian(x_, sigma_):
        const = np.sqrt(2. * math.pi)
        appo = 1. / const / sigma_ * np.exp(- np.power(x_, 2) / 2. / (sigma_ ** 2))
        return appo

    def set_a(self, val_):
        self.a = val_

    def set_sigma_a(self, val_):
        self.sigma_a = val_

    def set_b(self, val_):
        self.b = val_

    def set_sigma_b(self, val_):
        self.sigma_b = val_

    def set_c(self, val_):
        self.c = val_

    def set_sigma_c(self, val_):
        self.sigma_c = val_

    def get_resol_params(self):
        return self.a, self.b, self.c

    def get_resol_params_sigmas(self):
        return self.sigma_a, self.sigma_b, self.sigma_c

    ### convolution of a function f with a gaussian in a given range E
    ### given both initial and final energies

    def gaussian_smearing_fixed_sigma(self, f_, initial_energy_, final_energy_, sigma_,
                                      plot_this=False, plot_start=False):
        """ Evaluates the numerical convolution of the function f with a Gaussian with given fixed or variable width.

        :param f_: Function (samples) to be convolved with the Gaussian
        :param initial_energy_: Initial energy variable, it is usually Edep = Enu - 0.8
        :param final_energy_: Final energy variable, it is usually Evis, or also Erec
        :param sigma_: Fixed value of the width of the Gaussian
        :param plot_this: If True, plots the result of the convolution as a function of E_fin
        :param plot_start: If True, plots the initial function f as a function of E_in

        :return: smeared_spectrum: is the result of the numerical convolution
        """

        # gaussian with fixed given width
        self.smeared_spectrum = np.zeros(len(final_energy_))
        n = 0
        for E0 in final_energy_:
            appo = self.gaussian(initial_energy_ - E0, sigma_)
            prod = appo * f_
            self.smeared_spectrum[n] = integrate.simps(prod, initial_energy_)
            n += 1

        if plot_this:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(final_energy_, self.smeared_spectrum, 'b', linewidth=1, label='Convolved spectrum')
            ax1.set_xlabel(r'$E_{vis}$ [\si{MeV}]')
            ax1.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax1.set_ylim(-0.005, 0.095)
            ax1.set_title(r'Numerical convolution with a Gaussian' + '\nwith fixed width')
            ax1.grid()
            ax1.legend()

        if plot_start:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(initial_energy_, f_, 'b', linewidth=1, label='Unconvolved spectrum')
            ax.set_xlabel(r'$E_{dep}$ [\si{MeV}]')
            ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax.set_ylim(-0.005, 0.095)
            ax.set_title(r'Starting spectrum')
            ax.grid()
            ax.legend()

        return self.smeared_spectrum

    def gaussian_smearing_abc(self, f_, initial_energy_, final_energy_, plot_this=False, plot_start=False):
        """ Evaluates the numerical convolution of the function f with a Gaussian with given fixed or variable width.

        :param f_: Function (samples) to be convolved with the Gaussian
        :param initial_energy_: Initial energy variable, it is usually Edep = Enu - 0.8
        :param final_energy_: Final energy variable, it is usually Evis, or also Erec
        :param plot_this: If True, plots the result of the convolution as a function of E_fin
        :param plot_start: If True, plots the initial function f as a function of E_in

        :return: smeared_spectrum: is the result of the numerical convolution
        """

        # gaussian with variable width, set by a (stochastic term) and b (constant term)
        rad = self.a * self.a / initial_energy_ + self.b * self.b + self.c * self.c / np.power(initial_energy_, 2)
        sigma_energy = np.sqrt(rad) * initial_energy_

        self.smeared_spectrum = np.zeros(len(final_energy_))
        n = 0
        for E0 in final_energy_:
            appo = self.gaussian(initial_energy_ - E0, sigma_energy)
            prod = appo * f_
            self.smeared_spectrum[n] = integrate.simps(prod, initial_energy_)
            n += 1

        if plot_this:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(final_energy_, self.smeared_spectrum, 'b', linewidth=1, label='Convolved spectrum')
            ax1.set_xlabel(r'$E_{vis}$ [\si{MeV}]')
            ax1.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax1.set_ylim(-0.005, 0.095)
            ax1.set_title(r'Numerical convolution with a Gaussian' + '\nwith variable width')
            ax1.grid()
            ax1.legend()

        if plot_start:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(initial_energy_, f_, 'b', linewidth=1, label='Unconvolved spectrum')
            ax.set_xlabel(r'$E_{dep}$ [\si{MeV}]')
            ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax.set_ylim(-0.005, 0.095)
            ax.set_title(r'Starting spectrum')
            ax.grid()
            ax.legend()

        return self.smeared_spectrum

