import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate
import pandas as pd
from scipy.interpolate import interp1d

# TODO
# - add non linearity
# - initialize with json file? --> DONE
# - remove plotting parts
# - add nuisance parameters for a, b, and c


class DetectorResponse:

    def __init__(self, inputs_json_):

        self.smeared_spectrum = 0.

        self.a = inputs_json_["energy_resolution"]["a"]
        self.sigma_a = inputs_json_["energy_resolution"]["sigma_a"]
        self.b = inputs_json_["energy_resolution"]["b"]
        self.sigma_b = inputs_json_["energy_resolution"]["sigma_b"]
        self.c = inputs_json_["energy_resolution"]["c"]
        self.sigma_c = inputs_json_["energy_resolution"]["sigma_c"]

        self.nl_nominal = 0.
        self.nl_pull0 = 0.
        self.nl_pull1 = 0.
        self.nl_pull2 = 0.
        self.nl_pull3 = 0.
        self.alpha_pull0 = 0.
        self.alpha_pull1 = 0.
        self.alpha_pull2 = 0.
        self.alpha_pull3 = 0.

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

    def set_alphas_pull(self, a0_, a1_, a2_, a3_):
        self.alpha_pull0 = a0_
        self.alpha_pull1 = a1_
        self.alpha_pull2 = a2_
        self.alpha_pull3 = a3_

    def get_alphas_pull(self):
        return self.alpha_pull0, self.alpha_pull1, self.alpha_pull2, self.alpha_pull3

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

    def get_nl_curves(self, dep_energy_):

        input_nom = pd.read_csv("Inputs/positronScintNL.txt", sep="\t",
                                names=["dep_energy", "nl_nominal"], header=None)
        input_pull0 = pd.read_csv("Inputs/positronScintNLpull0.txt", sep="\t",
                                  names=["dep_energy", "nl_pull0"], header=None)
        input_pull1 = pd.read_csv("Inputs/positronScintNLpull1.txt", sep="\t",
                                  names=["dep_energy", "nl_pull1"], header=None)
        input_pull2 = pd.read_csv("Inputs/positronScintNLpull2.txt", sep="\t",
                                  names=["dep_energy", "nl_pull2"], header=None)
        input_pull3 = pd.read_csv("Inputs/positronScintNLpull3.txt", sep="\t",
                                  names=["dep_energy", "nl_pull3"], header=None)

        f_appo = interp1d(input_nom["dep_energy"], input_nom["nl_nominal"])
        self.nl_nominal = f_appo(dep_energy_)
        f_appo = interp1d(input_pull0["dep_energy"], input_pull0["nl_pull0"])
        self.nl_pull0 = f_appo(dep_energy_)
        f_appo = interp1d(input_pull1["dep_energy"], input_pull1["nl_pull1"])
        self.nl_pull1 = f_appo(dep_energy_)
        f_appo = interp1d(input_pull2["dep_energy"], input_pull2["nl_pull2"])
        self.nl_pull2 = f_appo(dep_energy_)
        f_appo = interp1d(input_pull3["dep_energy"], input_pull3["nl_pull3"])
        self.nl_pull3 = f_appo(dep_energy_)

        return self.nl_nominal, self.nl_pull0, self.nl_pull1, self.nl_pull2, self.nl_pull3

    def eval_non_linearity(self, dep_energy_):

        if not np.any(self.nl_nominal):
            self.get_nl_curves(dep_energy_)

        appo0 = self.alpha_pull0 * (self.nl_pull0 - self.nl_nominal)
        appo1 = self.alpha_pull1 * (self.nl_pull1 - self.nl_nominal)
        appo2 = self.alpha_pull2 * (self.nl_pull2 - self.nl_nominal)
        appo3 = self.alpha_pull3 * (self.nl_pull3 - self.nl_nominal)

        return self.nl_nominal + appo0 + appo1 + appo2 + appo3
