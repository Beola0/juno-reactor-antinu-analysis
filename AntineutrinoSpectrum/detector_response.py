import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate

# TODO
# - add non linearity
# - initialize with json file?

class DetectorResponse:

    def __init__(self):

        self.conv_num = 0.

    @staticmethod
    def gaussian(x, sigma):
        const = np.sqrt(2. * math.pi)
        appo = 1. / const / sigma * np.exp(- np.power(x, 2) / 2. / (sigma ** 2))
        return appo

    ### convolution of a function f with a gaussian in a given range E
    ### given both initial and final energies
    def gaussian_smearing(self, f, initial_energy, final_energy, sigma=None, a=None, b=None, c=None,
                          plot_this=False, plot_start=False):
        """ Evaluates the numerical convolution of the function f with a Gaussian with given fixed or variable width.

        :param f: Function (samples) to be convolved with the Gaussian
        :param initial_energy: Initial energy variable, it is usually Edep = Enu - 0.8
        :param final_energy: Final energy variable, it is usually Evis, or also Erec
        :param sigma: Fixed value of the width of the Gaussian
        :param a: Stochastic term used to evaluate the variable width of the Gaussian
        :param b: Constant term used to evaluate the variable width of the Gaussian
        :param c: PMT dark noise contribution to the variable width of the Gaussian
        :param plot_this: If True, plots the result of the convolution as a function of E_fin
        :param plot_start: If True, plots the initial function f as a function of E_in

        :return: conv_num: is the result of the numerical convolution
        """

        if sigma is not None and (a is not None or b is not None or c is not None):
            sigma = None
            print("Both 'sigma' and 'a-b' have been given. 'sigma' will be ignored, 'a-b' will be used.")

        # gaussian with fixed or given width
        if sigma is not None:

            self.conv_num = np.zeros(len(final_energy))
            n = 0
            for E0 in final_energy:
                appo = self.gaussian(initial_energy - E0, sigma)
                prod = appo * f
                self.conv_num[n] = integrate.simps(prod, initial_energy)
                n += 1

        # gaussian with variable width, set by a (stochastic term) and b (constant term)
        if a is not None or b is not None or c is not None:

            rad = a * a / initial_energy + b * b + c * c / np.power(initial_energy, 2)
            sigma_energy = np.sqrt(rad) * initial_energy

            self.conv_num = np.zeros(len(final_energy))
            n = 0
            for E0 in final_energy:
                appo = self.gaussian(initial_energy - E0, sigma_energy)
                prod = appo * f
                self.conv_num[n] = integrate.simps(prod, initial_energy)
                n += 1

        if plot_this:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(final_energy, self.conv_num, 'b', linewidth=1, label='Convolved spectrum')
            ax1.set_xlabel(r'$E_{vis}$ [\si{MeV}]')
            ax1.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax1.set_ylim(-0.005, 0.095)
            ax1.set_title(r'Numerical convolution with a Gaussian' + '\nwith variable width')
            ax1.grid()
            ax1.legend()

        if plot_start:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(initial_energy, f, 'b', linewidth=1, label='Unconvolved spectrum')
            ax.set_xlabel(r'$E_{dep}$ [\si{MeV}]')
            ax.set_ylabel(r'$N(\bar{\nu})$ [arb. unit]')
            ax.set_ylim(-0.005, 0.095)
            ax.set_title(r'Starting spectrum')
            ax.grid()
            ax.legend()

        return self.conv_num
