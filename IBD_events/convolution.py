import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate


class Convolution:

    def __init__(self):
        self.conv_np = 0
        self.conv_num = 0

    @staticmethod
    def gaussian(x, sigma):
        const = math.sqrt(2 * math.pi)
        appo = 1 / const / sigma * np.exp(- np.power(x, 2) / 2. / (sigma ** 2))
        return appo

    ### convolution of a function f with a gaussian in a given range E

    ### using np.convolve()
    def np_conv(self, f, E, sigma=None, a=None, b=None, plot_this=False, plot_start=False):

        Evis = E - 0.8

        if sigma is not None:
            g = self.gaussian(Evis - np.mean(Evis), sigma)
            self.conv_np = np.convolve(f, g, mode='same')

        if a is not None or b is not None:
            rad = a ** 2 / Evis + b ** 2
            sigma_Evis = np.sqrt(rad) * Evis

            g = self.gaussian(Evis - np.mean(Evis), sigma_Evis)
            self.conv_np = np.convolve(f, g, mode='same')

        if plot_this:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(Evis, self.conv_np, 'b', linewidth=1, label='Convolved spectrum')
            ax1.set_xlabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
            ax1.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
            ax1.set_title(r'Convolution (numpy) with a Gaussian' + '\nwith variable width')
            ax1.grid()
            ax1.legend()

        if plot_start:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(Evis, f, 'b', linewidth=1, label='Unconvolved spectrum')
            ax.set_xlabel(r'$\text{E}_{\text{dep}}$ [\si{MeV}]')
            ax.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
            ax.set_title(r'Starting spectrum')
            ax.grid()
            ax.legend()

        return self.conv_np

    ### using a numerical method: numerical convolution (over Evis)
    ### need (sigma) OR (a or b)
    def numerical_conv_old(self, f, E, sigma=None, a=None, b=None, plot_this=False, plot_start=False):

        # E is Evis or Erec
        Evis = E - 0.8

        if sigma is not None and (a is not None or b is not None):
            sigma = None
            print("Both 'sigma' and 'a-b' have been given. 'sigma' will be ignored, 'a-b' will be used.")

        # gaussian with fixed or given width
        if sigma is not None:

            self.conv_num = np.zeros(len(Evis))
            n = 0
            for E0 in Evis:
                appo = self.gaussian(Evis - E0, sigma)
                prod = appo * f
                self.conv_num[n] = integrate.simps(prod, Evis)
                n += 1

        # gaussian with variable width, set by a (stochastic term) and b (constant term)
        if a is not None or b is not None:

            rad = a ** 2 / Evis + b ** 2
            sigma_Evis = np.sqrt(rad) * Evis

            self.conv_num = np.zeros(len(Evis))
            n = 0
            for E0 in Evis:
                appo = self.gaussian(Evis - E0, sigma_Evis)
                prod = appo * f
                self.conv_num[n] = integrate.simps(prod, Evis)
                n += 1

        if plot_this:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(Evis, self.conv_num, 'b', linewidth=1, label='Convolved spectrum')
            ax1.set_xlabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
            ax1.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
            ax1.set_ylim(-0.005, 0.095)
            ax1.set_title(r'Numerical convolution with a Gaussian' + '\nwith variable width')
            ax1.grid()
            ax1.legend()

        if plot_start:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(Evis, f, 'b', linewidth=1, label='Unconvolved spectrum')
            ax.set_xlabel(r'$\text{E}_{\text{dep}}$ [\si{MeV}]')
            ax.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
            ax.set_ylim(-0.005, 0.095)
            ax.set_title(r'Starting spectrum')
            ax.grid()
            ax.legend()

        return self.conv_num

    ### given both initial and final energies
    def numerical_conv_NEW(self, f, E_in, E_fin, sigma=None, a=None, b=None, plot_this=False, plot_start=False):
        """ Evaluates the numerical convolution of the function f with a Gaussian with given fixed or variable width.

        :param f: Function (samples) to be convolved with the Gaussian
        :param E_in: Initial energy variable, E_in is usually Edep = Enu - 0.8
        :param E_fin: Final energy variable, E_fin is usually Evis, or also Erec
        :param sigma: Fixed value of the width of the Gaussian
        :param a: Stochastic term used to evaluate the variable width of the Gaussian
        :param b: Constant term used to evaluate the variable width of the Gaussian
        :param plot_this: If True, plots the result of the convolution as a function of E_fin
        :param plot_start: If True, plots the initial function f as a function of E_in

        :return: conv_num: is the result of the numerical convolution
        """

        # E_in is Edep = E_nu - 0.8
        # E_fin is Evis or Erec

        if sigma is not None and (a is not None or b is not None):
            sigma = None
            print("Both 'sigma' and 'a-b' have been given. 'sigma' will be ignored, 'a-b' will be used.")

        # gaussian with fixed or given width
        if sigma is not None:

            self.conv_num = np.zeros(len(E_fin))
            n = 0
            for E0 in E_fin:
                appo = self.gaussian(E_in - E0, sigma)
                prod = appo * f
                self.conv_num[n] = integrate.simps(prod, E_in)
                n += 1

        # gaussian with variable width, set by a (stochastic term) and b (constant term)
        if a is not None or b is not None:

            rad = a ** 2 / E_in + b ** 2
            sigma_E = np.sqrt(rad) * E_in

            self.conv_num = np.zeros(len(E_fin))
            n = 0
            for E0 in E_fin:
                appo = self.gaussian(E_in - E0, sigma_E)
                prod = appo * f
                self.conv_num[n] = integrate.simps(prod, E_in)
                n += 1

        if plot_this:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(E_fin, self.conv_num, 'b', linewidth=1, label='Convolved spectrum')
            ax1.set_xlabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
            ax1.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
            ax1.set_ylim(-0.005, 0.095)
            ax1.set_title(r'Numerical convolution with a Gaussian' + '\nwith variable width')
            ax1.grid()
            ax1.legend()

        if plot_start:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(E_in, f, 'b', linewidth=1, label='Unconvolved spectrum')
            ax.set_xlabel(r'$\text{E}_{\text{dep}}$ [\si{MeV}]')
            ax.set_ylabel(r'N($\bar{\nu}$) [arb. unit]')
            ax.set_ylim(-0.005, 0.095)
            ax.set_title(r'Starting spectrum')
            ax.grid()
            ax.legend()

        return self.conv_num
