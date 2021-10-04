import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import math

# TODO:
# - add methods to change osc params --> DONE
# - divide NO and IO into separate methods + third method for both
# - new method: change baseline --> self.baselines = [52.5]
# - remove part for plotting
# - add methods for osc probability in matter
# - initialize with .json file --> DONE


### survival probability
class OscillationProbability:

    def __init__(self, inputs_json_):
        self.baseline = 52.5  # [km], fixed for now -> can be modified, in class Spectrum

        self.sin2_12 = inputs_json_["oscillationParams"]["sin2_12"]
        self.deltam_21 = inputs_json_["oscillationParams"]["deltam_21"]  # [eV^2]

        # Normal Ordering: 3l = 31
        self.sin2_13_N = inputs_json_["oscillationParams"]["sin2_13_N"]
        self.deltam_3l_N = inputs_json_["oscillationParams"]["deltam_31_N"]  # [eV^2]

        # Inverted Ordering: 3l = 32
        self.sin2_13_I = inputs_json_["oscillationParams"]["sin2_13_I"]
        self.deltam_3l_I = inputs_json_["oscillationParams"]["deltam_32_I"]  # [eV^2]

        # Matter density for matter effect
        self.rho = inputs_json_["matter_effect"]["matter_density"]
        self.sigma_abs = inputs_json_["matter_effect"]["abs_sigma"]
        self.sigma_rel = inputs_json_["matter_effect"]["rel_sigma"]

        self.vacuum_prob_N = 0
        self.vacuum_prob_N_E = 0
        self.vacuum_prob_I = 0
        self.vacuum_prob_I_E = 0
        self.matter_prob_N_E = 0
        self.matter_prob_I_E = 0
        self.jhep_prob_N = 0
        self.jhep_prob_E_N = 0
        self.jhep_prob_I = 0
        self.jhep_prob_E_I = 0

    def set_rho(self, rho_, sigma_, sigma_rel_):
        self.rho = rho_
        self.sigma_abs = sigma_
        self.sigma_rel = sigma_rel_

    def get_rho(self):
        return self.rho, self.sigma_abs, self.sigma_rel

    def set_sin2_12(self, val_):
        self.sin2_12 = val_

    def get_sin2_12(self):
        return self.sin2_12

    def set_deltam_21(self, val_):
        self.deltam_21 = val_

    def get_deltam_21(self):
        return self.deltam_21

    def set_sin2_13_N(self, val_):
        self.sin2_13_N = val_

    def get_sin2_13_N(self):
        return self.sin2_13_N

    def set_deltam_3l_N(self, val_):
        self.deltam_3l_N = val_

    def get_deltam_3l_N(self):
        return self.deltam_3l_N

    def set_sin2_13_I(self, val_):
        self.sin2_13_I = val_

    def get_sin2_13_I(self):
        return self.sin2_13_I

    def set_deltam_3l_I(self, val_):
        self.deltam_3l_I = val_

    def get_deltam_3l_I(self):
        return self.deltam_3l_I

    @staticmethod
    def sin2(x_, dm2_):
        """
        Compute the sine squared with given parameters.

        Compute the sin squared of 1.27 * `dm2` * `x`, where `x` is the independent variable. This method is used to
        evaluate the oscillatory term in the oscillation probability.

        Parameters
        ----------
        x_ : numpy array
            Input independent variable, corresponding to the ratio L/E [m/MeV].
        dm2_ : float
            Mass squared difference.

        Returns
        -------
        numpy array
            The sine squared of 1.27 * `dm2` * `x`.
        """
        appo = 1.27 * dm2_ * x_  # x in [m/MeV]
        return np.power(np.sin(appo), 2)

    @staticmethod
    def sin(x_, dm2_):
        appo = 1.27 * dm2_ * x_  # x in [m/MeV]
        return np.sin(appo)

    @staticmethod
    def cos(x_, dm2_):
        appo = 1.27 * dm2_ * x_  # x in [m/MeV]
        return np.cos(appo)

    # PRD 78, 2008, https: // arxiv.org / abs / 0807.3203, eq.(4)
    def eval_vacuum_prob_N(self, plot_this=False):

        x = np.arange(0.01, 500000, 1.)  # [m/MeV]

        A_N = math.pow(1 - self.sin2_13_N, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
        B_N = (1 - self.sin2_12) * 4 * self.sin2_13_N * (1 - self.sin2_13_N)
        C_N = self.sin2_12 * 4 * self.sin2_13_N * (1 - self.sin2_13_N)

        self.vacuum_prob_N = 1. - A_N * self.sin2(x, self.deltam_21) \
                             - B_N * self.sin2(x, self.deltam_3l_N) \
                             - C_N * self.sin2(x, self.deltam_3l_N - self.deltam_21)

        if plot_this:

            fig = plt.figure(figsize=[8, 5.5])
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.grid(alpha=0.45)
            ax.set_xlim(left=0.04, right=35)
            ax.set_ylim(0.08, 1.02)
            ax.set_xlabel(r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]')
            ax.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            # ax.set_title(r'Survival probability')
            # ax.plot(x / 1000, self.prob_N, 'b', linewidth=1, label=r'NO')
            ax.semilogx(x / 1000, self.vacuum_prob_N, 'b', linewidth=1, label=r'NO')
            ax.legend(loc='lower left')
            # fig.savefig('SpectrumPlots/prob_N_LE.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_N_LE.pdf')

        return self.vacuum_prob_N

    def eval_vacuum_prob_I(self, plot_this=False):

        x = np.arange(0.01, 500000, 1.)  # [m/MeV]

        A_I = math.pow(1 - self.sin2_13_I, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
        B_I = (1 - self.sin2_12) * 4 * self.sin2_13_I * (1 - self.sin2_13_I)
        C_I = self.sin2_12 * 4 * self.sin2_13_I * (1 - self.sin2_13_I)

        self.vacuum_prob_I = 1. - A_I * self.sin2(x, self.deltam_21) \
                             - B_I * self.sin2(x, self.deltam_3l_I + self.deltam_21) \
                             - C_I * self.sin2(x, self.deltam_3l_I)

        if plot_this:

            fig = plt.figure(figsize=[8, 5.5])
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.grid(alpha=0.45)
            ax.set_xlim(left=0.04, right=35)
            ax.set_ylim(0.08, 1.02)
            ax.set_xlabel(r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]')
            ax.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            # ax.set_title(r'Survival probability')
            # ax.plot(x / 1000, self.prob_I, 'r', linewidth=1, label=r'IO')
            ax.semilogx(x / 1000, self.vacuum_prob_I, 'r', linewidth=1, label=r'IO')
            ax.legend(loc='lower left')
            # fig.savefig('SpectrumPlots/prob_I_LE.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_I_LE.pdf')

        return self.vacuum_prob_I

    def eval_vacuum_prob(self, plot_this=False):

        self.eval_vacuum_prob_N()
        self.eval_vacuum_prob_I()

        if plot_this:

            fig = plt.figure(figsize=[8, 5.5])
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax.grid(alpha=0.45)
            ax.set_xlim(left=0.04, right=35)
            ax.set_ylim(0.08, 1.02)
            ax.set_xlabel(r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]')
            ax.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            # ax.set_title(r'Survival probability')
            x = np.arange(0.01, 500000, 1.)  # [m/MeV]
            # ax.plot(x / 1000, self.prob_N, 'b', linewidth=1, label=r'NO')
            # ax.plot(x / 1000, self.prob_I, 'r--', linewidth=1, label=r'IO')
            ax.semilogx(x / 1000, self.vacuum_prob_N, 'b', linewidth=1, label=r'NO')
            ax.semilogx(x / 1000, self.vacuum_prob_I, 'r--', linewidth=1, label=r'IO')
            ax.legend(loc='lower left')
            # fig.savefig('SpectrumPlots/prob_LE.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_LE.pdf')

            return self.vacuum_prob_N, self.vacuum_prob_I

    def eval_vacuum_prob_N_energy(self, nu_energy, plot_this=False):

        x_energy = self.baseline * 1000 / nu_energy  # [m/MeV]

        aa = np.power(1. - self.sin2_13_N, 2) * 4. * self.sin2_12 * (1. - self.sin2_12)
        bb = (1. - self.sin2_12) * 4. * self.sin2_13_N * (1. - self.sin2_13_N)
        cc = self.sin2_12 * 4. * self.sin2_13_N * (1. - self.sin2_13_N)

        self.vacuum_prob_N_E = 1. - aa * self.sin2(x_energy, self.deltam_21) \
                               - bb * self.sin2(x_energy, self.deltam_3l_N) \
                               - cc * self.sin2(x_energy, self.deltam_3l_N - self.deltam_21)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig1 = plt.figure(figsize=[8, 5.5])
            ax1 = fig1.add_subplot(111)
            fig1.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_ylim(0.08, 1.02)
            # ax1.set_title(r'Survival probability')
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')
            ax1.plot(nu_energy, self.vacuum_prob_N_E, 'b', linewidth=1, label=r'NO')
            ax1.legend(loc='lower right')
            # fig1.savefig('SpectrumPlots/prob_N_E.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_N_E.pdf')

        return self.vacuum_prob_N_E

    def eval_vacuum_prob_I_energy(self, nu_energy, plot_this=False):

        x_energy = self.baseline * 1000 / nu_energy  # [m/MeV]

        aa = np.power(1. - self.sin2_13_I, 2) * 4. * self.sin2_12 * (1. - self.sin2_12)
        bb = (1. - self.sin2_12) * 4. * self.sin2_13_I * (1. - self.sin2_13_I)
        cc = self.sin2_12 * 4. * self.sin2_13_I * (1. - self.sin2_13_I)

        self.vacuum_prob_I_E = 1. - aa * self.sin2(x_energy, self.deltam_21) \
                               - bb * self.sin2(x_energy, self.deltam_3l_I + self.deltam_21) \
                               - cc * self.sin2(x_energy, self.deltam_3l_I)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig1 = plt.figure(figsize=[8, 5.5])
            ax1 = fig1.add_subplot(111)
            fig1.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_ylim(0.08, 1.02)
            # ax1.set_title(r'Survival probability')
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')
            ax1.plot(nu_energy, self.vacuum_prob_I_E, 'r', linewidth=1, label=r'IO')
            ax1.legend(loc='lower right')
            # fig1.savefig('SpectrumPlots/prob_I_E.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_I_E.pdf')

        return self.vacuum_prob_I_E

    def eval_vacuum_prob_energy(self, nu_energy, plot_this=False):

        self.eval_vacuum_prob_N_energy(nu_energy)
        self.eval_vacuum_prob_I_energy(nu_energy)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig1 = plt.figure(figsize=[8, 5.5])
            ax1 = fig1.add_subplot(111)
            fig1.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_ylim(0.08, 1.02)
            # ax1.set_title(r'Survival probability')
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')
            ax1.plot(nu_energy, self.vacuum_prob_N_E, 'b', linewidth=1, label=r'NO')
            ax1.plot(nu_energy, self.vacuum_prob_I_E, 'r--', linewidth=1, label=r'IO')
            ax1.legend(loc='lower right')
            # fig1.savefig('SpectrumPlots/prob_E.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_E.pdf')

        return self.vacuum_prob_N_E, self.vacuum_prob_I_E

    def potential(self, nu_energy):
        y_e = 0.5
        return -1.52e-4 * y_e * self.rho * nu_energy * 1.e-3

    def eval_matter_prob_N_energy(self, nu_energy, plot_this=False):

        x_energy = self.baseline * 1000 / nu_energy  # [m/MeV]

        deltam_ee = self.deltam_3l_N - self.sin2_12 * self.deltam_21
        deltam_32 = self.deltam_3l_N - self.deltam_21

        c2_12 = 1. - self.sin2_12
        c2_13 = 1. - self.sin2_13_N
        c_2_12 = 1. - 2. * self.sin2_12
        appo_12 = c2_13 * self.potential(nu_energy) / self.deltam_21

        sin2_12_m = self.sin2_12 * (1. + 2. * c2_12 * appo_12 + 3. * c2_12 * c_2_12 * appo_12 * appo_12)
        deltam_21_m = self.deltam_21 * (1. - c_2_12 * appo_12 + 2. * self.sin2_12 * c2_12 * appo_12 * appo_12)
        sin2_13_m = self.sin2_13_N * (1. + 2. * c2_13 * self.potential(nu_energy) / deltam_ee)
        deltam_31_m = self.deltam_3l_N \
                      * (1. - self.potential(nu_energy) / self.deltam_3l_N * (c2_12 * c2_13 - self.sin2_13_N
                                                                              - self.sin2_12 * c2_12 * c2_13 * appo_12))
        deltam_32_m = deltam_32 \
                      * (1. - self.potential(nu_energy) / deltam_32 * (self.sin2_12 * c2_13 - self.sin2_13_N
                                                                       + self.sin2_12 * c2_12 * c2_13 * appo_12))

        aa = np.power(1. - sin2_13_m, 2) * 4. * sin2_12_m * (1. - sin2_12_m)
        bb = (1. - sin2_12_m) * 4 * sin2_13_m * (1. - sin2_13_m)
        cc = sin2_12_m * 4 * sin2_13_m * (1. - sin2_13_m)

        self.matter_prob_N_E = 1. - aa * self.sin2(x_energy, deltam_21_m) \
                               - bb * self.sin2(x_energy, deltam_31_m) \
                               - cc * self.sin2(x_energy, deltam_32_m)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig1 = plt.figure(figsize=[8, 5.5])
            ax1 = fig1.add_subplot(111)
            fig1.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylabel(r'$P_{\text{mat}} (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_ylim(0.08, 1.02)
            # ax1.set_title(r'Survival probability')
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')
            ax1.plot(nu_energy, self.matter_prob_N_E, 'b', linewidth=1, label=r'NO')
            ax1.legend(loc='lower right')
            # fig1.savefig('SpectrumPlots/prob_N_E.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_N_E.pdf')

        return self.matter_prob_N_E

    def eval_matter_prob_I_energy(self, nu_energy, plot_this=False):

        x_energy = self.baseline * 1000 / nu_energy  # [m/MeV]

        deltam_ee = self.deltam_3l_I + self.deltam_21 * (1. - self.sin2_12)
        deltam_31 = self.deltam_3l_I + self.deltam_21

        c2_12 = 1. - self.sin2_12
        c2_13 = 1. - self.sin2_13_I
        c_2_12 = 1. - 2. * self.sin2_12
        appo_12 = c2_13 * self.potential(nu_energy) / self.deltam_21

        sin2_12_m = self.sin2_12 * (1. + 2. * c2_12 * appo_12 + 3. * c2_12 * c_2_12 * appo_12 * appo_12)
        deltam_21_m = self.deltam_21 * (1. - c_2_12 * appo_12 + 2. * self.sin2_12 * c2_12 * appo_12 * appo_12)
        sin2_13_m = self.sin2_13_I * (1. + 2. * c2_13 * self.potential(nu_energy) / deltam_ee)
        deltam_31_m = deltam_31 \
                      * (1. - self.potential(nu_energy) / deltam_31 * (c2_12 * c2_13 - self.sin2_13_I
                                                                       - self.sin2_12 * c2_12 * c2_13 * appo_12))
        deltam_32_m = self.deltam_3l_I \
                      * (1. - self.potential(nu_energy) / self.deltam_3l_I * (self.sin2_12 * c2_13 - self.sin2_13_I
                                                                              + self.sin2_12 * c2_12 * c2_13 * appo_12))

        aa = np.power(1. - sin2_13_m, 2) * 4. * sin2_12_m * (1. - sin2_12_m)
        bb = (1. - sin2_12_m) * 4 * sin2_13_m * (1. - sin2_13_m)
        cc = sin2_12_m * 4 * sin2_13_m * (1. - sin2_13_m)

        self.matter_prob_I_E = 1. - aa * self.sin2(x_energy, deltam_21_m) \
                               - bb * self.sin2(x_energy, deltam_31_m) \
                               - cc * self.sin2(x_energy, deltam_32_m)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig1 = plt.figure(figsize=[8, 5.5])
            ax1 = fig1.add_subplot(111)
            fig1.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylabel(r'$P_{\text{mat}} (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_ylim(0.08, 1.02)
            # ax1.set_title(r'Survival probability')
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')
            ax1.plot(nu_energy, self.matter_prob_I_E, 'b', linewidth=1, label=r'NO')
            ax1.legend(loc='lower right')
            # fig1.savefig('SpectrumPlots/prob_N_E.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_N_E.pdf')

        return self.matter_prob_I_E

    def eval_matter_prob_energy(self, nu_energy, plot_this=False):

        self.eval_matter_prob_N_energy(nu_energy)
        self.eval_matter_prob_I_energy(nu_energy)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)
            fig1 = plt.figure(figsize=[8, 5.5])
            ax1 = fig1.add_subplot(111)
            fig1.subplots_adjust(left=0.11, right=0.96, top=0.95)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylabel(r'$P_{\text{mat}} (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_ylim(0.08, 1.02)
            # ax1.set_title(r'Survival probability')
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')
            ax1.plot(nu_energy, self.matter_prob_N_E, 'b', linewidth=1, label=r'NO')
            ax1.plot(nu_energy, self.matter_prob_I_E, 'r--', linewidth=1, label=r'IO')
            ax1.legend(loc='lower right')
            # fig1.savefig('SpectrumPlots/prob_N_E.pdf', format='pdf', transparent=True)
            # print('\nThe plot has been saved in SpectrumPlots/prob_N_E.pdf')

        return self.matter_prob_N_E, self.matter_prob_I_E


    ### evaluation of the survival probability with the same formula as above written in an other way
    ### see JHEP05(2013)131, eq. (2.6), Japanese (https://arxiv.org/abs/1210.8141)
    def eval_prob_jhep(self, E, ordering, plot_this=False):

        if (ordering < -1) or (ordering > 1):
            print('Error: use 1 for NO, -1 for IO, 0 for both')
            return -1

        x = np.arange(0.01, 500000, 1.)  # [m/MeV]
        x_E = self.baseline * 1000 / E  # [m/MeV]

        # Normal Ordering
        if (ordering == 1) or (ordering == 0):
            A_N = math.pow(1 - self.sin2_13_N, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
            B_N = 4 * self.sin2_13_N * (1 - self.sin2_13_N)
            C_N = self.sin2_12 * B_N
            D_N = self.sin2_12 * B_N / 2.

            self.jhep_prob_N = 1. - A_N * self.sin2(x, self.deltam_21) - B_N * self.sin2(x, abs(self.deltam_3l_N)) \
                                  - C_N * self.sin2(x, self.deltam_21) * self.cos(x, 2 * abs(self.deltam_3l_N)) \
                                  + D_N * self.sin(x, 2 * abs(self.deltam_3l_N)) * self.sin(x, 2 * self.deltam_21)
            self.jhep_prob_E_N = 1. - A_N * self.sin2(x_E, self.deltam_21) - B_N * self.sin2(x_E, abs(self.deltam_3l_N)) \
                                    - C_N * self.sin2(x_E, self.deltam_21) * self.cos(x_E, 2 * abs(self.deltam_3l_N)) \
                                    + D_N * self.sin(x_E, 2 * abs(self.deltam_3l_N)) * self.sin(x_E, 2 * self.deltam_21)

        # Inverted Ordering
        if (ordering == -1) or (ordering == 0):
            A_I = math.pow(1 - self.sin2_13_I, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
            B_I = 4 * self.sin2_13_I * (1 - self.sin2_13_I)
            C_I = self.sin2_12 * B_I
            D_I = self.sin2_12 * B_I / 2.

            delta_appo = self.deltam_21 + self.deltam_3l_I
            print(delta_appo)

            self.jhep_prob_I = 1. - A_I * self.sin2(x, self.deltam_21) - B_I * self.sin2(x, abs(delta_appo)) \
                                  - C_I * self.sin2(x, self.deltam_21) * self.cos(x, 2 * abs(delta_appo)) \
                                  - D_I * self.sin(x, 2 * abs(delta_appo)) * self.sin(x, 2 * self.deltam_21)
            self.jhep_prob_E_I = 1. - A_I * self.sin2(x_E, self.deltam_21) - B_I * self.sin2(x_E, abs(delta_appo)) \
                                    - C_I * self.sin2(x_E, self.deltam_21) * self.cos(x_E, 2 * abs(delta_appo)) \
                                    - D_I * self.sin(x_E, 2 * abs(delta_appo)) * self.sin(x_E, 2 * self.deltam_21)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.11,right=0.96,top=0.9)
            ax.grid()
            ax.set_xlim(left=0.02, right=80)
            ax.set_xlabel(r'$L / E_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]')
            ax.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax.set_title(r'Survival probability' + '\n(jhep05(2013)131)')
            ax.set_xlim(left=0.04, right=60)
            ax.set_ylim(0.08, 1.02)

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            # fig1.subplots_adjust(left=0.11,right=0.96,top=0.9)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_title(r'Survival probability' + '\n(jhep05(2013)131)')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylim(0.08, 1.02)
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')

            if ordering == 1:  # NO

                ax.semilogx(x / 1000, self.jhep_prob_N, 'b', linewidth=1, label='NO')
                ax.legend(loc='lower left')
                # fig.savefig('SpectrumPlots/jhep_2013_prob_N_LE.pdf', format='pdf', transparent=True)
                # print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_N_LE.pdf')
                ax1.plot(E, self.jhep_prob_E_N, 'b', linewidth=1, label='NO')
                ax1.legend(loc='lower right')
                # fig1.savefig('SpectrumPlots/jhep_2013_prob_N_E.pdf', format='pdf', transparent=True)
                # print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_N_E.pdf')

            if ordering == -1:  # IO
                ax.semilogx(x / 1000, self.jhep_prob_I, 'r', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                # fig.savefig('SpectrumPlots/jhep_2013_prob_I_LE.pdf', format='pdf', transparent=True)
                # print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_I_LE.pdf')
                ax1.plot(E, self.jhep_prob_E_I, 'r', linewidth=1, label='IO')
                ax1.legend(loc='lower right')
                # fig1.savefig('SpectrumPlots/jhep_2013_prob_I_E.pdf', format='pdf', transparent=True)
                # print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_I_E.pdf')

            if ordering == 0:  # both NO and IO
                ax.semilogx(x / 1000, self.jhep_prob_N, 'b', linewidth=1, label='NO')
                ax.semilogx(x / 1000, self.jhep_prob_I, 'r--', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                # fig.savefig('SpectrumPlots/jhep_2013_prob_LE.pdf', format='pdf', transparent=True)
                # print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_LE.pdf')
                ax1.plot(E, self.jhep_prob_E_N, 'b', linewidth=1, label='NO')
                ax1.plot(E, self.jhep_prob_E_I, 'r--', linewidth=1, label='IO')
                ax1.legend(loc='lower right')
                # fig1.savefig('SpectrumPlots/jhep_2013_prob_E.pdf', format='pdf', transparent=True)
                # print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_E.pdf')

        if ordering == 1:
            return self.jhep_prob_E_N
        if ordering == -1:
            return self.jhep_prob_E_I
        if ordering == 0:
            return self.jhep_prob_E_N, self.jhep_prob_E_I

