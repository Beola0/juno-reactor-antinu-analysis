import matplotlib.pyplot as plt
import numpy as np
import math


### survival probability
class OscillationProbability:

    def __init__(self, t12, m21, t13_N, m3l_N, t13_I, m3l_I):
        self.baseline = 52.5  # [km], fixed for now -> can be modified in class Spectrum

        self.sin2_12 = t12
        self.deltam_21 = m21  # [eV^2]

        # Normal Ordering: 3l = 31
        self.sin2_13_N = t13_N
        self.deltam_3l_N = m3l_N  # [eV^2]

        # Inverted Ordering: 3l = 32
        self.sin2_13_I = t13_I
        self.deltam_3l_I = m3l_I  # [eV^2]

        self.prob_N = 0
        self.prob_E_N = 0
        self.prob_I = 0
        self.prob_E_I = 0
        self.jhep_prob_N = 0
        self.jhep_prob_E_N = 0
        self.jhep_prob_I = 0
        self.jhep_prob_E_I = 0

    ### this function evaluates the oscillatory term
    @staticmethod
    def sin2(x, dm2):
        appo = 1.27 * dm2 * x  # x in [m/MeV]
        return np.power(np.sin(appo), 2)

    ### this function evaluates the survival probability as function of L/E and E
    ### formula from PRD 78, 2008, Chinese, https://arxiv.org/abs/0807.3203, eq. (4)
    def eval_prob(self, E, ordering, plot_this_LE=False, plot_this_E=False):

        if (ordering < -1) or (ordering > 1):
            print('Error: use 1 for NO, -1 for IO, 0 for both')
            return -1

        x = np.arange(0.01, 500000, 1.)  # [m/MeV]
        x_E = self.baseline * 1000 / E  # [m/MeV]

        # Normal Ordering
        A_N = math.pow(1 - self.sin2_13_N, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
        B_N = (1 - self.sin2_12) * 4 * self.sin2_13_N * (1 - self.sin2_13_N)
        C_N = self.sin2_12 * 4 * self.sin2_13_N * (1 - self.sin2_13_N)

        self.prob_N = 1. - A_N * OscillationProbability.sin2(x, self.deltam_21) \
                      - B_N * OscillationProbability.sin2(x, self.deltam_3l_N) \
                      - C_N * OscillationProbability.sin2(x, self.deltam_3l_N - self.deltam_21)
        self.prob_E_N = 1. - A_N * OscillationProbability.sin2(x_E, self.deltam_21) \
                        - B_N * OscillationProbability.sin2(x_E, self.deltam_3l_N) \
                        - C_N * OscillationProbability.sin2(x_E, self.deltam_3l_N - self.deltam_21)

        # Inverted Ordering
        A_I = math.pow(1 - self.sin2_13_I, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
        B_I = (1 - self.sin2_12) * 4 * self.sin2_13_I * (1 - self.sin2_13_I)
        C_I = self.sin2_12 * 4 * self.sin2_13_I * (1 - self.sin2_13_I)

        self.prob_I = 1. - A_I * OscillationProbability.sin2(x, self.deltam_21) \
                      - B_I * OscillationProbability.sin2(x, self.deltam_3l_I + self.deltam_21) \
                      - C_I * OscillationProbability.sin2(x, self.deltam_3l_I)
        self.prob_E_I = 1. - A_I * OscillationProbability.sin2(x_E, self.deltam_21) \
                        - B_I * OscillationProbability.sin2(x_E, self.deltam_3l_I + self.deltam_21) \
                        - C_I * OscillationProbability.sin2(x_E, self.deltam_3l_I)

        if plot_this_LE:

            fig = plt.figure()
            # fig.suptitle(r'Survival probability') # as function of L/E
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.11,right=0.96,top=0.9)
            ax.grid()
            ax.set_xlim(left=0.02, right=80)
            ax.set_xlabel(r'L/$\text{E}_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]')
            ax.set_ylabel(r'P ($\bar{\nu}_{e} \rightarrow \bar{\nu}_{e}$)')
            ax.set_title(r'Survival probability')

            if ordering == 1:  # NO
                ax.semilogx(x / 1000, self.prob_N, 'b', linewidth=1, label='NO')
                ax.legend(loc='lower left')
                fig.savefig('SpectrumPlots/prob_N_LE.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/prob_N_LE.pdf')

            if ordering == -1:  # IO
                ax.semilogx(x / 1000, self.prob_I, 'r', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                fig.savefig('SpectrumPlots/prob_I_LE.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/prob_I_LE.pdf')

            if ordering == 0:  # both NO and IO
                ax.semilogx(x / 1000, self.prob_N, 'b', linewidth=1, label='NO')
                ax.semilogx(x / 1000, self.prob_I, 'r--', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                fig.savefig('SpectrumPlots/prob_LE.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/prob_LE.pdf')

        if plot_this_E:

            fig1 = plt.figure()
            # fig1.suptitle(r'Survival probability') # as function of E
            ax1 = fig1.add_subplot(111)
            # fig1.subplots_adjust(left=0.11,right=0.96,top=0.9)
            ax1.grid()
            ax1.set_xlabel(r'$\text{E}_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_ylabel(r'P ($\bar{\nu}_{e} \rightarrow \bar{\nu}_{e}$)')
            ax1.set_title(r'Survival probability')

            if ordering == 1:  # NO
                ax1.plot(E, self.prob_E_N, 'b', linewidth=1, label='NO')
                ax1.legend(loc='lower right')
                fig1.savefig('SpectrumPlots/prob_N_E.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/prob_N_E.pdf')

            if ordering == -1:  # IO
                ax1.plot(E, self.prob_E_I, 'r', linewidth=1, label='IO')
                ax1.legend(loc='lower right')
                fig1.savefig('SpectrumPlots/prob_I_E.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/prob_I_E.pdf')

            if ordering == 0:  # both NO and IO
                ax1.plot(E, self.prob_E_N, 'b', linewidth=1, label='NO')
                ax1.plot(E, self.prob_E_I, 'r--', linewidth=1, label='IO')
                ax1.legend(loc='lower right')
                fig1.savefig('SpectrumPlots/prob_E.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/prob_E.pdf')

        return self.prob_E_N, self.prob_E_I

    ### evaluation of the survival probability with the same formula as above written in an other way
    ### see JHEP05(2013)131, eq. (2.6), Japanese (https://arxiv.org/abs/1210.8141)

    @staticmethod
    def sin(x, dm2):
        appo = 1.27 * dm2 * x  # x in [m/MeV]
        return np.sin(appo)

    @staticmethod
    def cos(x, dm2):
        appo = 1.27 * dm2 * x  # x in [m/MeV]
        return np.cos(appo)

    def eval_prob_jhep(self, E, ordering, plot_this=False):

        if (ordering < -1) or (ordering > 1):
            print('Error: use 1 for NO, -1 for IO, 0 for both')
            return -1

        x = np.arange(0.01, 500000, 1.)  # [m/MeV]
        x_E = self.baseline * 1000 / E  # [m/MeV]

        # Normal Ordering
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

            fig = plt.figure()
            # fig.suptitle(r'Survival probability') # as function of L/E
            ax = fig.add_subplot(111)
            # fig.subplots_adjust(left=0.11,right=0.96,top=0.9)
            ax.grid()
            ax.set_xlim(left=0.02, right=80)
            ax.set_xlabel(r'L/$\text{E}_{\nu}$ [\si[per-mode=symbol]{\kilo\meter\per\MeV}]')
            ax.set_ylabel(r'P ($\bar{\nu}_{e} \rightarrow \bar{\nu}_{e}$)')
            ax.set_title(r'Survival probability' + '\n(jhep05(2013)131)')

            fig1 = plt.figure()
            # fig1.suptitle(r'Survival probability') # as function of E
            ax1 = fig1.add_subplot(111)
            # fig1.subplots_adjust(left=0.11,right=0.96,top=0.9)
            ax1.grid()
            ax1.set_xlabel(r'$\text{E}_{\nu}$ [\si[per-mode=symbol]{\MeV}]')
            ax1.set_ylabel(r'P ($\bar{\nu}_{e} \rightarrow \bar{\nu}_{e}$)')
            ax1.set_title(r'Survival probability' + '\n(jhep05(2013)131)')

            if ordering == 1:  # NO

                ax.semilogx(x / 1000, self.jhep_prob_N, 'b', linewidth=1, label='NO')
                ax.legend(loc='lower left')
                fig.savefig('SpectrumPlots/jhep_2013_prob_N_LE.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_N_LE.pdf')
                ax1.plot(E, self.jhep_prob_E_N, 'b', linewidth=1, label='NO')
                ax1.legend(loc='lower right')
                fig1.savefig('SpectrumPlots/jhep_2013_prob_N_E.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_N_E.pdf')

            if ordering == -1:  # IO
                ax.semilogx(x / 1000, self.jhep_prob_I, 'r', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                fig.savefig('SpectrumPlots/jhep_2013_prob_I_LE.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_I_LE.pdf')
                ax1.plot(E, self.jhep_prob_E_I, 'r', linewidth=1, label='IO')
                ax1.legend(loc='lower right')
                fig1.savefig('SpectrumPlots/jhep_2013_prob_I_E.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_I_E.pdf')

            if ordering == 0:  # both NO and IO
                ax.semilogx(x / 1000, self.jhep_prob_N, 'b', linewidth=1, label='NO')
                ax.semilogx(x / 1000, self.jhep_prob_I, 'r--', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                fig.savefig('SpectrumPlots/jhep_2013_prob_LE.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_LE.pdf')
                ax1.plot(E, self.jhep_prob_E_N, 'b', linewidth=1, label='NO')
                ax1.plot(E, self.jhep_prob_E_I, 'r--', linewidth=1, label='IO')
                ax1.legend(loc='lower right')
                fig1.savefig('SpectrumPlots/jhep_2013_prob_E.pdf', format='pdf', transparent=True)
                print('\nThe plot has been saved in SpectrumPlots/jhep_2013_prob_E.pdf')

        return self.jhep_prob_E_N, self.jhep_prob_E_I
