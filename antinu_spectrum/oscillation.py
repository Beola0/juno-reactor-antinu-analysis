import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import math
from .plot import plot_function

# TODO:
# - divide NO and IO into separate methods + third method for both --> DONE
# - new method: change baseline --> from json file --> DONE
# - remove part for plotting --> improved, DONE
# - put together E and L/E and use a flag (or energy='', if not do E) --> DONE
# - add nuisances: for matter density

HEADER = '\033[95m'
BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
NC = '\033[0m'

label_LE = r'$L / E_{\nu}$ [km/MeV]'


style = {
    "NO": 'b',
    "IO1": 'r',
    "IO2": 'r--'
}


### survival probability
class OscillationProbability:

    def __init__(self, inputs_json_):
        self.baseline = inputs_json_["baseline"]  # [km]

        self.sin2_12 = inputs_json_["oscillationParams"]["sin2_12"]
        self.deltam_21 = inputs_json_["oscillationParams"]["deltam_21"]  # [eV^2]

        # Normal Ordering: 3l = 31
        self.sin2_13_no = inputs_json_["oscillationParams"]["sin2_13_N"]
        self.deltam_3l_no = inputs_json_["oscillationParams"]["deltam_31_N"]  # [eV^2]

        # Inverted Ordering: 3l = 32
        self.sin2_13_io = inputs_json_["oscillationParams"]["sin2_13_I"]
        self.deltam_3l_io = inputs_json_["oscillationParams"]["deltam_32_I"]  # [eV^2]

        # Matter density for matter effect
        self.rho = inputs_json_["matter_effect"]["matter_density"]
        self.sigma_abs = inputs_json_["matter_effect"]["abs_sigma"]
        self.sigma_rel = inputs_json_["matter_effect"]["rel_sigma"]

        self.verbose = inputs_json_["verbose"]

        self.vacuum_prob_no = 0.
        self.vacuum_prob_no_E = 0.
        self.vacuum_prob_io = 0.
        self.vacuum_prob_io_E = 0.
        self.matter_prob_no = 0.
        self.matter_prob_no_E = 0.
        self.matter_prob_io = 0.
        self.matter_prob_io_E = 0.
        self.jhep_prob_no = 0.
        self.jhep_prob_E_no = 0.
        self.jhep_prob_io = 0.
        self.jhep_prob_E_io = 0.

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

    def set_sin2_13_no(self, val_):
        self.sin2_13_no = val_

    def get_sin2_13_no(self):
        return self.sin2_13_no

    def set_deltam_3l_no(self, val_):
        self.deltam_3l_no = val_

    def get_deltam_3l_no(self):
        return self.deltam_3l_no

    def set_sin2_13_io(self, val_):
        self.sin2_13_io = val_

    def get_sin2_13_io(self):
        return self.sin2_13_io

    def set_deltam_3l_io(self, val_):
        self.deltam_3l_io = val_

    def get_deltam_3l_io(self):
        return self.deltam_3l_io

    def set_baseline(self, val_):
        self.baseline = val_

    def get_baseline(self):
        return self.baseline

    @staticmethod
    def sin2(x_, dm2_):
        appo = 1.267 * dm2_ * x_  # x in [m/MeV]
        return np.power(np.sin(appo), 2)

    @staticmethod
    def sin(x_, dm2_):
        appo = 1.267 * dm2_ * x_  # x in [m/MeV]
        return np.sin(appo)

    @staticmethod
    def cos(x_, dm2_):
        appo = 1.267 * dm2_ * x_  # x in [m/MeV]
        return np.cos(appo)

    # PRD 78, 2008, https: // arxiv.org / abs / 0807.3203, eq.(4)
    # variable x is L/E
    def eval_vacuum_prob_no(self, nu_energy=None, plot_this=False):

        if self.verbose:
            print(f"\n{CYAN}OscProb: evaluating probability in vacuum for NO.{NC}")

        if nu_energy is not None:
            x = self.baseline * 1000 / nu_energy  # [m/MeV]
        else:
            x = np.arange(0.01, 500000, 1.)  # [m/MeV]

        aa = math.pow(1 - self.sin2_13_no, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
        bb = (1 - self.sin2_12) * 4 * self.sin2_13_no * (1 - self.sin2_13_no)
        cc = self.sin2_12 * 4 * self.sin2_13_no * (1 - self.sin2_13_no)

        self.vacuum_prob_no = 1. - aa * self.sin2(x, self.deltam_21) \
                             - bb * self.sin2(x, self.deltam_3l_no) \
                             - cc * self.sin2(x, self.deltam_3l_no - self.deltam_21)

        if plot_this:
            ylabel = r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$'
            if nu_energy is not None:
                plot_function(x_=[nu_energy], y_=[self.vacuum_prob_no], label_=[r'NO'], styles=[style["NO"]],
                              ylabel_=ylabel, xlim=[1.5, 10.5], ylim=[0.08, 1.02])
            else:
                xlabel = label_LE
                plot_function(x_=[x/1000.], y_=[self.vacuum_prob_no], label_=[r'NO'], styles=[style["NO"]],
                              ylabel_=ylabel, xlabel_=xlabel, xlim=[0.04, 35], ylim=[0.08, 1.02], logx=True)

        return self.vacuum_prob_no

    def eval_vacuum_prob_io(self, nu_energy=None, plot_this=False):

        if self.verbose:
            print(f"\n{CYAN}OscProb: evaluating probability in vacuum for IO.{NC}")

        if nu_energy is not None:
            x = self.baseline * 1000 / nu_energy  # [m/MeV]
        else:
            x = np.arange(0.01, 500000, 1.)  # [m/MeV]

        aa = math.pow(1 - self.sin2_13_io, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
        bb = (1 - self.sin2_12) * 4 * self.sin2_13_io * (1 - self.sin2_13_io)
        cc = self.sin2_12 * 4 * self.sin2_13_io * (1 - self.sin2_13_io)

        self.vacuum_prob_io = 1. - aa * self.sin2(x, self.deltam_21) \
                             - bb * self.sin2(x, self.deltam_3l_io + self.deltam_21) \
                             - cc * self.sin2(x, self.deltam_3l_io)

        if plot_this:
            ylabel = r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$'
            if nu_energy is not None:
                plot_function(x_=[nu_energy], y_=[self.vacuum_prob_io], label_=[r'IO'], styles=[style["IO1"]],
                              ylabel_=ylabel, xlim=[1.5, 10.5], ylim=[0.08, 1.02])
            else:
                xlabel = label_LE
                plot_function(x_=[x / 1000.], y_=[self.vacuum_prob_io], label_=[r'IO'], styles=[style["IO1"]],
                              ylabel_=ylabel, xlabel_=xlabel, xlim=[0.04, 35], ylim=[0.08, 1.02], logx=True)

        return self.vacuum_prob_io

    def eval_vacuum_prob(self, nu_energy=None, plot_this=False):

        self.eval_vacuum_prob_no(nu_energy=nu_energy)
        self.eval_vacuum_prob_io(nu_energy=nu_energy)

        if plot_this:
            ylabel = r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$'
            if nu_energy is not None:
                plot_function(x_=[nu_energy, nu_energy],
                              y_=[self.vacuum_prob_no, self.vacuum_prob_io],
                              label_=[r'NO', r'IO'],
                              styles=['b', 'r--'],
                              ylabel_=ylabel, xlim=[1.5, 10.5], ylim=[0.08, 1.02])
            else:
                x = np.arange(0.01, 500000, 1.)  # [m/MeV]
                xlabel = label_LE
                plot_function(x_=[x / 1000., x / 1000.],
                              y_=[self.vacuum_prob_no, self.vacuum_prob_io],
                              label_=[r'NO', r'IO'],
                              styles=[style["NO"], style["IO2"]],
                              ylabel_=ylabel, xlabel_=xlabel, xlim=[0.04, 35], ylim=[0.08, 1.02], logx=True,
                              fig_length=9, fig_height=4)

        return self.vacuum_prob_no, self.vacuum_prob_io

    def potential(self, nu_energy):
        y_e = 0.5
        return -1.52e-4 * y_e * self.rho * nu_energy * 1.e-3

    def eval_matter_prob_no(self, nu_energy=None, plot_this=False):

        if self.verbose:
            print(f"\n{CYAN}OscProb: evaluating probability in matter for NO.{NC}")

        if nu_energy is not None:
            x = self.baseline * 1000 / nu_energy  # [m/MeV]
            nu_en = nu_energy
        else:
            x = np.arange(0.01, 500000, 1.)  # [m/MeV]
            nu_en = self.baseline * 1000 / x

        deltam_ee = self.deltam_3l_no - self.sin2_12 * self.deltam_21
        deltam_32 = self.deltam_3l_no - self.deltam_21

        c2_12 = 1. - self.sin2_12
        c2_13 = 1. - self.sin2_13_no
        c_2_12 = 1. - 2. * self.sin2_12
        pot = self.potential(nu_en)
        appo_12 = c2_13 * pot / self.deltam_21

        sin2_12_m = self.sin2_12 * (1. + 2. * c2_12 * appo_12 + 3. * c2_12 * c_2_12 * appo_12 * appo_12)
        deltam_21_m = self.deltam_21 * (1. - c_2_12 * appo_12 + 2. * self.sin2_12 * c2_12 * appo_12 * appo_12)
        sin2_13_m = self.sin2_13_no * (1. + 2. * c2_13 * pot / deltam_ee)
        deltam_31_m = self.deltam_3l_no \
                      * (1. - pot / self.deltam_3l_no * (c2_12 * c2_13 - self.sin2_13_no
                                                         - self.sin2_12 * c2_12 * c2_13 * appo_12))
        deltam_32_m = deltam_32 \
                      * (1. - pot / deltam_32 * (self.sin2_12 * c2_13 - self.sin2_13_no
                                                 + self.sin2_12 * c2_12 * c2_13 * appo_12))

        aa = np.power(1. - sin2_13_m, 2) * 4. * sin2_12_m * (1. - sin2_12_m)
        bb = (1. - sin2_12_m) * 4 * sin2_13_m * (1. - sin2_13_m)
        cc = sin2_12_m * 4 * sin2_13_m * (1. - sin2_13_m)

        self.matter_prob_no = 1. - aa * self.sin2(x, deltam_21_m) \
                               - bb * self.sin2(x, deltam_31_m) \
                               - cc * self.sin2(x, deltam_32_m)

        if plot_this:
            ylabel = r'$P_{\textrm{mat}} (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$'
            if nu_energy is not None:
                plot_function(x_=[nu_energy], y_=[self.matter_prob_no], label_=[r'NO'], styles=[style["NO"]],
                              ylabel_=ylabel, xlim=[1.5, 10.5], ylim=[0.08, 1.02])
            else:
                xlabel = label_LE
                plot_function(x_=[x / 1000.], y_=[self.matter_prob_no], label_=[r'NO'], styles=[style["NO"]],
                              ylabel_=ylabel, xlabel_=xlabel, xlim=[0.04, 35], ylim=[0.08, 1.02], logx=True)

        return self.matter_prob_no

    def eval_matter_prob_io(self, nu_energy=None, plot_this=False):

        if self.verbose:
            print(f"\n{CYAN}OscProb: evaluating probability in matter for IO.{NC}")

        if nu_energy is not None:
            x = self.baseline * 1000 / nu_energy  # [m/MeV]
            nu_en = nu_energy
        else:
            x = np.arange(0.01, 500000, 1.)  # [m/MeV]
            nu_en = self.baseline * 1000 / x

        deltam_ee = self.deltam_3l_io + self.deltam_21 * (1. - self.sin2_12)
        deltam_31 = self.deltam_3l_io + self.deltam_21

        c2_12 = 1. - self.sin2_12
        c2_13 = 1. - self.sin2_13_io
        c_2_12 = 1. - 2. * self.sin2_12
        pot = self.potential(nu_en)
        appo_12 = c2_13 * pot / self.deltam_21

        sin2_12_m = self.sin2_12 * (1. + 2. * c2_12 * appo_12 + 3. * c2_12 * c_2_12 * appo_12 * appo_12)
        deltam_21_m = self.deltam_21 * (1. - c_2_12 * appo_12 + 2. * self.sin2_12 * c2_12 * appo_12 * appo_12)
        sin2_13_m = self.sin2_13_io * (1. + 2. * c2_13 * pot / deltam_ee)
        deltam_31_m = deltam_31 \
                      * (1. - pot / deltam_31 * (c2_12 * c2_13 - self.sin2_13_io
                                                 - self.sin2_12 * c2_12 * c2_13 * appo_12))
        deltam_32_m = self.deltam_3l_io \
                      * (1. - pot / self.deltam_3l_io * (self.sin2_12 * c2_13 - self.sin2_13_io
                                                         + self.sin2_12 * c2_12 * c2_13 * appo_12))

        aa = np.power(1. - sin2_13_m, 2) * 4. * sin2_12_m * (1. - sin2_12_m)
        bb = (1. - sin2_12_m) * 4 * sin2_13_m * (1. - sin2_13_m)
        cc = sin2_12_m * 4 * sin2_13_m * (1. - sin2_13_m)

        self.matter_prob_io = 1. - aa * self.sin2(x, deltam_21_m) \
                             - bb * self.sin2(x, deltam_31_m) \
                             - cc * self.sin2(x, deltam_32_m)

        if plot_this:
            ylabel = r'$P_{\textrm{mat}} (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$'
            if nu_energy is not None:
                plot_function(x_=[nu_energy], y_=[self.matter_prob_io], label_=[r'IO'], styles=[style["IO1"]],
                              ylabel_=ylabel, xlim=[1.5, 10.5], ylim=[0.08, 1.02])
            else:
                xlabel = label_LE
                plot_function(x_=[x / 1000.], y_=[self.matter_prob_io], label_=[r'IO'], styles=[style["IO1"]],
                              ylabel_=ylabel, xlabel_=xlabel, xlim=[0.04, 35], ylim=[0.08, 1.02], logx=True)

        return self.matter_prob_io

    def eval_matter_prob(self, nu_energy=None, plot_this=False):

        self.eval_matter_prob_no(nu_energy=nu_energy)
        self.eval_matter_prob_io(nu_energy=nu_energy)

        if plot_this:
            ylabel = r'$P_{\textrm{mat}} (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$'
            if nu_energy is not None:
                plot_function(x_=[nu_energy, nu_energy], y_=[self.matter_prob_no, self.matter_prob_io],
                              label_=[r'NO', r'IO'], styles=[style["NO"], style["IO2"]],
                              ylabel_=ylabel, xlim=[1.5, 10.5], ylim=[0.08, 1.02])
            else:
                x = np.arange(0.01, 500000, 1.)  # [m/MeV]
                xlabel = label_LE
                plot_function(x_=[x / 1000., x / 1000.], y_=[self.matter_prob_no, self.matter_prob_io],
                              label_=[r'NO', r'IO'], styles=[style["NO"], style["IO2"]],
                              ylabel_=ylabel, xlabel_=xlabel, xlim=[0.04, 35], ylim=[0.08, 1.02], logx=True)

        return self.matter_prob_no, self.matter_prob_io

    ### evaluation of the survival probability with the same formula as above written in an other way
    ### see JHEP05(2013)131, eq. (2.6), Japanese (https://arxiv.org/abs/1210.8141)
    def eval_prob_jhep(self, energy, ordering, plot_this=False):

        if (ordering < -1) or (ordering > 1):
            print('Error: use 1 for NO, -1 for IO, 0 for both')
            return -1

        x = np.arange(0.01, 500000, 1.)  # [m/MeV]
        x_en = self.baseline * 1000 / energy  # [m/MeV]

        # Normal Ordering
        if (ordering == 1) or (ordering == 0):
            a_n = math.pow(1 - self.sin2_13_no, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
            b_n = 4 * self.sin2_13_no * (1 - self.sin2_13_no)
            c_n = self.sin2_12 * b_n
            d_n = self.sin2_12 * b_n / 2.

            self.jhep_prob_no = 1. - a_n * self.sin2(x, self.deltam_21) - b_n * self.sin2(x, abs(self.deltam_3l_no)) \
                                  - c_n * self.sin2(x, self.deltam_21) * self.cos(x, 2 * abs(self.deltam_3l_no)) \
                                  + d_n * self.sin(x, 2 * abs(self.deltam_3l_no)) * self.sin(x, 2 * self.deltam_21)
            self.jhep_prob_E_no = 1. - a_n * self.sin2(x_en, self.deltam_21) - b_n * self.sin2(x_en, abs(self.deltam_3l_no)) \
                                    - c_n * self.sin2(x_en, self.deltam_21) * self.cos(x_en, 2 * abs(self.deltam_3l_no)) \
                                    + d_n * self.sin(x_en, 2 * abs(self.deltam_3l_no)) * self.sin(x_en, 2 * self.deltam_21)

        # ionverted Ordering
        if (ordering == -1) or (ordering == 0):
            a_i = math.pow(1 - self.sin2_13_io, 2) * 4. * self.sin2_12 * (1 - self.sin2_12)
            b_i = 4 * self.sin2_13_io * (1 - self.sin2_13_io)
            c_i = self.sin2_12 * b_i
            d_i = self.sin2_12 * b_i / 2.

            delta_appo = self.deltam_21 + self.deltam_3l_io
            print(delta_appo)

            self.jhep_prob_io = 1. - a_i * self.sin2(x, self.deltam_21) - b_i * self.sin2(x, abs(delta_appo)) \
                                  - c_i * self.sin2(x, self.deltam_21) * self.cos(x, 2 * abs(delta_appo)) \
                                  - d_i * self.sin(x, 2 * abs(delta_appo)) * self.sin(x, 2 * self.deltam_21)
            self.jhep_prob_E_io = 1. - a_i * self.sin2(x_en, self.deltam_21) - b_i * self.sin2(x_en, abs(delta_appo)) \
                                    - c_i * self.sin2(x_en, self.deltam_21) * self.cos(x_en, 2 * abs(delta_appo)) \
                                    - d_i * self.sin(x_en, 2 * abs(delta_appo)) * self.sin(x_en, 2 * self.deltam_21)

        if plot_this:

            loc = plticker.MultipleLocator(base=2.0)
            loc1 = plticker.MultipleLocator(base=0.5)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid()
            ax.set_xlim(left=0.02, right=80)
            ax.set_xlabel(label_LE)
            ax.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax.set_title(r'Survival probability' + '\n(jhep05(2013)131)')
            ax.set_xlim(left=0.04, right=60)
            ax.set_ylim(0.08, 1.02)

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.grid(alpha=0.45)
            ax1.set_xlabel(r'$E_{\nu}$ [MeV]')
            ax1.set_ylabel(r'$P (\bar{\nu}_{e} \rightarrow \bar{\nu}_{e})$')
            ax1.set_title(r'Survival probability' + '\n(jhep05(2013)131)')
            ax1.set_xlim(1.5, 10.5)
            ax1.set_ylim(0.08, 1.02)
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_minor_locator(loc1)
            ax1.tick_params('both', direction='out', which='both')

            if ordering == 1:  # NO

                ax.semilogx(x / 1000, self.jhep_prob_no, 'b', linewidth=1, label='NO')
                ax.legend(loc='lower left')
                ax1.plot(energy, self.jhep_prob_E_no, 'b', linewidth=1, label='NO')
                ax1.legend(loc='lower right')

            if ordering == -1:  # IO
                ax.semilogx(x / 1000, self.jhep_prob_io, 'r', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                ax1.plot(energy, self.jhep_prob_E_io, 'r', linewidth=1, label='IO')
                ax1.legend(loc='lower right')

            if ordering == 0:  # both NO and IO
                ax.semilogx(x / 1000, self.jhep_prob_no, 'b', linewidth=1, label='NO')
                ax.semilogx(x / 1000, self.jhep_prob_io, 'r--', linewidth=1, label='IO')
                ax.legend(loc='lower left')
                ax1.plot(energy, self.jhep_prob_E_no, 'b', linewidth=1, label='NO')
                ax1.plot(energy, self.jhep_prob_E_io, 'r--', linewidth=1, label='IO')
                ax1.legend(loc='lower right')

        if ordering == 1:
            return self.jhep_prob_E_no
        if ordering == -1:
            return self.jhep_prob_E_io
        if ordering == 0:
            return self.jhep_prob_E_no, self.jhep_prob_E_io
