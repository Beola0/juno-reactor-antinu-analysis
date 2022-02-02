import numpy as np
import pandas as pd
import sys
import math
import uproot
from scipy.interpolate import interp1d
from plot import plot_function


# TODO:
# - initialize class with .json file with fission fractions --> DONE
# - create spectrum from inputs, like DYB spectrum --> DONE
# - remove part for plotting --> improved, DONE
# - add methods to change fission fractions --> DONE
# - move IBD part to DetectorResponse (???) --> NOPE
# - add SNF and NonEq contributions --> DONE
# - add nuisances: for SNF and NonEq
# - check interpolation and extrapolation methods for DYB-based reactor model - WIP
# - correct NonEq for DYB model --> DONE
# - include all possible scenarios for reactor model: EF tabulated data, HM tabulated data,
#   SNF applied at the end, NonEq correction applied only to some isotopes, time dependence, ...

HEADER = '\033[95m'
BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
NC = '\033[0m'

names_huber = ["energy", "spectrum", "neg_stat", "pos_stat", "neg_bias", "pos_bias", "neg_z", "pos_z", "neg_wm",
               "pos_wm", "neg_norm", "pos_norm", "neg_tot", "pos_tot"]
names_mueller = ["energy", "spectrum"]
names_ef = ["energy", "spectrum"]
names_dyb = ["energy", "IBD_spectrum", "spectrum"]


class UnoscillatedReactorSpectrum:

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

        self.verbose = inputs_json_["verbose"]
        self.root_file = inputs_json_["ROOT_file"]
        self.path_to_input_spectra = inputs_json_["input_spectra"]
        self.inputs_json = inputs_json_

        self.u5_huber = pd.DataFrame()
        self.pu9_huber = pd.DataFrame()
        self.pu1_huber = pd.DataFrame()
        self.u5_mueller = pd.DataFrame()
        self.u8_mueller = pd.DataFrame()
        self.pu9_mueller = pd.DataFrame()
        self.pu1_mueller = pd.DataFrame()
        self.u5_ef = pd.DataFrame()
        self.u8_ef = pd.DataFrame()
        self.pu9_ef = pd.DataFrame()
        self.pu1_ef = pd.DataFrame()
        self.u5_dyb = pd.DataFrame()
        self.pu_combo_dyb = pd.DataFrame()
        self.total_dyb = pd.DataFrame()

        self.xs_strumiavissani_commoninputs = pd.DataFrame()
        self.xs_vogelbeacom_commoninputs = pd.DataFrame()

        self.iso_spectrum = 0.
        self.react_spectrum = 0.
        self.react_flux = 0.
        self.x_sec = 0.
        self.x_sec_np = 0.
        self.spectrum_unosc = 0.
        self.proton_number = 0.
        self.snf = 0.
        self.noneq = 0.
        self.dybfluxdump = 0.

        self.bool_snf = False
        self.bool_noneq = False
        # self.which_xsec = ''
        self.which_isospectrum = ''

    #####################
    # Fission fractions #
    #####################
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

    ####################
    # Fission energies #
    ####################
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

    ############################
    # Thermal power + baseline #
    ############################
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

    ### from DYB arXiv:1607.05378 - common inputs
    def get_snf_ratio(self, nu_energy_):
        if self.verbose:
            print(f"{BLUE}Reading SNF from Common Inputs{NC}")

        input_ = uproot.open(self.root_file + ":SNF_FluxRatio").to_numpy()

        xx = np.zeros(len(input_[0]))
        for i_ in np.arange(len(xx)):
            xx[i_] = (input_[1][i_ + 1] + input_[1][i_]) / 2.

        f_appo = interp1d(xx, input_[0])
        self.snf = f_appo(nu_energy_)
        return self.snf

    ### from DYB arXiv:1607.05378 - common inputs
    def get_noneq_ratio(self, nu_energy_):
        if self.verbose:
            print(f"{BLUE}Reading NonEq from Common Inputs{NC}")

        input_ = uproot.open(self.root_file + ":NonEq_FluxRatio").to_numpy()

        xx = np.zeros(len(input_[0]))
        for i_ in np.arange(len(xx)):
            xx[i_] = (input_[1][i_ + 1] + input_[1][i_]) / 2.

        f_appo = interp1d(xx, input_[0])
        self.noneq = f_appo(nu_energy_)
        return self.noneq

    ### from common inputs
    def get_dybfluxbump_ratio(self, nu_energy_):
        if self.verbose:
            print(f"{BLUE}Reading DYB flux bump ratio from Common Inputs{NC}")

        # input_[0] are the values(), input_[1] are axis().edges()
        input_ = uproot.open(self.root_file + ":DYBFluxBump_ratio").to_numpy()

        xx = np.zeros(len(input_[0]))
        for i_ in np.arange(len(xx)):
            xx[i_] = (input_[1][i_+1] + input_[1][i_])/2.

        f_appo = interp1d(xx, input_[0])
        self.dybfluxdump = f_appo(nu_energy_)
        return self.dybfluxdump

    ########################
    # Huber tabulated data #
    ########################
    def get_235u_huber(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Huber 235U{NC}")
        self.u5_huber = pd.read_csv(self.path_to_input_spectra+"U235-anti-neutrino-flux-250keV.txt", sep='\t', skiprows=19,
                                    header=None, index_col=0, names=names_huber)
        return self.u5_huber

    def get_239pu_huber(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Huber 239Pu{NC}")
        self.pu9_huber = pd.read_csv(self.path_to_input_spectra + "Pu239-anti-neutrino-flux-250keV.txt", sep='\t',
                                     skiprows=19, header=None, index_col=0, names=names_huber)
        return self.pu9_huber

    def get_241pu_huber(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Huber 241Pu{NC}")
        self.pu1_huber = pd.read_csv(self.path_to_input_spectra + "Pu241-anti-neutrino-flux-250keV.txt", sep='\t',
                                     skiprows=19, header=None, index_col=0, names=names_huber)
        return self.pu1_huber

    ##########################
    # Mueller tabulated data #
    ##########################
    def get_235u_mueller(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Mueller 235U{NC}")
        self.u5_mueller = pd.read_csv(self.path_to_input_spectra+"u235_mueller.csv", sep=',', skiprows=1,
                                      header=None, index_col=0, names=names_mueller)
        return self.u5_mueller

    def get_238u_mueller(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Mueller 238U{NC}")
        self.u8_mueller = pd.read_csv(self.path_to_input_spectra+"u238_mueller.csv", sep=',', skiprows=1,
                                      header=None, index_col=0, names=names_mueller)
        return self.u8_mueller

    def get_239pu_mueller(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Mueller 239Pu{NC}")
        self.pu9_mueller = pd.read_csv(self.path_to_input_spectra+"pu239_mueller.csv", sep=',', skiprows=1,
                                       header=None, index_col=0, names=names_mueller)
        return self.pu9_mueller

    def get_241pu_mueller(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Mueller 241Pu{NC}")
        self.pu1_mueller = pd.read_csv(self.path_to_input_spectra+"pu241_mueller.csv", sep=',', skiprows=1,
                                       header=None, index_col=0, names=names_mueller)
        return self.pu1_mueller

    ##################################
    # Estienne-Fallot tabulated data #
    ##################################
    def get_235u_ef(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Estienne-Fallot 235U{NC}")
        self.u5_ef = pd.read_csv(self.path_to_input_spectra + "u235_EF.csv", sep=',', skiprows=1,
                                 header=None, index_col=0, names=names_ef)
        return self.u5_ef

    def get_238u_ef(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Estienne-Fallot 238U{NC}")
        self.u8_ef = pd.read_csv(self.path_to_input_spectra + "u238_EF.csv", sep=',', skiprows=1,
                                 header=None, index_col=0, names=names_ef)
        return self.u8_ef

    def get_239pu_ef(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Estienne-Fallot 239Pu{NC}")
        self.pu9_ef = pd.read_csv(self.path_to_input_spectra + "pu239_EF.csv", sep=',', skiprows=1,
                                  header=None, index_col=0, names=names_ef)
        return self.pu9_ef

    def get_241pu_ef(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading Estienne-Fallot 241Pu{NC}")
        self.pu1_ef = pd.read_csv(self.path_to_input_spectra + "pu241_EF.csv", sep=',', skiprows=1,
                                  header=None, index_col=0, names=names_ef)
        return self.pu1_ef

    ########################
    # DYB unfolded spectra #
    ########################
    def get_235u_dyb(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading DYB unfolded 235U{NC}")
        self.u5_dyb = pd.read_csv(self.path_to_input_spectra + "u235_unfolded_DYB.csv", sep=',', skiprows=1,
                                  header=None, index_col=0, names=names_ef)
        return self.u5_dyb

    def get_pu_combo_dyb(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading DYB unfolded Pu combo{NC}")
        self.pu_combo_dyb = pd.read_csv(self.path_to_input_spectra + "pu_combo_unfolded_DYB.csv", sep=',', skiprows=1,
                                        header=None, index_col=0, names=names_ef)
        return self.pu_combo_dyb

    def get_total_dyb(self):
        if self.verbose:
            print(f"\n{BLUE}Isotopic spectrum: Reading DYB unfolded total{NC}")
        self.total_dyb = pd.read_csv(self.path_to_input_spectra + "total_unfolded_DYB.csv", sep=',', skiprows=1,
                                     header=None, index_col=0, names=names_ef)
        return self.total_dyb

    ###############################
    # Evaluating isotopic spectra #
    ###############################
    @staticmethod
    def interpolating_function(df_, x_, loc_):
        f_appo = interp1d(df_.index, df_.iloc[:, loc_].apply(lambda x: math.log(x)),
                          kind='linear', fill_value="extrapolate")
        return np.exp(f_appo(x_))

    def eval_235u(self, nu_energy_, which_input='DYB'):
        if self.verbose:
            print(f"\n{CYAN}Isotopic spectrum: Using {which_input} for 235U{NC}")
        if which_input == 'Huber':
            if self.u5_huber.empty:
                self.get_235u_huber()
            return self.interpolating_function(self.u5_huber, nu_energy_, 0)
        elif which_input == 'Mueller':
            if self.u5_mueller.empty:
                self.get_235u_mueller()
            return self.interpolating_function(self.u5_mueller, nu_energy_, 0)
        elif which_input == 'EF':
            if self.u5_ef.empty:
                self.get_235u_ef()
            return self.interpolating_function(self.u5_ef, nu_energy_, 0)
        elif which_input == 'DYB':
            if self.u5_dyb.empty:
                self.get_235u_dyb()
            return self.interpolating_function(self.u5_dyb, nu_energy_, 1)
        else:
            print("Error")
            sys.exit()

    def eval_238u(self, nu_energy_, which_input='Mueller'):
        if self.verbose:
            print(f"\n{CYAN}Isotopic spectrum: Using {which_input} for 238U{NC}")
        if which_input == 'Mueller':
            if self.u8_mueller.empty:
                self.get_238u_mueller()
            return self.interpolating_function(self.u8_mueller, nu_energy_, 0)
        elif which_input == 'EF':
            if self.u8_ef.empty:
                self.get_238u_ef()
            return self.interpolating_function(self.u8_ef, nu_energy_, 0)
        else:
            print("Error")
            sys.exit()

    def eval_239pu(self, nu_energy_, which_input='Huber'):
        if self.verbose:
            print(f"\n{CYAN}Isotopic spectrum: Using {which_input} for 239Pu/PuCombo{NC}")
        if which_input == 'Huber':
            if self.pu9_huber.empty:
                self.get_239pu_huber()
            return self.interpolating_function(self.pu9_huber, nu_energy_, 0)
        elif which_input == 'Mueller':
            if self.pu9_mueller.empty:
                self.get_239pu_mueller()
            return self.interpolating_function(self.pu9_mueller, nu_energy_, 0)
        elif which_input == 'EF':
            if self.pu9_ef.empty:
                self.get_239pu_ef()
            return self.interpolating_function(self.pu9_ef, nu_energy_, 0)
        elif which_input == 'DYB_combo':
            if self.pu_combo_dyb.empty:
                self.get_pu_combo_dyb()
            return self.interpolating_function(self.pu_combo_dyb, nu_energy_, 1)
        else:
            print("Error")
            sys.exit()

    def eval_241pu(self, nu_energy_, which_input='Huber'):
        if self.verbose:
            print(f"\n{CYAN}Isotopic spectrum: Using {which_input} for 241Pu{NC}")
        if which_input == 'Huber':
            if self.pu1_huber.empty:
                self.get_241pu_huber()
            return self.interpolating_function(self.pu1_huber, nu_energy_, 0)
        elif which_input == 'Mueller':
            if self.pu1_mueller.empty:
                self.get_241pu_mueller()
            return self.interpolating_function(self.pu1_mueller, nu_energy_, 0)
        elif which_input == 'EF':
            if self.pu1_ef.empty:
                self.get_241pu_ef()
            return self.interpolating_function(self.pu1_ef, nu_energy_, 0)
        else:
            print("Error")
            sys.exit()

    def eval_total(self, nu_energy_, which_input="DYB"):
        if self.verbose:
            print(f"\n{CYAN}Isotopic spectrum: Using {which_input} for total{NC}")
        if which_input == 'DYB':
            if self.total_dyb.empty:
                self.get_total_dyb()
            return self.interpolating_function(self.total_dyb, nu_energy_, 1)
        else:
            print("Error")
            sys.exit()

    #####################
    # Parametric Models #
    #####################
    def isotopic_spectrum_vogel_parametric(self, nu_energy_, bool_noneq=False, plot_this=False):

        self.which_isospectrum = 'V'
        if self.verbose:
            print(f"\n{CYAN}Using Vogel isotopic spectra with exp-pol parametrization{NC}")

        ### params taken from Vogel, Engel, PRD 39-11 pp 3378, 1989
        ### exponential of a polynomial of second order
        params_u235 = [0.870, -0.160, -0.091]
        params_pu239 = [0.896, -0.239, -0.0981]
        params_u238 = [0.976, -0.162, -0.0790]
        params_pu241 = [0.793, -0.080, -0.1085]
        u235 = self.isotopic_spectrum_exp(nu_energy_, params_u235)
        pu239 = self.isotopic_spectrum_exp(nu_energy_, params_pu239)
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        self.iso_spectrum = self.fiss_frac_235u * u235 + self.fiss_frac_239pu * pu239 \
                            + self.fiss_frac_238u * u238 + self.fiss_frac_241pu * pu241

        if bool_noneq:
            self.bool_noneq = True
            if self.verbose:
                print(f"\n{CYAN}Adding NonEq contribution{NC}")
            if not np.any(self.noneq):
                self.get_noneq_ratio(nu_energy_)
            self.iso_spectrum = self.iso_spectrum + self.noneq * self.iso_spectrum
        else:
            self.bool_noneq = False

        if plot_this:
            ylabel = r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (Vogel)'
            plot_function(x_=[nu_energy_, nu_energy_, nu_energy_, nu_energy_, nu_energy_],
                          y_=[self.iso_spectrum, u235, pu239, u238, pu241],
                          label_=['Total', r'$^{235}$U', r'$^{239}$Pu', r'$^{238}$U', r'$^{241}$Pu'],
                          styles=['k', 'b--', 'r-.', 'g:', 'y'],
                          ylabel_=ylabel, xlim=[1.5, 10.5])

        return self.iso_spectrum

    def isotopic_spectrum_hubermueller_parametric(self, nu_energy_, bool_noneq=False, plot_this=False):

        self.which_isospectrum = 'HM'
        if self.verbose:
            print(f"\n{CYAN}Using Huber+Mueller isotopic spectra with exp-pol parametrization{NC}")

        ### params taken from Mueller PRC 83 (2011) for 238U and Huber PRC 84 (2011) for others
        ### exponential of a polynomial of fifth order
        params_u235 = [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]
        params_pu239 = [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]
        params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
        params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]
        u235 = self.isotopic_spectrum_exp(nu_energy_, params_u235)
        pu239 = self.isotopic_spectrum_exp(nu_energy_, params_pu239)
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        self.iso_spectrum = self.fiss_frac_235u * u235 + self.fiss_frac_239pu * pu239 \
                            + self.fiss_frac_238u * u238 + self.fiss_frac_241pu * pu241

        if bool_noneq:
            self.bool_noneq = True
            if self.verbose:
                print(f"\n{CYAN}Adding NonEq contribution{NC}")
            if not np.any(self.noneq):
                self.get_noneq_ratio(nu_energy_)
            self.iso_spectrum = self.iso_spectrum + self.noneq * self.iso_spectrum
        else:
            self.bool_noneq = False

        if plot_this:
            ylabel = r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (H+M)'
            plot_function(x_=[nu_energy_, nu_energy_, nu_energy_, nu_energy_, nu_energy_],
                          y_=[self.iso_spectrum, u235, pu239, u238, pu241],
                          label_=['Total', r'$^{235}$U', r'$^{239}$Pu', r'$^{238}$U', r'$^{241}$Pu'],
                          styles=['k', 'b--', 'r-.', 'g:', 'y'],
                          ylabel_=ylabel, xlim=[1.5, 10.5])

        return self.iso_spectrum

    def isotopic_spectrum_dyb(self, nu_energy_, bool_noneq=False, plot_this=False):

        self.which_isospectrum = 'DYB'
        if self.verbose:
            print(f"\n{CYAN}Using DYB isotopic spectra (default){NC}")

        ### params taken from Mueller PRC 83 (2011) for 238U and Huber PRC 84 (2011) for 241Pu
        params_u238 = [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
        params_pu241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]
        u238 = self.isotopic_spectrum_exp(nu_energy_, params_u238)
        pu241 = self.isotopic_spectrum_exp(nu_energy_, params_pu241)

        f235_dyb = 0.564
        f239_dyb = 0.304
        f238_dyb = 0.076
        f241_dyb = 0.056

        df_235 = self.fiss_frac_235u - f235_dyb
        df_239 = self.fiss_frac_239pu - f239_dyb
        df_238 = self.fiss_frac_238u - f238_dyb
        df_241 = self.fiss_frac_241pu - f241_dyb

        ### unfolded spectra from DYB, arXiv:2102.04614
        unfolded_spectrum = pd.read_csv("Inputs/spectra/total_unfolded_DYB.csv", sep=",", skiprows=1,
                                        names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=None)
        unfolded_u235 = pd.read_csv("Inputs/spectra/u235_unfolded_DYB.csv", sep=",", skiprows=1,
                                    names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=None)
        unfolded_pu_combo = pd.read_csv("Inputs/spectra/pu_combo_unfolded_DYB.csv", sep=",", skiprows=1,
                                        names=["bin_center", "IBD_spectrum", "isotopic_spectrum"], header=None)

        s_total = interp1d(unfolded_spectrum["bin_center"], unfolded_spectrum["isotopic_spectrum"], kind='cubic')
        s_235 = interp1d(unfolded_u235["bin_center"], unfolded_u235["isotopic_spectrum"], kind='cubic')
        s_combo = interp1d(unfolded_pu_combo["bin_center"], unfolded_pu_combo["isotopic_spectrum"], kind='cubic')

        if bool_noneq:
            self.bool_noneq = True
            if self.verbose:
                print(f"\n{CYAN}Adding NonEq contribution{NC}")
            if not np.any(self.noneq):
                self.get_noneq_ratio(nu_energy_)
            self.iso_spectrum = s_total(nu_energy_) + df_235 * s_235(nu_energy_) + df_239 * s_combo(nu_energy_) \
                                + df_238 * u238 * (1+self.noneq) + (df_241 - 0.183 * df_239) * pu241 * (1+self.noneq)
        else:
            self.bool_noneq = False
            self.iso_spectrum = s_total(nu_energy_) + df_235 * s_235(nu_energy_) + df_239 * s_combo(nu_energy_) \
                                + df_238 * u238 + (df_241 - 0.183 * df_239) * pu241

        if plot_this:
            ylabel = r'$S_{\nu}$ [$\text{N}_{\nu}/\text{fission}/\si{\MeV}$] (DYB)'
            plot_function(x_=[nu_energy_], y_=[self.iso_spectrum], label_=['Total'], styles=['k'],
                          ylabel_=ylabel, xlim=[1.5, 10.5])

        return self.iso_spectrum

    def reactor_spectrum(self, nu_energy_, which_isospectrum='HM', bool_noneq=False, plot_this=False):

        const = 6.241509e21

        if which_isospectrum == 'V':
            self.isotopic_spectrum_vogel_parametric(nu_energy_, bool_noneq=bool_noneq)
        elif which_isospectrum == 'HM':
            self.isotopic_spectrum_hubermueller_parametric(nu_energy_, bool_noneq=bool_noneq)
        elif which_isospectrum == 'DYB':
            self.isotopic_spectrum_dyb(nu_energy_, bool_noneq=bool_noneq)
        else:
            print(f"\n{RED}Error: only 'V', 'VB' or 'DYB' are accepted values for which_isospectrum argument, "
                  f"in reactor_spectrum function, ReactorSpectrum class.{NC}")
            sys.exit()

        en_per_fiss = self.fiss_frac_235u * self.fiss_en_235u + self.fiss_frac_239pu * self.fiss_en_239pu \
                      + self.fiss_frac_238u * self.fiss_en_238u + self.fiss_frac_241pu * self.fiss_en_241pu

        self.react_spectrum = self.thermal_power / en_per_fiss * self.iso_spectrum * const

        if plot_this:
            ylabel = r'$\Phi_{\nu}$ [$\text{N}_{\nu}/\si{\s}/\si{\MeV}$]'
            plot_function(x_=[nu_energy_], y_=[self.react_spectrum], label_=['Reactor spectrum'], styles=['k'],
                          ylabel_=ylabel, xlim=[1.5, 10.5])

        return self.react_spectrum

    def reactor_flux_no_osc(self, nu_energy_, which_isospectrum='HM',
                            bool_snf=False, bool_noneq=False, plot_this=False):

        den = 4. * math.pi * np.power(self.baseline * 1.e5, 2)  # baseline in [cm]

        if which_isospectrum == 'V':
            self.reactor_spectrum(nu_energy_, which_isospectrum=which_isospectrum, bool_noneq=bool_noneq)
        elif which_isospectrum == 'HM':
            self.reactor_spectrum(nu_energy_, which_isospectrum=which_isospectrum, bool_noneq=bool_noneq)
        elif which_isospectrum == 'DYB':
            self.reactor_spectrum(nu_energy_, which_isospectrum=which_isospectrum, bool_noneq=bool_noneq)
        else:
            print(f"\n{RED}Error: only 'V', 'VB' or 'DYB' are accepted values for which_isospectrum argument, "
                  f"in reactor_flux_no_osc function, ReactorSpectrum class.{NC}")
            sys.exit()

        self.react_flux = self.react_spectrum / den

        if bool_snf:
            self.bool_snf = True
            if self.verbose:
                print(f"\n{CYAN}Adding SNF contribution{NC}")
            if not np.any(self.snf):
                self.get_snf_ratio(nu_energy_)
            self.react_flux = self.react_flux + self.react_flux * self.snf
        else:
            self.bool_snf = False

        if plot_this:
            ylabel = r'$\Phi_{\nu}$ [$\text{N}_{\nu}/\si{\s}/\si{\MeV}/\si{\centi\m\squared}$]'
            plot_function(x_=[nu_energy_], y_=[self.react_flux], label_=['Reactor flux'], styles=['k'],
                          ylabel_=ylabel, xlim=[1.5, 10.5])

        return self.react_flux

    #####################
    # IBD cross section #
    #####################
    def eval_n_protons(self):

        target_mass = self.inputs_json["detector"]["mass"] * 1000 * 1000.
        h_mass = self.inputs_json["detector"]["m_H"] * 1.660539066e-27
        h_fraction = self.inputs_json["detector"]["f_H"]
        h1_abundance = self.inputs_json["detector"]["alpha_H"]
        self.proton_number = target_mass * h_fraction * h1_abundance / h_mass

        return self.proton_number

    def get_xs_strumiavissani_commoninputs(self):
        if self.verbose:
            print(f"\n{CYAN}Reading Strumia-Vissani cross section from common inputs{NC}")
        input_ = uproot.open(self.root_file + ":IBDXsec_StrumiaVissani").to_numpy()
        xx = np.zeros(len(input_[0]))
        for i_ in np.arange(len(xx)):
            xx[i_] = (input_[1][i_ + 1] + input_[1][i_]) / 2.
        self.xs_strumiavissani_commoninputs = pd.DataFrame(input_[0], index=xx, columns=["cross_section"])
        self.xs_strumiavissani_commoninputs.index.rename("energy", inplace=True)
        return self.xs_strumiavissani_commoninputs

    def get_xs_vogelbeacom_commoninputs(self):
        if self.verbose:
            print(f"\n{CYAN}Reading Vogel-Beacom cross section from common inputs{NC}")
        input_ = uproot.open(self.root_file + ":IBDXsec_VogelBeacom_DYB").to_numpy()
        xx = np.zeros(len(input_[0]))
        for i_ in np.arange(len(xx)):
            xx[i_] = (input_[1][i_ + 1] + input_[1][i_]) / 2.
        self.xs_vogelbeacom_commoninputs = pd.DataFrame(input_[0], index=xx, columns=["cross_section"])
        self.xs_vogelbeacom_commoninputs.index.rename("energy", inplace=True)
        return self.xs_vogelbeacom_commoninputs

    ### cross section from Strumia, Vissani, https://arxiv.org/abs/astro-ph/0302055, eq. (25)
    ### approximated formula for low energy (below 300 MeV)
    @staticmethod
    def eval_xs_strumiavissani_approx(nu_energy_):

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

        return const * p_e * positron_energy * energy_exp

    def eval_xs(self, nu_energy_, bool_protons=True, which_xs='SV_approx'):

        if which_xs == 'SV_CI':
            if self.verbose:
                print(f"\n{CYAN}Using Strumia-Vissani cross section from common inputs{NC}")
            if self.xs_strumiavissani_commoninputs.empty:
                self.get_xs_strumiavissani_commoninputs()
            f_appo = interp1d(self.xs_strumiavissani_commoninputs.index,
                              self.xs_strumiavissani_commoninputs["cross_section"])
            self.x_sec = f_appo(nu_energy_)
        elif which_xs == 'VB_CI':
            if self.verbose:
                print(f"\n{CYAN}Using Vogel-Beacom cross section from common inputs{NC}")
            if self.xs_vogelbeacom_commoninputs.empty:
                self.get_xs_vogelbeacom_commoninputs()
            f_appo = interp1d(self.xs_vogelbeacom_commoninputs.index,
                              self.xs_vogelbeacom_commoninputs["cross_section"])
            self.x_sec = f_appo(nu_energy_)
        elif which_xs == 'SV_approx':
            if self.verbose:
                print(f"\n{CYAN}Using Strumia-Vissani cross section - approximation for low energies{NC}")
            self.x_sec = self.eval_xs_strumiavissani_approx(nu_energy_)
        else:
            print(f"\n{RED}Error: only 'SV_CI', 'VB_CI' or 'SV_approx' are accepted values for which_xs argument, " 
                  f"in eval_xs function, ReactorSpectrum class.{NC}")
            sys.exit()

        if bool_protons:
            if self.proton_number == 0.:
                self.eval_n_protons()
            self.x_sec_np = self.x_sec * self.proton_number
            return self.x_sec_np

        return self.x_sec

    #################################
    # Unoscillated Reactor Spectrum #
    #################################
    def antinu_spectrum_no_osc(self, nu_energy_, which_isospectrum='HM', which_xs='SV', bool_protons=True,
                               bool_snf=False, bool_noneq=False, plot_this=False):

        if which_isospectrum == 'V':
            self.reactor_flux_no_osc(nu_energy_, which_isospectrum=which_isospectrum,
                                     bool_snf=bool_snf, bool_noneq=bool_noneq)
        elif which_isospectrum == 'HM':
            self.reactor_flux_no_osc(nu_energy_, which_isospectrum=which_isospectrum,
                                     bool_snf=bool_snf, bool_noneq=bool_noneq)
        elif which_isospectrum == 'DYB':
            self.reactor_flux_no_osc(nu_energy_, which_isospectrum=which_isospectrum,
                                     bool_snf=bool_snf, bool_noneq=bool_noneq)
        else:
            print(f"\n{RED}Error: only 'V', 'VB' or 'DYB' are accepted values for which_isospectrum argument, "
                  f"in antinu_spectrum_no_osc function, ReactorSpectrum class.{NC}")
            sys.exit()

        xs = self.eval_xs(nu_energy_, which_xs=which_xs, bool_protons=bool_protons)

        self.spectrum_unosc = self.react_flux * xs

        if plot_this:
            ylabel = r'$S_{\bar{\nu}}$ [N$_{\nu}$/\si{\MeV}/\si{s}]'
            plot_function(x_=[nu_energy_], y_=[self.spectrum_unosc], label_=['spectrum'], styles=['k'],
                          ylabel_=ylabel, xlim=[1.5, 10.5])

        return self.spectrum_unosc
