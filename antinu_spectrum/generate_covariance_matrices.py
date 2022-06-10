import numpy as np
from scipy import interpolate, stats
import pandas as pd


# TODO:
# - add partial correlation in one uncertainty from Mueller (also for EF)

path_to_input = '/Users/beatricejelmini/Desktop/JUNO/JUNO_codes/JUNO_ReactorNeutrinosAnalysis/data/spectra/'

names_huber = ["energy", "spectrum", "neg_stat", "pos_stat", "neg_bias", "pos_bias", "neg_z", "pos_z", "neg_wm",
               "pos_wm", "neg_norm", "pos_norm", "neg_tot", "pos_tot"]
names_mueller_238 = ["energy", "spectrum", "nuclear_db", "forbidden_treatment", "corrections", "missing_info"]

huber_diagonal = {
    'stat': True,
    'bias': True,
    'z': False,
    'wm': False,
    'norm': False
}

mueller_diagonal = {
    'nuclear_db': True,
    'forbidden_treatment': True,
    'corrections': False,
    'missing_info': True,
}


def get_covariance(df_, uncert_, diagonal_=False, absolute_=True, scale_=None, huber=False):
    n_ = len(df_.index)
    appo_matrix = np.zeros((n_, n_))

    if huber:
        appo_array = np.maximum(abs(df_["neg_"+uncert_]), df_["pos_"+uncert_])
    else:
        appo_array = df_[uncert_]

    if type(appo_array) is not np.ndarray:
        appo_array = appo_array.to_numpy()

    if absolute_ and scale_ is None:
        spectrum = df_["spectrum"].to_numpy()
    elif scale_ is not None:
        if type(scale_) is not np.ndarray:
            scale_ = scale_.to_numpy()
        spectrum = df_["spectrum"].to_numpy() * scale_
    else:
        spectrum = np.ones(n_,)

    if diagonal_:
        for i_ in np.arange(n_):
            appo_matrix[i_][i_] = appo_array[i_] * appo_array[i_] * spectrum[i_] * spectrum[i_]
    else:
        for i_ in np.arange(n_):
            for j_ in np.arange(n_):
                appo_matrix[i_][j_] = appo_array[i_] * appo_array[j_] * spectrum[i_] * spectrum[j_]

    return appo_matrix


def generate_from_huber(isotope_, uncertainty_="total", absolute_=True, scale_=None):
    file_end = '_huber.csv'
    if isotope_ == 'U235' or isotope_ == '235U' or isotope_ == '235':
        file = path_to_input + 'u235' + file_end
    elif isotope_ == 'Pu239' or isotope_ == '239Pu' or isotope_ == '239':
        file = path_to_input + 'pu239' + file_end
    elif isotope_ == 'Pu241' or isotope_ == '241Pu' or isotope_ == '241':
        file = path_to_input + 'pu241' + file_end
    else:
        isotopes = ['U235', '235U', '235', 'Pu239', '239Pu', '239', 'Pu241', '241Pu', '241']
        raise ValueError("Invalid isotope_ value. Expected one of: %s" % isotopes)

    input_df = pd.read_csv(file, sep='\t', skiprows=19, header=None, index_col=0, names=names_huber)

    uncert = ["stat", "bias", "z", "wm", "norm", "total"]
    if uncertainty_ == 'total':
        n = len(input_df.index)
        cov_matrix = np.zeros((n, n))
        for uncert_ in uncert[:-1]:
            cov_matrix += get_covariance(input_df, uncert_, diagonal_=huber_diagonal[uncert_], absolute_=absolute_,
                                         scale_=scale_, huber=True)
    elif uncertainty_ in uncert:
        cov_matrix = get_covariance(input_df, uncertainty_, diagonal_=huber_diagonal[uncertainty_], absolute_=absolute_,
                                    scale_=scale_, huber=True)
    else:
        raise ValueError("Invalid uncertainty_ value. Expected one of: %s" % uncert)

    return cov_matrix


def generate_from_ef(isotope_, uncertainty_="total", absolute_=True, scale_=None):
    file_end = '_EF_rebinned.csv'
    if isotope_ == 'U238' or isotope_ == '238U' or isotope_ == '238':
        file = path_to_input + 'u238' + file_end
    else:
        isotopes = ['U238', '238U', '238']
        raise ValueError("Invalid isotope_ value. Expected one of: %s" % isotopes)

    input_df = pd.read_csv(file, sep=',', skiprows=1, header=None, index_col=0, names=names_mueller_238)

    uncert = ["nuclear_db", "forbidden_treatment", "corrections", "missing_info", "total"]
    if uncertainty_ == 'total':
        n = len(input_df.index)
        cov_matrix = np.zeros((n, n))
        for uncert_ in uncert[:-1]:
            cov_matrix += get_covariance(input_df, uncert_, diagonal_=mueller_diagonal[uncert_], absolute_=absolute_,
                                         scale_=scale_, huber=False)
    elif uncertainty_ in uncert:
        cov_matrix = get_covariance(input_df, uncertainty_, diagonal_=mueller_diagonal[uncertainty_],
                                    absolute_=absolute_, scale_=scale_, huber=False)
    else:
        raise ValueError("Invalid uncertainty_ value. Expected one of: %s" % uncert)

    return cov_matrix


def generate_from_mueller(isotope_, uncertainty_="total", absolute_=True, scale_=None):
    file_end = '_mueller.csv'
    if isotope_ == 'U238' or isotope_ == '238U' or isotope_ == '238':
        file = path_to_input + 'u238' + file_end
    else:
        isotopes = ['U238', '238U', '238']
        raise ValueError("Invalid isotope_ value. Expected one of: %s" % isotopes)

    input_df = pd.read_csv(file, sep=',', skiprows=1, header=None, index_col=0, names=names_mueller_238)

    uncert = ["nuclear_db", "forbidden_treatment", "corrections", "missing_info", "total"]
    if uncertainty_ == 'total':
        n = len(input_df.index)
        cov_matrix = np.zeros((n, n))
        for uncert_ in uncert[:-1]:
            cov_matrix += get_covariance(input_df, uncert_, diagonal_=mueller_diagonal[uncert_], absolute_=absolute_,
                                         scale_=scale_, huber=False)
    elif uncertainty_ in uncert:
        cov_matrix = get_covariance(input_df, uncertainty_, diagonal_=mueller_diagonal[uncertainty_],
                                    absolute_=absolute_, scale_=scale_, huber=False)
    else:
        raise ValueError("Invalid uncertainty_ value. Expected one of: %s" % uncert)

    return cov_matrix


def evaluate_cov_matrix_from_samples(samples_, central_values_):
    if type(central_values_) is not np.ndarray:
        raise TypeError(f"Expected central_values_ as numpy.ndarray, got {type(central_values_)} instead.")

    n_samples = samples_.shape[0]
    n_bins = len(central_values_)
    matrix = np.zeros((n_bins, n_bins))

    # print('evaluate_cov_matrix: samples: %s' % n_samples)

    for j_ in np.arange(0, n_bins):
        for k_ in np.arange(0, n_bins):
            matrix[j_, k_] = ((samples_[:, j_] - central_values_[j_]) * (samples_[:, k_] - central_values_[k_])).sum()
    matrix = matrix / n_samples

    return matrix


# def samples_from_covariance_matrix():
#
#
#
#     return


def reshape_cov_matrix(new_bins_, old_bins_, central_values_, cov_matrix_, n_samples_=50):

    pure_samples_ = stats.multivariate_normal.rvs(mean=central_values_, cov=cov_matrix_, size=n_samples_)
    pure_samples_ = pure_samples_[pure_samples_.min(axis=1) >= 0, :]  # get rid of negative values
    n_samples_ = pure_samples_.shape[0]
    reshaped_samples_ = np.zeros((n_samples_, len(new_bins_)))

    for n_ in np.arange(n_samples_):
        f_appo_ = interpolate.interp1d(old_bins_, np.log(pure_samples_[n_]), kind='linear', fill_value="extrapolate")
        reshaped_samples_[n_] = np.exp(f_appo_(new_bins_))

    new_central_values_ = np.exp(
        interpolate.interp1d(old_bins_, np.log(central_values_), kind='linear', fill_value="extrapolate")(new_bins_))

    new_cov_ = evaluate_cov_matrix_from_samples(reshaped_samples_, new_central_values_)

    return new_cov_, n_samples_  # , pure_samples_
