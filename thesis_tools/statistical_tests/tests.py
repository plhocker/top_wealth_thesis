import numpy as np
import pandas as pd
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import ks_2samp

# from thesis_tools.statistical_tests.test_statistics import * Should not import from this anymore
from thesis_tools.utils.data import *
from thesis_tools.models.frequentist import Distribution, Pareto, Exponential, Gompertz, Weibull, GeneralisedPareto

def sample_measurement_errors(
    n_samples: int,
    type: str='empirical_relative',
) -> np.array:
    """
    Sample measurement errors from a given distribution.
    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    type : str
        The type of measurement error to generate.
        Options are 'empirical_relative' and 'empirical_absolute'.
    Returns
    -------
    np.array
        The generated measurement errors.
    Raises
    ------
    ValueError
        If the type is not one of the available options.
    """
    df_forbes = read_billionaires_data(only_years=['2021'])
    df_bloomberg = read_bloomberg_data()
    df_merged = pd.merge(df_forbes, df_bloomberg, on='full_name', how='inner', suffixes=('_forbes', '_bloomberg'))
    df_merged['forbes_over_bloomberg'] = df_merged['net_worth_forbes'] / df_merged['net_worth_bloomberg']
    df_merged['log_forbes_minus_log_bloomberg'] = np.log(df_merged['net_worth_forbes']) - np.log(df_merged['net_worth_bloomberg'])
    df_merged['forbes_over_bloomberg_normalised'] = (df_merged['forbes_over_bloomberg'] / df_merged['forbes_over_bloomberg'].mean())
    df_merged['log_forbes_minus_log_bloomberg_normalised'] = (df_merged['log_forbes_minus_log_bloomberg'] - df_merged['log_forbes_minus_log_bloomberg'].mean())
    # ensure all values are closer to 1
    df_merged['forbes_over_bloomberg_normalised_condensed'] = 1 + (df_merged['forbes_over_bloomberg_normalised'] - 1) / 2
    
    def get_kde_samples(data, n_samples):
        kde = KDEUnivariate(data)
        kde.fit(kernel='gau', bw='normal_reference', fft='True')
        x_vals = np.linspace(min(data), max(data), 1000)
        kde_vals = kde.evaluate(x_vals)
        cdf = np.cumsum(kde_vals)
        cdf = cdf / cdf[-1]  
        random_values = np.random.rand(n_samples)
        sampled_indices = np.searchsorted(cdf, random_values)
        samples = x_vals[sampled_indices]
        return samples

    if type == 'none':
        return np.zeros(n_samples)
    elif type == 'empirical_relative':
        data = df_merged['forbes_over_bloomberg']
        return get_kde_samples(data, n_samples)
    elif type == 'empirical_log':
        data = df_merged['log_forbes_minus_log_bloomberg']
        return get_kde_samples(data, n_samples)
    elif type == 'empirical_relative_trimmed':
        data = df_merged['forbes_over_bloomberg']
        # remove the top and bottom 10% of the data
        data = data[(data > data.quantile(0.1)) & (data < data.quantile(0.9))]
        return get_kde_samples(data, n_samples)
    elif type == 'empirical_log_trimmed':
        data = df_merged['log_forbes_minus_log_bloomberg']
        # remove the top and bottom 10% of the data
        data = data[(data > data.quantile(0.1)) & (data < data.quantile(0.9))]
        return get_kde_samples(data, n_samples)
    elif type == 'empirical_relative_normalised':
        data = df_merged['forbes_over_bloomberg_normalised']
        return get_kde_samples(data, n_samples)
    elif type == 'empirical_log_normalised':
        data = np.log(df_merged['forbes_over_bloomberg_normalised'])
        return get_kde_samples(data, n_samples)
    elif type == 'empirical_relative_normalised_condensed':
        data = df_merged['forbes_over_bloomberg_normalised_condensed']
        return get_kde_samples(data, n_samples)
    else:
        raise ValueError('Measurement error type not recognized.')

def R_stat(
    data: np.ndarray,
    order: int
) -> float:
    """
    Calculate the R-statistic for a given order.
    The R-statistic is a measure of the relative importance of the order-th moment in the data.
    It is defined as the ratio of the order-th moment to the product of the order-th factorial and the mean to the power of the order.
    R_k = mu_k / (k! * mu^k)
    where mu_k is the order-th moment, k is the order and mu is the mean of the data.
    Parameters
    ----------
    data : np.ndarray
        The data to calculate the R-statistic for.
    order : int
        The order of the moment to calculate the R-statistic for.
    Returns
    -------
    float
        The R-statistic for the given order.
    Raises
    ------
    ValueError
        If the order is less than or equal to 0.
    """
    
    if order <= 0:
        raise ValueError('Order must be greater or equal to 1.')

    # if data contains nan values, remove them and raise a warning
    if np.isnan(data).any():
        data = data[~np.isnan(data)]
        print('Warning: data contains nan values, they were removed.')

    mu = np.mean(data)
    mu_k = np.mean(data ** order)
    # k = order
    # R_k = mu_k / (k! * mu^k)
    return mu_k / (np.math.factorial(order) * mu**order)

def R_stat_pareto_test(
    log_data: np.ndarray,
    order: int=2,
    n_samples_empirical_distribution: int=10000,
    round_decimals_non_log: int=None,
    measurement_error: str='None'
) -> (float, float, float, float, np.array):
    """
    Calculate the R-statistic for a given order.
    The R-statistic is a measure of the relative importance of the order-th moment in the data.
    It is defined as the ratio of the order-th moment to the product of the order-th factorial and the mean to the power of the order.
    R_k = mu_k / (k! * mu^k)
    where mu_k is the order-th moment, k is the order and mu is the mean of the data.
    Parameters
    ----------
    data : np.ndarray
        The data to calculate the R-statistic for.
    order : int
        The order of the moment to calculate the R-statistic for.
    Returns
    -------
    float
        The R-statistic for the given order.
    Raises
    ------
    ValueError
        If the order is less than or equal to 0.
    """
    
    if order <= 1:
        raise ValueError('Order must be greater or equal to 2.')

    # if data contains nan values, remove them and raise a warning
    if np.isnan(log_data).any():
        data = data[~np.isnan(log_data)]
        print('Warning: data contains nan values, they were removed.')

    test_stat = R_stat(log_data, order)

    # Calculate the p-value by comparing the test statistic to an empirical distribution
    # Estimated lambda for the Exponential distribution (log)
    alpha_hat = 1 / np.mean(log_data)

    # Add measurement error if needed
    if measurement_error != 'None':
        measurement_errors = sample_measurement_errors(len(log_data)*n_samples_empirical_distribution, measurement_error)
    else:
        measurement_errors = np.zeros(len(log_data)*n_samples_empirical_distribution)

    R_stats_empirical = []
    for i in range(n_samples_empirical_distribution):
        a, m = alpha_hat, 1
        empirical_distribution = (np.random.pareto(a, len(log_data)) + 1) * m
        empirical_distribution = empirical_distribution + measurement_errors[i*len(log_data):(i+1)*len(log_data)]
        # Round if needed
        if round_decimals_non_log is not None:
            empirical_distribution = np.round(empirical_distribution, round_decimals_non_log)
        empirical_distribution = np.log(empirical_distribution)
        R_stats_empirical.append(R_stat(empirical_distribution, order))
    R_stats_empirical = np.array(R_stats_empirical)

    p_value_left = np.mean(R_stats_empirical <= test_stat)
    p_value_right = np.mean(R_stats_empirical >= test_stat)
    p_value_two_sided = 2 * min(p_value_left, p_value_right)

    return test_stat, p_value_left, p_value_right, p_value_two_sided, R_stats_empirical

def Kolmogorov_Smirnov_H_0_test(
    data: np.ndarray,
    H_0_distribution: Distribution,
    n_samples_empirical_distribution: int=100000,
    round_decimals_non_log: int=None,
    measurement_error: str=None
) -> (float, float):
    """
    Perform a Kolmogorov-Smirnov test to test a given null hypothesis.
    The null hypothesis is that the data comes from a given distribution.
    Parameters
    ----------
    data : np.ndarray
        The data to test the null hypothesis for.
    H_0 : str
        The null hypothesis to test.
        Options are 'exponential', 'lognormal', 'pareto', 'weibull', 'normal', 'uniform'.
    Returns
    -------
    float
        The test statistic.
    float
        The p-value.
    float
        The location of the statistic.
    float
        The sign of the statistic.
    """
    
    # Sample a very large number of samples from the null hypothesis distribution
    H_0_samples = H_0_distribution.sample(n_samples_empirical_distribution)

    # Round if needed
    if round_decimals_non_log is not None:
        H_0_samples = np.round(H_0_samples, round_decimals_non_log)
    
    # Add measurement error if needed
    if measurement_error is not None:
        measurement_errors = sample_measurement_errors(n_samples_empirical_distribution, measurement_error)
        # Add measurement error to the samples -> only works for non-log data
        H_0_samples = H_0_samples * measurement_errors 

    # Perform the Kolmogorov-Smirnov test
    test_stat, p_value = Kolmogorov_Smirnov_two_sample_test(data, H_0_samples)
    return test_stat, p_value

def Kolmogorov_Smirnov_two_sample_test(
    data1: np.ndarray,
    data2: np.ndarray
) -> (float, float, float, int):
    """
    Perform a two-sample Kolmogorov-Smirnov test.
    The null hypothesis is that the two samples are drawn from the same distribution.
    Parameters
    ----------
    data1 : np.ndarray
        The first sample.
    data2 : np.ndarray
        The second sample.
    Returns
    -------
    float
        The test statistic.
    float
        The p-value.
    float
        The location of the statistic.
    int
        The sign of the statistic.
    """
    test_stat, p_value = ks_2samp(data1, data2, alternative='two-sided', method='auto')
    return test_stat, p_value

def Anderson_Darling_statistic(
    data_1: np.ndarray,
    data_2: np.ndarray,
    weight_function = 'Cramer-von Mises'
) -> (float, float):
    """
    Calculate the Anderson-Darling statistic for two samples.
    The Anderson-Darling statistic is a measure of how well the data fits a given distribution.
    Parameters
    ----------
    data_1 : np.ndarray
        The first sample.
    data_2 : np.ndarray
        The second sample (should be large).
    weight_function : str
        The weight function to use.
        Options are 'Cramer-von Mises' and 'Anderson-Darling'.
    Returns
    -------
    float
        The Anderson-Darling statistic.
    """

    def F_n(data, x):
        return np.mean(data <= x)

    def w(x):
        if weight_function == 'Cramer-von Mises':
            return 1
        elif weight_function == 'Anderson-Darling':
            if F_n(data_2, x) == 0 or F_n(data_2, x) == 1:
                # return a very large number to avoid division by zero
                return 1e9
            else:
                return 1 / (F_n(data_2, x) * (1 - F_n(data_2, x)))
        else:
            raise ValueError('Weight function not recognized.')

    # Do numerical integration
    lb = min(min(data_1), min(data_2))
    ub = max(max(data_1), max(data_2))
    n_points = 100
    x_vals = np.linspace(lb, ub, n_points)
    integral = 0
    for i in range(1, n_points):
        integral += w(x_vals[i]) * (F_n(data_1, x_vals[i]) - F_n(data_2, x_vals[i]))**2 * (x_vals[i] - x_vals[i-1])
    return len(data_1) * integral, integral

def Anderson_Darling_H_0_test(
    data,
    H_0_distribution: Distribution,
    weight_function: str='Cramer-von Mises',
    n_samples_empirical_distribution: int=10000,
    round_decimals_non_log: int=None,
    measurement_error: str=None
) -> (float, float, float):
    """
    Perform an Anderson-Darling test to test a given null hypothesis.
    The null hypothesis is that the data comes from a given distribution.
    Parameters
    ----------
    data : np.ndarray
        The data to test the null hypothesis for.
    H_0 : str
        The null hypothesis to test.
        Options are 'exponential', 'lognormal', 'pareto', 'weibull', 'normal', 'uniform'.
    Returns
    -------
    float
        The test statistic.
    float
        The p-value.
    """
    
    # Sample a very large number of samples from the null hypothesis distribution
    H_0_samples = H_0_distribution.sample(n_samples_empirical_distribution)

    # Round if needed
    if round_decimals_non_log is not None:
        H_0_samples = np.round(H_0_samples, round_decimals_non_log)
    
    # Add measurement error if needed
    if measurement_error is not None:
        measurement_errors = sample_measurement_errors(n_samples_empirical_distribution, measurement_error)
        # Add measurement error to the samples -> only works for non-log data
        H_0_samples = H_0_samples * measurement_errors 

    # Perform the Anderson-Darling test
    test_stat, normalised_test_stat = Anderson_Darling_statistic(data, H_0_samples, weight_function)

    # Find the empirical distribution of the test statistic
    test_stats_empirical = []
    for i in range(1000):
        H_0_samples_small = H_0_distribution.sample(len(data))
        H_0_samples_large = H_0_distribution.sample(n_samples_empirical_distribution)
        test_stat_empirical, _ = Anderson_Darling_statistic(
            H_0_samples_small, 
            H_0_samples_large, 
            weight_function
        )
        test_stats_empirical.append(test_stat_empirical)

    p_value = np.mean(test_stats_empirical >= test_stat)

    return test_stat, normalised_test_stat, p_value