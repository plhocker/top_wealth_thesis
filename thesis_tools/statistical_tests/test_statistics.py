import numpy as np


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