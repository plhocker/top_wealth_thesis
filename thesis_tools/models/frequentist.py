# Module for fitting of frequentist models to data

# Imports
import numpy as np
import scipy.special
from scipy.optimize import minimize

class Pareto:
    def __init__(
        self, 
        alpha: float=None, 
        x_m: float=1.0
    ):
        self.alpha = alpha
        self.x_m = x_m
    
    def pdf(
        self, 
        x_m: float, 
        alpha: float
    ) -> float:
        """
        Calculate the probability density function of the Pareto distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the probability density function for.
        Returns
        -------
        float
            The probability density function of the Pareto distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if x_m is None:
            print('x_m parameter not set, using parameter stored in the class: x_m=', self.x_m)
            x_m = self.x_m

        return (1/alpha) * x_m**(1/alpha) / x**(1/alpha)

    def likelihood(
        self, 
        data: np.ndarray, 
        x_m:float=None, 
        set_alpha: float=None
    ) -> float:
        """
        Calculate the likelihood of the data given the Pareto distribution.
        The likelihood is calculated as the product of the probability density function evaluated at each data point.
        Parameters
        ----------
        data : np.ndarray
            The data to calculate the likelihood for.
        Returns
        -------
        float
            The likelihood of the data given the Pareto distribution.
        Raises
        ------
        ValueError
            If the alpha parameter is not set.
        """
        
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha

        if x_m is None:
            print('x_m parameter not set, using parameter stored in the class: x_m=', self.x_m)
            x_m = self.x_m
        
        return np.prod(self.pdf(data))

    def fit(
        self, 
        data: np.ndarray,
        given_alpha: float=None,
        given_x_m: float=1.0,
        set_class_parameters: bool=True
    ) -> (float, float):
        """
        Fit the Pareto distribution to the data.
        The method of moments is used to estimate the parameters of the distribution.
        Parameters
        ----------
        data : np.ndarray
            The data to fit the distribution to.
        """
        
        if given_x_m is None:
            estimated_x_m = np.min(data)
        else:
            estimated_x_m = given_x_m

        if given_alpha is not None:
            print('Alpha parameter set to:', self.alpha, 'using given_alpha.')
            if set_class_parameters:
                self.alpha = given_alpha
                self.x_m = estimated_x_m
            return given_alpha, estimated_x_m
        
        estimated_alpha = np.sum(np.log(data / estimated_x_m)) / len(data)

        if set_class_parameters:
            self.alpha = estimated_alpha
            self.x_m = estimated_x_m
        
        return estimated_alpha, estimated_x_m

    def mean(self) -> float:
        """
        Calculate the mean of the Pareto distribution.
        Returns
        -------
        float
            The mean of the Pareto distribution.
        """
        if 1 < 1/self.alpha:
            return self.x_m / (1 - self.alpha)
        else:
            return np.inf

class Gompertz:
    def __init__(
        self, 
        gamma: float=None,
        alpha: float=None
    ):
        self.alpha = alpha
        self.gamma = gamma

    def pdf(
        self, 
        x: float, 
        gamma: float=None,
        alpha: float=None 
    ) -> float:
        """
        Calculate the probability density function of the Gompertz distribution.
        Parameters
        ----------
        x : float
            The data to calculate the probability density function for.
        Returns
        -------
        float
            The probability density function of the Gompertz distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma

        return (1/alpha) * np.exp(gamma * x - (np.exp(gamma * x) - 1) / (alpha * gamma))

    def likelihood(
        self, 
        data: np.ndarray,
        gamma: float=None,
        alpha: float=None 
    ) -> float:
        """
        Calculate the likelihood of the data given the Gompertz distribution.
        The likelihood is calculated as the product of the probability density function evaluated at each data point.
        Parameters
        ----------
        data : np.ndarray
            The data to calculate the likelihood for.
        Returns
        -------
        float
            The likelihood of the data given the Gompertz distribution.
        Raises
        ------
        ValueError
            If the alpha parameter is not set.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma

        return np.prod(self.pdf(data, alpha=alpha, gamma=gamma))

    def log_likelihood(
        self, 
        data: np.ndarray, 
        gamma: float=None,
        alpha: float=None, 
    ) -> float:
        """
        Calculate the log likelihood of the data given the Gompertz distribution.
        The log likelihood is calculated as the sum of the log of the probability density function evaluated at each data point.
        Parameters
        ----------
        data : np.ndarray
            The data to calculate the log likelihood for.
        Returns
        -------
        float
            The log likelihood of the data given the Gompertz distribution.
        Raises
        ------
        ValueError
            If the alpha parameter is not set.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if alpha == 0:
            return -1e15

        if gamma == 0:
            return -1e15

        # Using the results from Teulings & Toussaint (2023) to avoid overflow errors
        N = len(data)
        temp = -np.log(alpha) + gamma * np.mean(data) - (1 / (alpha * gamma)) * (np.mean(np.exp(gamma * data))-1)
        return N * temp

    def fit(
        self, 
        data: np.ndarray, 
        given_gamma: float=None, 
        given_alpha: float=None, 
        set_class_parameters: bool=True
    ) -> (float, float):
        """
        Fit the Gompertz distribution to the data.
        The method of moments is used to estimate the parameters of the distribution.
        Parameters
        ----------
        data : np.ndarray
            The data to fit the distribution to.
        """
        if given_gamma is not None:
            print('Gamma parameter set to:', self.gamma, 'using given_gamma to estimate alpha using Teulings & Toussaint (2023).')
            estimated_alpha = (1 / given_gamma) * (np.mean(np.exp(given_gamma * data)) - 1)
            if set_class_parameters:
                self.alpha = estimated_alpha
                self.gamma = given_gamma
            return given_gamma, estimated_alpha
        
        # Set the correct bounds for the optimization
        if given_alpha is not None:
            alpha_bounds = (given_alpha, given_alpha)
        else:
            alpha_bounds = (0, None)
        gamma_bounds = (0, None)

        # Maximise the log likelihood
        result = minimize(
            lambda x: -self.log_likelihood(data, gamma=x[0], alpha=x[1]), 
            x0=[1, 1], 
            bounds=[gamma_bounds, alpha_bounds],
        ).x

        if set_class_parameters:
            self.gamma = result[0]
            self.alpha = result[1]
        
        return result[0], result[1]

    def mean(self) -> float:
        """
        Calculate the mean of the Gompertz distribution.
        Returns
        -------
        float
            The mean of the Gompertz distribution.
        """
        return (1/self.gamma) * np.exp(1/(self.alpha*self.gamma)) * scipy.special.expi(1/(self.alpha*self.gamma))


class Weibull:
    def __init__(
        self, 
        gamma: float=None,
        alpha: float=None
    ):
        self.alpha = alpha
        self.gamma = gamma
    
    def mean(self) -> float:
        """
        Calculate the mean of the Weibull distribution.
        Returns
        -------
        float
            The mean of the Weibull distribution.
        """
        term1 = (self.alpha * self.gamma) ** (1 / self.gamma)
        term2 = np.exp(1/(self.alpha*self.gamma))
        term3 = scipy.special.gamma(1 + 1 / self.gamma)
        term4 = scipy.special.gammaincc(1 + 1 / self.gamma, 1 / (self.alpha * self.gamma))
    
        return term1 * term2 * term3 * term4