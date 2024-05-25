# Module for fitting of frequentist models to data
from abc import ABC, abstractmethod

# Imports
import numpy as np
import scipy.special
from scipy.optimize import minimize

class Distribution(ABC):
    @abstractmethod
    def pdf(self):
        pass

    @abstractmethod
    def cdf(self):
        pass

    @abstractmethod
    def likelihood(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def support(self):
        pass

class Pareto(Distribution):
    def __init__(
        self, 
        alpha: float=None, 
        x_m: float=1.0
    ):
        self.alpha = alpha
        self.x_m = x_m
    
    def pdf(
        self,
        x: float, 
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

    def cdf(
        self,
        x: float,
        x_m: float,
        alpha: float
    ) -> float:
        """
        Calculate the cumulative distribution function of the Pareto distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the cumulative distribution function for.
        Returns
        -------
        float
            The cumulative distribution function of the Pareto distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if x_m is None:
            print('x_m parameter not set, using parameter stored in the class: x_m=', self.x_m)
            x_m = self.x_m

        return 1 - (x / x_m)**(-1/alpha)

    def inverse_cdf(
        self,
        p: float,
        x_m: float,
        alpha: float
    ) -> float:
        """
        Calculate the inverse cumulative distribution function of the Pareto distribution.
        Parameters
        ----------
        p : np.ndarray
            The data to calculate the inverse cumulative distribution function for.
        Returns
        -------
        float
            The inverse cumulative distribution function of the Pareto distribution.
        """
        return x_m * (1 - p)**(-alpha)

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

    def sample(
        self,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample from the Pareto distribution.
        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the distribution.
        Returns
        -------
        np.ndarray
            The samples drawn from the Pareto distribution.
        """
        
        # Do inverse transform sampling
        u = np.random.uniform(0, 1, n_samples)
        return self.inverse_cdf(u, self.x_m, self.alpha)

    def mean(
        self
    ) -> float:
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

    def support(
        self
    ) -> (float, float):
        """
        Calculate the support of the Pareto distribution.
        Returns
        -------
        float
            The lower bound of the Pareto distribution.
        float
            The upper bound of the Pareto distribution.
        """
        return self.x_m, np.inf

class Exponential(Distribution):
    def __init__(
        self, 
        alpha: float=None
    ):
        self.alpha = alpha
    
    def pdf(
        self,
        x: float,
        alpha: float=None
    ) -> float:
        """
        Calculate the probability density function of the Exponential distribution.
        Parameters
        ----------
        x : float
            The data to calculate the probability density function for.
        Returns
        -------
        float
            The probability density function of the Exponential distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha

        return (1 / alpha) * np.exp(-x / alpha)

    def cdf(
        self,
        x: float,
        alpha: float=None
    ) -> float:
        """
        Calculate the cumulative distribution function of the Exponential distribution.
        Parameters
        ----------
        x : float
            The data to calculate the cumulative distribution function for.
        Returns
        -------
        float
            The cumulative distribution function of the Exponential distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha

        return 1 - np.exp(-x / alpha)

    def inverse_cdf(
        self,
        p: float,
        alpha: float
    ) -> float:
        """
        Calculate the inverse cumulative distribution function of the Exponential distribution.
        Parameters
        ----------
        p : float
            The data to calculate the inverse cumulative distribution function for.
        Returns
        -------
        float
            The inverse cumulative distribution function of the Exponential distribution.
        """
        return -alpha * np.log(1 - p)

    def likelihood(
        self,
        data: np.ndarray,
        alpha: float=None
    ) -> float:
        """
        Calculate the likelihood of the data given the Exponential distribution.
        The likelihood is calculated as the product of the probability density function evaluated at each data point.
        Parameters
        ----------
        data : np.ndarray
            The data to calculate the likelihood for.
        Returns
        -------
        float
            The likelihood of the data given the Exponential distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha

        return np.prod(self.pdf(data, alpha=alpha))

    def fit(
        self,
        data: np.ndarray,
        given_alpha: float=None,
        set_class_parameters: bool=True
    ) -> float:
        """
        Fit the Exponential distribution to the data.
        The method of moments is used to estimate the parameters of the distribution.
        Parameters
        ----------
        data : np.ndarray
            The data to fit the distribution to.
        """
        if given_alpha is not None:
            print('Alpha parameter set to:', self.alpha, 'using given_alpha.')
            if set_class_parameters:
                self.alpha = given_alpha
            return given_alpha
        
        estimated_alpha = np.mean(data)

        if set_class_parameters:
            self.alpha = estimated_alpha
        
        return estimated_alpha

    def sample(
        self,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample from the Exponential distribution.
        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the distribution.
        Returns
        -------
        np.ndarray
            The samples drawn from the Exponential distribution.
        """
        # Do inverse transform sampling
        u = np.random.uniform(0, 1, n_samples)
        return self.inverse_cdf(u, self.alpha)

    def mean(
        self
    ) -> float:
        """
        Calculate the mean of the Exponential distribution.
        Returns
        -------
        float
            The mean of the Exponential distribution.
        """
        return self.alpha

    def support(
        self
    ) -> (float, float):
        """
        Calculate the support of the Exponential distribution.
        Returns
        -------
        float
            The lower bound of the Exponential distribution.
        float
            The upper bound of the Exponential distribution.
        """
        return 0, np.inf
    
class Gompertz(Distribution):
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

    def cdf(
        self,
        x: float,
        gamma: float=None,
        alpha: float=None
    ) -> float:
        """
        Calculate the cumulative distribution function of the Gompertz distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the cumulative distribution function for.
        Returns
        -------
        float
            The cumulative distribution function of the Gompertz distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma

        return 1 - np.exp(-(np.exp(gamma * x) - 1) / (alpha * gamma))

    def inverse_cdf(
        self,
        p: float,
        gamma: float=None,
        alpha: float=None
    ) -> float:
        """
        Calculate the inverse cumulative distribution function of the Gompertz distribution.
        Parameters
        ----------
        p : np.ndarray
            The data to calculate the inverse cumulative distribution function for.
        Returns
        -------
        float
            The inverse cumulative distribution function of the Gompertz distribution.
        """

        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        return (1 / gamma) * np.log(1 - alpha * gamma * np.log(1 - p))

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

    def sample(
        self,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample from the Gompertz distribution.
        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the distribution.
        Returns
        -------
        np.ndarray
            The samples drawn from the Gompertz distribution.
        """
        # Do inverse transform sampling
        u = np.random.uniform(0, 1, n_samples)
        return self.inverse_cdf(u, self.gamma, self.alpha)

    def mean(
        self
    ) -> float:
        """
        Calculate the mean of the Gompertz distribution.
        Returns
        -------
        float
            The mean of the Gompertz distribution.
        """
        return (1/self.gamma) * np.exp(1/(self.alpha*self.gamma)) * scipy.special.expi(1/(self.alpha*self.gamma))

    def support(
        self
    ) -> (float, float):
        """
        Calculate the support of the Gompertz distribution.
        Returns
        -------
        float
            The lower bound of the Gompertz distribution.
        float
            The upper bound of the Gompertz distribution.
        """
        return 0, np.inf
     
class Weibull(Distribution):
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
        Calculate the probability density function of the Weibull distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the probability density function for.
        Returns
        -------
        float
            The probability density function of the Weibull distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma

        return (1 / alpha) * x**(gamma - 1) * np.exp((1 - x**gamma) / (alpha * gamma)) # My corrected formula

    def cdf(
        self,
        x: float,
        gamma: float=None,
        alpha: float=None
    ) -> float:
        """
        Calculate the cumulative distribution function of the Weibull distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the cumulative distribution function for.
        Returns
        -------
        float
            The cumulative distribution function of the Weibull distribution.
        """
        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma

        # return 1 - np.exp(-((1 + x)**(gamma + 1) - 1) / (alpha * (gamma + 1))) This is the formula from the paper, it seems incorrect
        return 1 - np.exp((1 - x ** gamma) / (alpha * gamma)) # my corrected formula

    def inverse_cdf(
        self,
        p: float,
        gamma: float=None,
        alpha: float=None
    ) -> float:
        """
        Calculate the inverse cumulative distribution function of the Weibull distribution.
        Parameters
        ----------
        p : np.ndarray
            The data to calculate the inverse cumulative distribution function for.
        Returns
        -------
        float
            The inverse cumulative distribution function of the Weibull distribution.
        """

        if alpha is None:
            print('Alpha parameter not set, using parameter stored in the class: alpha=', self.alpha)
            alpha = self.alpha
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        # return (1 - alpha * (gamma + 1) * np.log(1 - p))**(1 / (gamma + 1)) - 1 Based on the wrong formula from the paper
        return (1 - np.log(1 - p) * alpha * gamma)**(1 / gamma) # My corrected formula

    def likelihood(
        self,
        data: np.ndarray,
        gamma: float=None,
        alpha: float=None
    ) -> float:
        """
        Calculate the likelihood of the data given the Weibull distribution.
        The likelihood is calculated as the product of the probability density function evaluated at each data point.
        Parameters
        ----------
        data : np.ndarray
            The data to calculate the likelihood for.
        Returns
        -------
        float
            The likelihood of the data given the Weibull distribution.
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

        return np.prod(self.pdf(data, gamma=gamma, alpha=alpha))

    def fit(
        self,
        data: np.ndarray,
        given_gamma: float=None,
        given_alpha: float=None,
        set_class_parameters: bool=True
    ) -> (float, float):
        """
        Fit the Weibull distribution to the data.
        Fit it by instead fitting a Gompertz distribution to the log_data.
        Parameters
        ----------
        data : np.ndarray
            The data to fit the distribution to.
        """
        log_data = np.log(data)
        temp_Gompertz = Gompertz()
        
        gamma_hat, alpha_hat = temp_Gompertz.fit(
            log_data, 
            given_gamma=given_gamma, 
            given_alpha=given_alpha, 
            set_class_parameters=False
        )
        
        if set_class_parameters:
            self.gamma = gamma_hat
            self.alpha = alpha_hat
        
        return gamma_hat, alpha_hat

    def sample(
        self,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample from the Weibull distribution.
        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the distribution.
        Returns
        -------
        np.ndarray
            The samples drawn from the Weibull distribution.
        """
        # Do inverse transform sampling
        u = np.random.uniform(0, 1, n_samples)
        return self.inverse_cdf(u, self.gamma, self.alpha)

    def mean(
        self
    ) -> float:
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

    def support(
        self
    ) -> (float, float):
        """
        Calculate the support of the Weibull distribution.
        Returns
        -------
        float
            The lower bound of the Weibull distribution.
        float
            The upper bound of the Weibull distribution.
        """
        return 0, np.inf

class GeneralisedPareto(Distribution):
    def __init__(
        self,
        gamma: float=None, 
        sigma: float=None,
        mu: float=0.0
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.mu = mu
    
    def pdf(
        self,
        x: float,
        gamma: float=None,
        sigma: float=None,
        mu: float=None
    ) -> float:
        """
        Calculate the probability density function of the Generalised Pareto distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the probability density function for.
        Returns
        -------
        float
            The probability density function of the Generalised Pareto distribution.
        """
        if mu is None:
            mu = self.mu
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if sigma is None:
            print('Sigma parameter not set, using parameter stored in the class: sigma=', self.sigma)
            sigma = self.sigma

        # special case for gamma = 0
        if gamma == 0:
            return (1/sigma) * np.exp(-x/sigma)
        
        # Check if x is in the support of the distribution
        if gamma > 0 and x < mu:
            return 0
        if gamma < 0 and (x < mu or x > mu - sigma/gamma):
            return 0
        
        return (1/sigma) * (1 + gamma*(x-mu)/sigma)**(-1/gamma-1)
    
    def log_pdf(
        self,
        x: float,
        gamma: float=None,
        sigma: float=None,
        mu: float=None
    ) -> float:
        """
        Calculate the log probability density function of the Generalised Pareto distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the log probability density function for.
        Returns
        -------
        float
            The log probability density function of the Generalised Pareto distribution.
        """
        if mu is None:
            mu = self.mu
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if sigma is None:
            print('Sigma parameter not set, using parameter stored in the class: sigma=', self.sigma)
            sigma = self.sigma

        # special case for gamma = 0
        if gamma == 0:
            return -np.log(sigma) - x/sigma
        
        # Check if x is in the support of the distribution
        if gamma > 0 and x < mu:
            return -1e10
        if gamma < 0 and (x < mu or x > mu - sigma / gamma):
            return -1e10
        
        return -np.log(sigma) - (1+1/gamma)*np.log(1 + gamma*(x-mu)/sigma)
    
    def cdf(
        self,
        x: float,
        gamma: float=None,
        sigma: float=None,
        mu: float=None
    ) -> float:
        """
        Calculate the cumulative distribution function of the Generalised Pareto distribution.
        Parameters
        ----------
        x : np.ndarray
            The data to calculate the cumulative distribution function for.
        Returns
        -------
        float
            The cumulative distribution function of the Generalised Pareto distribution.
        """
        if mu is None:
            mu = self.mu
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if sigma is None:
            print('Sigma parameter not set, using parameter stored in the class: sigma=', self.sigma)
            sigma = self.sigma

        # special case for gamma = 0
        if gamma == 0:
            return 1 - np.exp(-x/sigma)
        
        return 1 - (1 + gamma*(x-mu)/sigma)**(-1/gamma)

    def inverse_cdf(
        self,
        p: float,
        gamma: float=None,
        sigma: float=None,
        mu: float=None
    ) -> float:
        """
        Calculate the inverse cumulative distribution function of the Generalised Pareto distribution.
        Parameters
        ----------
        p : np.ndarray
            The data to calculate the inverse cumulative distribution function for.
        Returns
        -------
        float
            The inverse cumulative distribution function of the Generalised Pareto distribution.
        """
        if mu is None:
            mu = self.mu
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if sigma is None:
            print('Sigma parameter not set, using parameter stored in the class: sigma=', self.sigma)
            sigma = self.sigma

        if gamma == 0:
            return mu - sigma * np.log(1-p)
        else:
            return mu + sigma * ((1 / (1 - p)**gamma - 1)) / gamma

    def likelihood(
        self,
        data: np.ndarray,
        gamma: float=None,
        sigma: float=None,
        mu: float=None
    ) -> float:
        """
        Calculate the likelihood of the data given the Generalised Pareto distribution.
        The likelihood is calculated as the product of the probability density function evaluated at each data point.
        Parameters
        ----------
        data : np.ndarray
            The data to calculate the likelihood for.
        Returns
        -------
        float
            The likelihood of the data given the Generalised Pareto distribution.
        """
        if mu is None:
            mu = self.mu
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if sigma is None:
            print('Sigma parameter not set, using parameter stored in the class: sigma=', self.sigma)
            sigma = self.sigma

        return np.prod(self.pdf(data, gamma=gamma, sigma=sigma, mu=mu))

    def log_likelihood(
        self,
        data: np.ndarray,
        gamma: float=None,
        sigma: float=None,
        mu: float=None
    ) -> float:
        """
        Calculate the log likelihood of the data given the Generalised Pareto distribution.
        The log likelihood is calculated as the sum of the log of the probability density function evaluated at each data point.
        Parameters
        ----------
        data : np.ndarray
            The data to calculate the log likelihood for.
        Returns
        -------
        float
            The log likelihood of the data given the Generalised Pareto distribution.
        """
        if mu is None:
            mu = self.mu
        
        if gamma is None:
            print('Gamma parameter not set, using parameter stored in the class: gamma=', self.gamma)
            gamma = self.gamma
        
        if sigma is None:
            print('Sigma parameter not set, using parameter stored in the class: sigma=', self.sigma)
            sigma = self.sigma
        
        cum_sum = 0
        for x in data:
            cum_sum += self.log_pdf(x=x, gamma=gamma, sigma=sigma, mu=mu)
        return cum_sum
    
    def fit(
        self,
        data: np.ndarray,
        given_gamma: float=None,
        given_sigma: float=None,
        given_mu: float=0.0,
        set_class_parameters: bool=True
    ) -> (float, float):
        """
        Fit the Generalised Pareto distribution to the data.
        The method of moments is used to estimate the parameters of the distribution.
        Parameters
        ----------
        data : np.ndarray
            The data to fit the distribution to.
        """
        if given_gamma is not None:
            print('Gamma parameter set to:', self.gamma, 'using given_gamma.')
            gamma_bounds = (given_gamma, given_gamma)
            initial_gamma = given_gamma
        else:
            gamma_bounds = (None, None)
            initial_gamma = 0.4
        
        if given_sigma is not None:
            print('Sigma parameter set to:', self.sigma, 'using given_sigma.')
            sigma_bounds = (given_sigma, given_sigma)
        else:
            sigma_bounds = (0, None)
            initial_sigma = 1

        def objective_function(x):
            return -self.log_likelihood(data, gamma=x[0], sigma=x[1], mu=given_mu)

        # Maximise the log likelihood
        result = minimize(
            lambda x: objective_function(x), 
            x0=[initial_gamma, initial_sigma], 
            bounds=[gamma_bounds, sigma_bounds],
            method='L-BFGS-B'
        ).x

        if set_class_parameters:
            self.gamma = result[0]
            self.sigma = result[1]
            self.mu = given_mu
        
        return result[0], result[1], given_mu
    
    def sample(
        self,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample from the Generalised Pareto distribution.
        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the distribution.
        Returns
        -------
        np.ndarray
            The samples drawn from the Generalised Pareto distribution.
        """
        # Do inverse transform sampling
        u = np.random.uniform(0, 1, n_samples)
        return self.inverse_cdf(u, self.gamma, self.sigma, self.mu)

    def mean(
        self
    ) -> float:
        """
        Calculate the mean of the Generalised Pareto distribution.
        Returns
        -------
        float
            The mean of the Generalised Pareto distribution.
        """
        if self.gamma < 0:
            return self.mu + self.sigma / (1 - self.gamma)
        else:
            print('Mean is not defined for gamma >= 1.')
            return None

    def support(
        self
    ) -> (float, float):
        """
        Calculate the support of the Generalised Pareto distribution.
        Returns
        -------
        float
            The lower bound of the Generalised Pareto distribution.
        float
            The upper bound of the Generalised Pareto distribution.
        """
        
        if self.gamma < 0:
            return self.mu, self.mu - self.sigma / self.gamma
        else:
            return self.mu, np.inf
