import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from .abstract_bayesian import AbstractBayesianModel as ABM

class Pareto_One_Stage(ABM):
    def __init__(
        self, 
        y_data: np.ndarray,
        hyperprior_alpha: float=41, # Estimated from the data
        hyperprior_beta: float=44 # Estimated from the data
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_alpha > 0, "hyperprior_alpha must be positive"
        assert hyperprior_beta > 0, "hyperprior_beta must be positive"
        
        self.hyperprior_alpha = hyperprior_alpha
        self.hyperprior_beta = hyperprior_beta

        self.trace = None
        self.y_data = y_data
        self.model = pm.Model()
        with self.model:
            # Priors
            one_over_alpha = pm.Gamma('one_over_alpha', alpha=hyperprior_alpha, beta=hyperprior_beta) # TODO: check that this should not be inverse gamma
            alpha = pm.Deterministic('alpha', 1/one_over_alpha) # Ensuring that "alpha" is the same as in Teulings & Toussaint (2023)
            # Likelihood
            y = pm.Pareto('y', alpha=one_over_alpha, m=1, observed=y_data)

    def fit(
        self, 
        draws: int=1000,
        tune: int=1000,
        chains: int=4,
        cores: int=4,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model """
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                nuts_sampler=nuts_sampler
            )

    def prior_predictive(
        self, 
        samples: int=1000,
        var_names: list[str]=None
    ):
        """ Generate prior predictive samples """
        with self.model:
            pm_prior = pm.sample_prior_predictive(
                samples=samples
            )
        prior = pm_prior['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = pm_prior['prior_predictive'].to_dataframe().reset_index(drop=True)
        merged_prior = pd.concat([prior, prior_pred], axis=1)
        return merged_prior

    def posterior_predictive(
        self, 
        var_names: list[str]=None,
        progressbar: bool=True
    ) -> np.ndarray:
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(
                self.trace,
                progressbar=progressbar
            )
        pm_post = pm_post['posterior_predictive'].to_dataframe().reset_index(drop=True)
        merged_post = pd.concat([pm_trace, pm_post], axis=1)
        return merged_post

    def parameter_estimates(
        self, 
        var_names: list[str]=None
    ):
        """ Extract parameter estimates """
        return az.summary(self.trace, var_names=var_names)

    def y_estimate(
        self
    ) -> float:
        """ Extract the estimate for y """
        return az.summary(self.posterior_predictive())

    def plot_trace(
        self,
        var_names: list[str]=None
    ):
        """ Plot the trace """
        az.plot_trace(self.trace)
        plt.tight_layout()

class Weibull_One_Stage(ABM):
    def __init__(
        self, 
        y_data: np.ndarray,
        hyperprior_gamma_alpha: float=1.7, # Estimated from the data
        hyperprior_gamma_beta: float=5.9, # Estimated from the data
        hyperprior_alpha_alpha: float=1,
        hyperprior_alpha_beta: float=1
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_gamma_alpha > 0, "hyperprior_gamma_alpha must be positive"
        assert hyperprior_gamma_beta > 0, "hyperprior_gamma_beta must be positive"
        assert hyperprior_alpha_alpha > 0, "hyperprior_alpha_alpha must be positive"
        assert hyperprior_alpha_beta > 0, "hyperprior_alpha_beta must be positive"
        
        self.hyperprior_gamma_alpha = hyperprior_gamma_alpha
        self.hyperprior_gamma_beta = hyperprior_gamma_beta
        self.hyperprior_alpha_alpha = hyperprior_alpha_alpha
        self.hyperprior_alpha_beta = hyperprior_alpha_beta

        self.trace = None
        self.y_data = y_data
        self.model = pm.Model()
        with self.model:
            # Priors
            alpha_pymc = pm.InverseGamma('alpha_pymc', alpha=hyperprior_gamma_alpha, beta=hyperprior_gamma_beta)
            beta_pymc = pm.Gamma('beta_pymc', alpha=hyperprior_alpha_alpha, beta=hyperprior_alpha_beta)
            # Transformations to notation from Teulings & Toussaint (2023)
            gamma = pm.Deterministic('gamma', alpha_pymc)
            alpha = pm.Deterministic('alpha', beta_pymc ** alpha_pymc / alpha_pymc)
            
            # Likelihood
            y_non_truncated = pm.Weibull.dist(alpha=alpha_pymc, beta=beta_pymc)
            y = pm.Truncated('y', y_non_truncated, lower=1, upper=None, observed=self.y_data)

    def fit(
        self, 
        draws: int=2000,
        tune: int=2000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model """
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler
            )

    def prior_predictive(
        self, 
        samples: int=1000,
        var_names: list[str]=None
    ):
        """ Generate prior predictive samples """
        with self.model:
            pm_prior = pm.sample_prior_predictive(
                samples=samples
            )
        prior = pm_prior['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = pm_prior['prior_predictive'].to_dataframe().reset_index(drop=True)
        merged_prior = pd.concat([prior, prior_pred], axis=1)
        return merged_prior

    def posterior_predictive(
        self, 
        var_names: list[str]=None,
        progressbar: bool=True
    ) -> np.ndarray:
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(
                self.trace,
                progressbar=progressbar
            )
        pm_post = pm_post['posterior_predictive'].to_dataframe().reset_index(drop=True)
        merged_post = pd.concat([pm_trace, pm_post], axis=1)
        return merged_post

    def parameter_estimates(
        self, 
        var_names: list[str]=None
    ):
        """ Extract parameter estimates """
        return az.summary(self.trace, var_names=var_names)

    def y_estimate(
        self
    ) -> float:
        """ Extract the estimate for y """
        return az.summary(self.posterior_predictive())

class GeneralisedPareto_One_Stage(ABM):
    def __init__(
        self, 
        y_data: np.ndarray,
        hyperprior_gamma_alpha: float=33, # Estimated from the data
        hyperprior_gamma_beta: float=30, # Estimated from the data
        hyperprior_sigma_alpha: float=1,
        hyperprior_sigma_beta: float=1
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_gamma_alpha > 0, "hyperprior_gamma_alpha must be positive"
        assert hyperprior_gamma_beta > 0, "hyperprior_gamma_beta must be positive"
        assert hyperprior_sigma_alpha > 0, "hyperprior_sigma_alpha must be positive"
        assert hyperprior_sigma_beta > 0, "hyperprior_sigma_beta must be positive"
        
        self.hyperprior_gamma_alpha = hyperprior_gamma_alpha
        self.hyperprior_gamma_beta = hyperprior_gamma_beta
        self.hyperprior_sigma_alpha = hyperprior_sigma_alpha
        self.hyperprior_sigma_beta = hyperprior_sigma_beta

        self.trace = None
        self.y_data = y_data
        self.model = pm.Model()
        with self.model:
            # Priors
            mu = 1 # threshold fixed at 1
            gamma_false_loc = pm.Gamma('gamma_false_loc', alpha=hyperprior_gamma_alpha, beta=hyperprior_gamma_beta)
            gamma = pm.Deterministic('gamma', gamma_false_loc - 0.5) # From Example 2.8, Dombry, Padoan and Rizelli (2023)
            sigma = pm.Gamma('sigma', alpha=hyperprior_sigma_alpha, beta=hyperprior_sigma_beta)

            # Functions for the custom distribution
            def gpd_logp(value, mu, gamma, sigma):
                if gamma == 0:
                    return -np.log(sigma) - (value - mu) / sigma
                else:
                    return -np.log(sigma) - (1 + 1 / gamma) * np.log(1 + gamma * (value - mu) / sigma)

            def gpd_random(mu, gamma, sigma, rng=None, size=None):
                # generate uniforms
                u = rng.uniform(size=size)
                if gamma == 0:
                    return mu - sigma * np.log(1-u)
                else:
                    return mu + sigma * ((1 / (1 - u)**gamma - 1)) / gamma

            # Likelihood
            y = pm.CustomDist('y', mu, gamma, sigma, logp=gpd_logp, random=gpd_random, observed=self.y_data)

    def fit(
        self, 
        draws: int=2000,
        tune: int=2000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model """
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler
            )

    def prior_predictive(
        self, 
        samples: int=1000,
        var_names: list[str]=None
    ):
        """ Generate prior predictive samples """
        with self.model:
            pm_prior = pm.sample_prior_predictive(
                samples=samples
            )
        prior = pm_prior['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = pm_prior['prior_predictive'].to_dataframe().reset_index(drop=True)
        merged_prior = pd.concat([prior, prior_pred], axis=1)
        return merged_prior

    def posterior_predictive(
        self, 
        var_names: list[str]=None,
        progressbar: bool=True
    ) -> np.ndarray:
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(
                self.trace,
                progressbar=progressbar
            )
        pm_post = pm_post['posterior_predictive'].to_dataframe().reset_index(drop=True)
        merged_post = pd.concat([pm_trace, pm_post], axis=1)
        return merged_post

    def parameter_estimates(
        self, 
        var_names: list[str]=None
    ):
        """ Extract parameter estimates """
        return az.summary(self.trace, var_names=var_names)
    
    def y_estimate(
        self
    ) -> float:
        """ Extract the estimate for y """
        return az.summary(self.posterior_predictive())

class Pareto_Two_Stage(ABM):
    def __init__(
        self, 
        y_data: np.ndarray,
        hyperprior_half_normal_sigma: float=40 # Very vague prior 
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_half_normal_sigma > 0, "hyperprior_half_normal_sigma must be positive"
        
        self.hyperprior_half_normal_sigma = hyperprior_half_normal_sigma

        self.trace = None
        self.y_data = y_data
        self.model = pm.Model()
        with self.model:
            # Priors
            hyperprior_alpha = pm.HalfNormal('hyperprior_alpha', sigma=hyperprior_half_normal_sigma)
            hyperprior_beta = pm.HalfNormal('hyperprior_beta', sigma=hyperprior_half_normal_sigma)
            one_over_alpha = pm.Gamma('one_over_alpha', alpha=hyperprior_alpha, beta=hyperprior_beta) # TODO: check that this should not be inverse gamma
            alpha = pm.Deterministic('alpha', 1/one_over_alpha) # Ensuring that "alpha" is the same as in Teulings & Toussaint (2023)
            # Likelihood
            y = pm.Pareto('y', alpha=one_over_alpha, m=1, observed=y_data)

    def fit(
        self, 
        draws: int=1000,
        tune: int=1000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model """
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler
            )
    
    def prior_predictive(
        self, 
        samples: int=1000,
        var_names: list[str]=None
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior = pm.sample_prior_predictive(
                samples=samples
            )
        return prior

    def posterior_predictive(
        self, 
        var_names: list[str]=None,
        progressbar: bool=True
    ) -> np.ndarray:
        """ Generate posterior predictive samples """
        with self.model:
            posterior = pm.sample_posterior_predictive(
                self.trace,
                progressbar=progressbar
            )
        # Flatten values
        return posterior.posterior_predictive['y'].values.flatten()

    def parameter_estimates(
        self, 
        var_names: list[str]=None
    ):
        """ Extract parameter estimates """
        return az.summary(self.trace, var_names=var_names)

    def y_estimate(
        self
    ) -> float:
        """ Extract the estimate for y """
        return az.summary(self.posterior_predictive())

class Weibull_Two_Stage(ABM):
    def __init__(
        self, 
        y_data: np.ndarray,
        hyperprior_half_normal_sigma: float=40 # Very vague prior
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_half_normal_sigma > 0, "hyperprior_half_normal_sigma must be positive"

        self.hyperprior_half_normal_sigma = hyperprior_half_normal_sigma

        self.trace = None
        self.y_data = y_data
        self.model = pm.Model()

        with self.model:
            # Priors
            hyperprior_alpha_alpha = pm.HalfNormal('hyperprior_alpha_alpha', sigma=hyperprior_half_normal_sigma)
            hyperprior_alpha_beta = pm.HalfNormal('hyperprior_alpha_beta', sigma=hyperprior_half_normal_sigma)
            hyperprior_gamma_alpha = pm.HalfNormal('hyperprior_gamma_alpha', sigma=hyperprior_half_normal_sigma)
            hyperprior_gamma_beta = pm.HalfNormal('hyperprior_gamma_beta', sigma=hyperprior_half_normal_sigma)
            alpha_pymc = pm.InverseGamma('alpha_pymc', alpha=hyperprior_alpha_alpha, beta=hyperprior_alpha_beta)
            beta_pymc = pm.Gamma('beta_pymc', alpha=hyperprior_gamma_alpha, beta=hyperprior_gamma_beta)
            # Transformations to notation from Teulings & Toussaint (2023)
            gamma = pm.Deterministic('gamma', alpha_pymc)
            alpha = pm.Deterministic('alpha', beta_pymc ** alpha_pymc / alpha_pymc)
            
            # Likelihood
            y_non_truncated = pm.Weibull.dist(alpha=alpha_pymc, beta=beta_pymc)
            y = pm.Truncated('y', y_non_truncated, lower=1, upper=None, observed=self.y_data)

    def fit(
        self, 
        draws: int=2000,
        tune: int=2000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model """
        with self.model:
            # Sample
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                nuts_sampler=nuts_sampler
            )
        return self.trace
    
    def prior_predictive(
        self, 
        samples: int=1000,
        var_names: list[str]=None
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior = pm.sample_prior_predictive(
                samples=samples
            )
        return prior
    
    def posterior_predictive(
        self, 
        var_names: list[str]=None,
        progressbar: bool=True
    ) -> np.ndarray:
        """ Generate posterior predictive samples """
        with self.model:
            posterior = pm.sample_posterior_predictive(
                self.trace,
                progressbar=progressbar
            )
        # Flatten values
        return posterior.posterior_predictive['y'].values.flatten()

    def parameter_estimates(
        self, 
        var_names: list[str]=None
    ):
        """ Extract parameter estimates """
        return az.summary(self.trace, var_names=var_names)

    def y_estimate(
        self
    ) -> float:
        """ Extract the estimate for y """
        return az.summary(self.posterior_predictive())
    
class GeneralisedPareto_Two_Stage(ABM):
    def __init__(
        self, 
        y_data: np.ndarray,
        hyperprior_half_normal_sigma: float=40 # Very vague prior
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_half_normal_sigma > 0, "hyperprior_half_normal_sigma must be positive"

        self.hyperprior_half_normal_sigma = hyperprior_half_normal_sigma

        self.trace = None
        self.y_data = y_data
        self.model = pm.Model()
        with self.model:
            # Priors
            mu = 1 # threshold fixed at 1
            hyperprior_gamma_alpha = pm.HalfNormal('hyperprior_gamma_alpha', sigma=hyperprior_half_normal_sigma)
            hyperprior_gamma_beta = pm.HalfNormal('hyperprior_gamma_beta', sigma=hyperprior_half_normal_sigma)
            hyperprior_sigma_alpha = pm.HalfNormal('hyperprior_sigma_alpha', sigma=hyperprior_half_normal_sigma)
            hyperprior_sigma_beta = pm.HalfNormal('hyperprior_sigma_beta', sigma=hyperprior_half_normal_sigma)
            gamma_false_loc = pm.Gamma('gamma_false_loc', alpha=hyperprior_gamma_alpha, beta=hyperprior_gamma_beta)
            gamma = pm.Deterministic('gamma', gamma_false_loc - 0.5) # From Example 2.8, Dombry, Padoan and Rizelli (2023)
            sigma = pm.Gamma('sigma', alpha=hyperprior_sigma_alpha, beta=hyperprior_sigma_beta)

            # Functions for the custom distribution
            def gpd_logp(value, mu, gamma, sigma):
                if gamma == 0:
                    return -np.log(sigma) - (value - mu) / sigma
                else:
                    return -np.log(sigma) - (1 + 1 / gamma) * np.log(1 + gamma * (value - mu) / sigma)

            def gpd_random(mu, gamma, sigma, rng=None, size=None):
                # generate uniforms
                u = rng.uniform(size=size)
                if gamma == 0:
                    return mu - sigma * np.log(1-u)
                else:
                    return mu + sigma * ((1 / (1 - u)**gamma - 1)) / gamma

            # Likelihood
            y = pm.CustomDist('y', mu, gamma, sigma, logp=gpd_logp, random=gpd_random, observed=self.y_data)

    def fit(
        self, 
        draws: int=1000,
        tune: int=1000,
        chains: int=4,
        cores: int=4,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model """
        with self.model:
            # Sample
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                nuts_sampler=nuts_sampler
            )
        return self.trace

    def prior_predictive(
        self, 
        samples: int=1000,
        var_names: list[str]=None
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior = pm.sample_prior_predictive(
                samples=samples
            )
        return prior

    def posterior_predictive(
        self, 
        var_names: list[str]=None,
        progressbar: bool=True
    ) -> np.ndarray:
        """ Generate posterior predictive samples """
        with self.model:
            posterior = pm.sample_posterior_predictive(
                self.trace,
                progressbar=progressbar
            )
        # Flatten values
        return posterior.posterior_predictive['y'].values.flatten()

    def parameter_estimates(
        self, 
        var_names: list[str]=None
    ):
        """ Extract parameter estimates """
        return az.summary(self.trace, var_names=var_names)
