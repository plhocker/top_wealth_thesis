import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import copy

from .abstract_bayesian import AbstractBayesianModel as ABM
from .frequentist import * # To help sampling

class Multivariate_Pareto(ABM):
    def __init__(
        self, 
        data: pd.DataFrame,
        dependent_variable: str='net_worth',
        group_variable: str='sub_region',
        hyperprior_alpha: float=41,
        hyperprior_beta: float=44
    ):
        """ Multivariate Pareto model for cross-sectional data. """

        # Assert that hyperprior_alpha and hyperprior_beta are positive
        assert hyperprior_alpha > 0, "hyperprior_alpha must be positive."
        assert hyperprior_beta > 0, "hyperprior_beta must be positive."

        # Set-up the data
        df = copy.deepcopy(data)
        df = df[[dependent_variable, group_variable]]
        df.columns = ['dependent', 'group']
        df['group_TE'] = df['group'].astype('category').cat.codes
        self.df = df

        # Set-up the groups
        groups = df['group_TE'].values
        n_groups = df['group_TE'].nunique()
        y_obs = df['dependent'].values
        self.TE_to_group = dict(zip(df['group_TE'], df['group']))

        # Set-up the model
        model = pm.Model()
        with model:
            one_over_alpha_group = pm.Gamma(
                'one_over_alpha_group', 
                alpha=hyperprior_alpha, 
                beta=hyperprior_beta, 
                shape=n_groups
            )

            alpha_group = pm.Deterministic('alpha_group', 1/one_over_alpha_group)
            
            alpha_obs = one_over_alpha_group[groups]

            y = pm.Pareto('y', alpha=alpha_obs, m=1, observed=y_obs)

        self.model = model
        self.trace = None

    def fit(
        self, 
        draws: int=1000, 
        tune: int=1000, 
        chains: int=4, 
        cores: int=4,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model. """
        with self.model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                nuts_sampler=nuts_sampler
            )
        self.trace = trace

    def prior_predictive(
        self,
        n_dependent_samples: int=1000
    ) -> dict:
        with self.model:
            prior_trace = pm.sample_prior_predictive()

        alpha_samples = prior_trace.prior['alpha_group'].to_dataframe()
        alpha_samples = alpha_samples.groupby('alpha_group_dim_0')['alpha_group'].apply(list).to_dict()

        # reindex the dictionary to group names
        alpha_samples = {self.TE_to_group[k]: v for k, v in alpha_samples.items()}

        # sample from the prior predictive for the dependent variable
        y_pred = {}
        for group in alpha_samples.keys():
            group_samples = []
            for i in range(n_dependent_samples):
                # take a random element from the alpha samples
                alpha = np.random.choice(alpha_samples[group])
                group_samples.append(Pareto(alpha=alpha, x_m=1).sample(1)[0])
            y_pred[group] = group_samples
        
        return_dict = {}
        return_dict['alpha'] = alpha_samples
        return_dict['y'] = y_pred

        return return_dict

    def posterior_predictive(
        self,
        n_dependent_samples: int=1000
    ) -> dict:
        alpha_samples = self.trace.posterior['alpha_group'].to_dataframe()
        alpha_samples = alpha_samples.groupby('alpha_group_dim_0')['alpha_group'].apply(list).to_dict()

        # reindex the dictionary to group names
        alpha_samples = {self.TE_to_group[k]: v for k, v in alpha_samples.items()}

        # sample from the posterior predictive for the dependent variable
        y_pred = {}
        for group in alpha_samples.keys():
            group_samples = []
            for i in range(n_dependent_samples):
                # take a random element from the alpha samples
                alpha = np.random.choice(alpha_samples[group])
                group_samples.append(Pareto(alpha=alpha, x_m=1).sample(1)[0])
            y_pred[group] = group_samples

        return_dict = {}
        return_dict['alpha'] = alpha_samples
        return_dict['y'] = y_pred

        return return_dict

    def get_trace(self):
        return self.trace

    def get_prior_summary(self):
        with self.model:
            prior = pm.sample_prior_predictive()
        
        df = az.summary(prior, var_names=['alpha_group'], round_to=2)

        # Extract the prefix and index from each row
        prefix_index_tuples = [(idx.split('[')[0], int(idx.split('[')[1][:-1])) for idx in df.index]

        # Create a new DataFrame with the original prefix and country name as index
        df['prefix'] = [prefix for prefix, _ in prefix_index_tuples]
        df['group'] = [self.TE_to_group[i] for _, i in prefix_index_tuples]

        # Group by the original prefix
        df.set_index('group', inplace=True, drop=True)
        grouped_df = df.groupby('prefix')

        return grouped_df

    def get_posterior_summary(self):
        df = az.summary(self.trace, var_names=['alpha_group'], round_to=2)

        # Extract the prefix and index from each row
        prefix_index_tuples = [(idx.split('[')[0], int(idx.split('[')[1][:-1])) for idx in df.index]

        # Create a new DataFrame with the original prefix and country name as index
        df['prefix'] = [prefix for prefix, _ in prefix_index_tuples]
        df['group'] = [self.TE_to_group[i] for _, i in prefix_index_tuples]

        # Group by the original prefix
        df.set_index('group', inplace=True, drop=True)
        grouped_df = df.groupby('prefix')

        return grouped_df

class Multivariate_Weibull(ABM):
    def __init__(
        self, 
        data: pd.DataFrame,
        dependent_variable: str='net_worth',
        group_variable: str='sub_region',
        hyperprior_gamma_alpha: float=1.7, # estimated from the data
        hyperprior_gamma_beta: float=5.9, # estimated from the data
        hyperprior_alpha_alpha: float=12.4, # estimated from the data
        hyperprior_alpha_beta: float=14.2 # estimated from the data
    ):
        """ Multivariate Weibull model for cross-sectional data. """

        # Assert that hyperprior_alpha and hyperprior_beta are positive
        assert hyperprior_gamma_alpha > 0, "hyperprior_gamma_alpha must be positive."
        assert hyperprior_gamma_beta > 0, "hyperprior_gamma_beta must be positive."
        assert hyperprior_alpha_alpha > 0, "hyperprior_alpha_alpha must be positive."
        assert hyperprior_alpha_beta > 0, "hyperprior_alpha_beta must be positive."

        # Set-up the data
        df = copy.deepcopy(data)
        df = df[[dependent_variable, group_variable]]
        df.columns = ['dependent', 'group']
        df['group_TE'] = df['group'].astype('category').cat.codes
        self.df = df

        # Set-up the groups
        groups = df['group_TE'].values
        n_groups = df['group_TE'].nunique()
        y_obs = df['dependent'].values
        self.TE_to_group = dict(zip(df['group_TE'], df['group']))

        # Set-up the model
        model = pm.Model()
        with model: # TODO: CHECK THIS MODEL SPECIFICATION, ESTIMATED PARAMETERS DO NOT MATCH MLE
            alpha_pymc = pm.Gamma(
                'alpha_pymc', 
                alpha=hyperprior_gamma_alpha, 
                beta=hyperprior_gamma_beta, 
                shape=n_groups
            )

            beta_pymc = pm.InverseGamma(
                'beta_pymc', 
                alpha=hyperprior_alpha_alpha, 
                beta=hyperprior_alpha_beta, 
                shape=n_groups
            )
            
            gamma = pm.Deterministic('gamma_obs', alpha_pymc) # Transform to notation from Teulings & Toussaint (2023)
            alpha = pm.Deterministic('alpha_obs', beta_pymc ** alpha_pymc / alpha_pymc)

            alpha_pymc_obs = alpha_pymc[groups]
            beta_pymc_obs = beta_pymc[groups]
            gamma_obs = gamma[groups]
            alpha_obs = alpha[groups]
            
            y_non_truncated = pm.Weibull.dist(alpha=alpha_pymc_obs, beta=beta_pymc_obs)
            y = pm.Truncated('y', y_non_truncated, lower=1, upper=None, observed=y_obs)

        self.model = model
        self.trace = None
    
    def fit(
        self, 
        draws: int=1000, 
        tune: int=1000, 
        chains: int=4, 
        cores: int=4,
        nuts_sampler: str='nutpie'
    ):
        """ Fit the model. """
        with self.model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                nuts_sampler=nuts_sampler
            )
        self.trace = trace

    def prior_predictive(
        self,
        n_dependent_samples: int=1000
    ) -> dict:
        with self.model:
            prior_trace = pm.sample_prior_predictive()

        gamma_samples = prior_trace.prior['gamma_obs'].to_dataframe()
        gamma_samples = gamma_samples.groupby('gamma_obs_dim_0')['gamma_obs'].apply(list).to_dict()

        alpha_samples = prior_trace.prior['alpha_obs'].to_dataframe()
        alpha_samples = alpha_samples.groupby('alpha_obs_dim_0')['alpha_obs'].apply(list).to_dict()
        self.gamma_samples = gamma_samples # TODO: remove this
        # reindex the dictionary to group names
        gamma_samples = {self.TE_to_group[k]: v for k, v in gamma_samples.items()}
        alpha_samples = {self.TE_to_group[k]: v for k, v in alpha_samples.items()}

        # sample from the prior predictive for the dependent variable
        y_pred = {}
        for group in alpha_samples.keys():
            group_samples = []
            for i in range(n_dependent_samples):
                # take a random element from the alpha samples
                gamma = np.random.choice(gamma_samples[group])
                alpha = np.random.choice(alpha_samples[group])
                group_samples.append(Weibull(gamma=gamma, alpha=alpha).sample(1)[0])
            y_pred[group] = group_samples
        
        return_dict = {}
        return_dict['gamma'] = gamma_samples
        return_dict['alpha'] = alpha_samples
        return_dict['y'] = y_pred

        return return_dict

    def posterior_predictive(
        self,
        n_dependent_samples: int=1000
    ) -> dict:
        gamma_samples = self.trace.posterior['gamma_obs'].to_dataframe()
        gamma_samples = gamma_samples.groupby('gamma_obs_dim_0')['gamma_obs'].apply(list).to_dict()

        alpha_samples = self.trace.posterior['alpha_obs'].to_dataframe()
        alpha_samples = alpha_samples.groupby('alpha_obs_dim_0')['alpha_obs'].apply(list).to_dict()

        # reindex the dictionary to group names
        gamma_samples = {self.TE_to_group[k]: v for k, v in gamma_samples.items()}
        alpha_samples = {self.TE_to_group[k]: v for k, v in alpha_samples.items()}

        # sample from the posterior predictive for the dependent variable
        y_pred = {}
        for group in alpha_samples.keys():
            group_samples = []
            for i in range(n_dependent_samples):
                # take a random element from the alpha samples
                gamma = np.random.choice(gamma_samples[group])
                alpha = np.random.choice(alpha_samples[group])
                group_samples.append(Weibull(gamma=gamma, alpha=alpha).sample(1)[0])
            y_pred[group] = group_samples

        return_dict = {}
        return_dict['gamma'] = gamma_samples
        return_dict['alpha'] = alpha_samples
        return_dict['y'] = y_pred

        return return_dict

    def get_trace(self):
        return self.trace

    def get_prior_summary(self):
        with self.model:
            prior = pm.sample_prior_predictive()
        
        df = az.summary(prior, var_names=['gamma_obs', 'alpha_obs'], round_to=2)

        # Extract the prefix and index from each row
        prefix_index_tuples = [(idx.split('[')[0], int(idx.split('[')[1][:-1])) for idx in df.index]

        # Create a new DataFrame with the original prefix and country name as index
        df['prefix'] = [prefix for prefix, _ in prefix_index_tuples]
        df['group'] = [self.TE_to_group[i] for _, i in prefix_index_tuples]

        # Group by the original prefix
        df.set_index('group', inplace=True, drop=True)
        grouped_df = df.groupby('prefix')

        return grouped_df

    def get_posterior_summary(self):
        df = az.summary(self.trace, var_names=['gamma_obs', 'alpha_obs'], round_to=2)

        # Extract the prefix and index from each row
        prefix_index_tuples = [(idx.split('[')[0], int(idx.split('[')[1][:-1])) for idx in df.index]

        # Create a new DataFrame with the original prefix and country name as index
        df['prefix'] = [prefix for prefix, _ in prefix_index_tuples]
        df['group'] = [self.TE_to_group[i] for _, i in prefix_index_tuples]

        # Group by the original prefix
        df.set_index('group', inplace=True, drop=True)
        grouped_df = df.groupby('prefix')

        return grouped_df
    