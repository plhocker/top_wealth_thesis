import numpy as np
import pandas as pd
import copy
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

from .abstract_bayesian import AbstractBayesianModel as ABM

class Multivariate_GeneralisedPareto_TimeSeries(ABM):
    def __init__(
        self,
        panel_data: pd.DataFrame,
        group_variable: str='group',
        dependent_variable: str='net_worth',
        independent_variables: list=['log_change_gdp_pc', 'log_change_MSCI', 'log_change_SPX'],
        time_variable: str='year', # TODO: check if this is needed
        hyperprior_gamma_alpha: float=33, # estimated from the data
        hyperprior_gamma_beta: float=30, # estimated from the data
        hyperprior_sigma_alpha: float=8, # estimated from the data
        hyperprior_sigma_beta: float=8 # estimated from the data
    ):
        """ Multivariate Generalised Pareto model for time-series data with covariates. """
        
        # Assert that the hyperpriors are positive
        assert hyperprior_gamma_alpha > 0, "Hyperprior alpha for gamma distribution must be positive."
        assert hyperprior_gamma_beta > 0, "Hyperprior beta for gamma distribution must be positive."
        assert hyperprior_sigma_alpha > 0, "Hyperprior alpha for sigma distribution must be positive."
        assert hyperprior_sigma_beta > 0, "Hyperprior beta for sigma distribution must be positive."

        # Assert that the group variable is in the data
        assert group_variable in panel_data.columns, "Group variable not in the data."

        # Assert that the dependent variable is in the data
        assert dependent_variable in panel_data.columns, "Dependent variable not in the data."

        # Assert that the independent variables are in the data
        for var in independent_variables:
            assert var in panel_data.columns, f"{var} not in the data."

        # Set up the data
        df = copy.deepcopy(panel_data)

        # Set up the model
        generalised_pareto_model = pm.Model()

        with generalised_pareto_model:
            # Define initial sigma for each group
            initial_sigma = {}
            for group in df[group_variable].unique():
                initial_sigma[group] = pm.Gamma(
                    f"initial_sigma_{group}",
                    alpha=hyperprior_sigma_alpha,
                    beta=hyperprior_sigma_beta
                )

            # Define betas for each covariate and each group
            n_groups = len(df[group_variable].unique())
            n_covariates = len(independent_variables)

            betas = {}
            for cov in independent_variables:
                beta_variances = pm.Gamma(
                    f"beta_variance_{cov}",
                    alpha=1,
                    beta=1,
                    shape=n_groups
                )
                beta_cov = pt.diag(beta_variances)

                beta_mu = np.zeros(n_groups)
                betas[cov] = pm.MvNormal(
                    f"beta_{cov}",
                    mu=beta_mu,
                    tau=beta_cov,
                    shape=(n_groups)
                )
            
            # Prepare betas for each group
            group_betas = {group: [] for group in df[group_variable].unique()}
            for i, group in enumerate(df[group_variable].unique()):
                for cov in independent_variables:
                    group_betas[group].append(betas[cov][i])
                group_betas[group] = pt.stack(group_betas[group])

            # Define sigma for each time step and group
            sigma = {}
            for group in df[group_variable].unique():
                sigma[group] = {}
                min_T = df[df[group_variable] == group][time_variable].min()
                sigma[group][min_T] = initial_sigma[group]

            for group in df[group_variable].unique():
                group_df = df[df[group_variable] == group].reset_index()
                T = group_df[time_variable].unique()
                X = {}
                for t in T:
                    X[t] = group_df[group_df[time_variable] == t][independent_variables].values

                for t in T[1:]:
                    sigma_epsilon = pm.Gamma(
                        f"sigma_epsilon_{group}_{t}",
                        alpha=hyperprior_sigma_alpha,
                        beta=hyperprior_sigma_beta
                    )
                    epsilon = pm.Normal(
                        f"epsilon_{group}_{t}",
                        mu=0,
                        sigma=sigma_epsilon
                    )
                    sigma[group][t] = (
                        sigma[group][t-1] * pm.math.exp(
                            pm.math.dot(X[t], group_betas[group])
                        ) + epsilon
                    )

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
            
            # Gamma distribution for the shape parameter
            # gammas = {}
            # for group in df['group'].unique():
            #     gammas[group] = pm.Gamma(
            #         f'gamma_{group}', 
            #         hyperprior_gamma_alpha, 
            #         hyperprior_gamma_beta
            #     )
            # common_gamma = 0.5
            common_gamma = pm.Gamma('gamma', 1, 1)
            
            # Threshold is fixed at 1
            mu=1.0

            y_obs = []
            for group in df['group'].unique():
                # group_df = df[df['group'] == group].reset_index()
                group_df = df[df['group'] == group].set_index(time_variable, drop=False)
                T = group_df['year'].unique()
                for t in T:
                    y_obs.append(pm.CustomDist(
                        f'y_{group}_{t}',
                        mu,
                        common_gamma,
                        # gammas[group],
                        sigma[group][t],
                        logp=gpd_logp, 
                        random=gpd_random,
                        observed=group_df.loc[t, dependent_variable],
                    ))

        self.model = generalised_pareto_model
        self.data = None

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

    def prior_predictive():
        """ Generate prior predictive samples. """
        pass

    def posterior_predictive():
        """ Generate posterior predictive samples. """
        pass
