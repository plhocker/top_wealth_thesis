import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from .abstract_bayesian import AbstractBayesianModel as ABM
from .frequentist import Pareto, Weibull, GeneralisedPareto

class Univariate_Pareto_TimeSeries(ABM):
    def __init__(
        self, 
        panel_df: pd.DataFrame,
        train_until: int=None,
        y_column: str='net_worth',
        X_columns: list[str]=['constant', 'log_change_gdp_pc', 'log_change_MSCI'], 
        hyperprior_alpha: float=41, # Estimated from the data
        hyperprior_beta: float=44 # Estimated from the data
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_alpha > 0, "hyperprior_alpha must be positive"
        assert hyperprior_beta > 0, "hyperprior_beta must be positive"
        
        self.hyperprior_alpha = hyperprior_alpha
        self.hyperprior_beta = hyperprior_beta

        self.y_column = y_column
        self.X_columns = X_columns

        self.trace = None
        self.df = panel_df.set_index('year')
        if train_until is not None:
            self.df_train = self.df.loc[:train_until]
            self.df_test = self.df.loc[train_until+1:]
        else:
            self.df_train = self.df
            self.df_test = None

        # Check how long the panel is
        MAX_LENGTH = 15
        if len(self.df_train) > MAX_LENGTH:
            print(f"Panel is too long, truncating to {MAX_LENGTH} years")
            self.df_train = self.df_train.tail(MAX_LENGTH)

        # Check that there are at least len(X_columns)+2 observations
        assert len(self.df_train) >= len(X_columns)+2, "Not enough observations in the panel"

        self.model = pm.Model()
        with self.model:    
            betas = []
            for cov in X_columns:
                beta_variance = pm.Gamma(f'beta_variance_{cov}', alpha=1, beta=1)
                betas.append(pm.Normal(f'beta_{cov}', mu=0, sigma=beta_variance))

            years = self.df_train.index.values
            X = self.df_train[X_columns]

            initial_one_over_alpha = pm.Gamma(
                f'one_over_alpha_{years[0]}', 
                alpha=self.hyperprior_alpha, 
                beta=self.hyperprior_beta
            ) 
            initial_alpha = pm.Deterministic(f'alpha_{years[0]}', 1/initial_one_over_alpha)
            one_over_alphas = {years[0]: initial_one_over_alpha}
            alphas = {years[0]: initial_alpha}

            epsilon_alpha = pm.Gamma('epsilon_alpha', alpha=1, beta=1)
            for year in years[1:]:
                epsilon = pm.Normal(f'epsilon_{year}', mu=0, sigma=epsilon_alpha)
                one_over_alpha = pm.Deterministic(
                    f'one_over_alpha_{year}', 
                    one_over_alphas[year-1] / pm.math.exp(pm.math.dot(X.loc[year], betas) + epsilon)
                )
                alpha = pm.Deterministic(f'alpha_{year}', 1 / one_over_alpha)
                one_over_alphas[year] = one_over_alpha
                alphas[year] = alpha

            y_obs = []
            for year in years:
                y_obs.append(
                    pm.Pareto(
                        f'y_{year}', 
                        alpha=one_over_alphas[year], 
                        m = 1.0,
                        observed=self.df_train.loc[year][y_column])
                )

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
                nuts_sampler=nuts_sampler
            )

    def get_trace(self):
        """ Return the trace """
        return self.trace

    def prior_predictive(
        self,
        samples: int=1000
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior_pm = pm.sample_prior_predictive(samples=samples)
        prior = prior_pm['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = prior_pm['prior_predictive']
        prior_df = pd.DataFrame()
        for var in list(prior_pred.data_vars):
            vals = prior_pred[var].values.flatten()
            # select 10000 random samples from the prior
            prior_df[var] = np.random.choice(vals, 10000)
        merged_prior = pd.concat([prior, prior_df], axis=1)
        return merged_prior

    def posterior_predictive(
        self
    ):
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(self.trace)
        post_pred = pm_post['posterior_predictive']
        post_df = pd.DataFrame()
        for var in list(post_pred.data_vars):
            vals = post_pred[var].values.flatten()
            # select 10000 random samples from the posterior
            post_df[var] = np.random.choice(vals, 10000)
        merged_post = pd.concat([pm_trace, post_df], axis=1)
        return merged_post
        
    def predict(
        self
    ):
        def predict_row(row, df_test, X_columns, year):
            row[f'epsilon_{year}'] = np.random.normal(loc=0, scale=row['epsilon_alpha'])
            cum_sum = 0
            for X_column in X_columns:
                cum_sum += row[f'beta_{X_column}'] * df_test.loc[year, X_column]
            cum_sum += row[f'epsilon_{year}']
            row[f'alpha_{year}'] = row[f'alpha_{year - 1}'] * np.exp(cum_sum)
            row[f'y_{year}'] = Pareto(alpha=row[f'alpha_{year}'], x_m=1).sample(1)[0]
            return row

        trace_df = self.trace['posterior'].to_dataframe().reset_index(drop=True)

        # Apply this to the trace_df
        for year in self.df_test.index:
            trace_df = trace_df.apply(
                lambda row: predict_row(
                    row=row, 
                    df_test=self.df_test, 
                    X_columns=self.X_columns,
                    year=year
                ),
            axis=1)

        return trace_df

        
class Univariate_Weibull_TimeSeries(ABM):
    def __init__(
        self,
        panel_df: pd.DataFrame,
        train_until: int=None,
        y_column: str='net_worth',
        X_columns: list[str]=['constant', 'log_change_gdp_pc', 'log_change_MSCI'],
        hyperprior_gamma_alpha: float=1.7, # Estimated from the data
        hyperprior_gamma_beta: float=5.9, # Estimated from the data
        hyperprior_alpha_alpha: float=1.0,
        hyperprior_alpha_beta: float=1.0
    ):
        """ Initialize the model """
        self.y_column = y_column
        self.X_columns = X_columns

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
        self.df = panel_df.set_index('year')
        if train_until is not None:
            self.df_train = self.df.loc[:train_until]
            self.df_test = self.df.loc[train_until+1:]
        else:
            self.df_train = self.df
            self.df_test = None

        # Check how long the panel is
        MAX_LENGTH = 15
        if len(self.df_train) > MAX_LENGTH:
            print(f"Panel is too long, truncating to {MAX_LENGTH} years")
            self.df_train = self.df_train.tail(MAX_LENGTH)

        # Check that there are at least len(X_columns)+2 observations
        assert len(self.df_train) >= len(X_columns)+2, "Not enough observations in the panel"

        self.model = pm.Model()

        with self.model:
            betas = []
            for cov in X_columns:
                beta_variance = pm.Gamma(f'beta_variance_{cov}', alpha=1, beta=1)
                betas.append(pm.Normal(f'beta_{cov}', mu=0, sigma=beta_variance))

            years = self.df_train.index.values
            X = self.df_train[X_columns]

            gamma = pm.InverseGamma(
                'gamma', 
                alpha=hyperprior_gamma_alpha, 
                beta=hyperprior_gamma_beta
            )

            initial_alpha = pm.Gamma(
                f'alpha_{years[0]}', 
                alpha=hyperprior_alpha_alpha, 
                beta=hyperprior_alpha_beta
            ) 
            alphas = {years[0]: initial_alpha}

            epsilon_sigma = pm.Gamma('epsilon_sigma', alpha=1, beta=1)
            for year in years[1:]:
                epsilon = pm.Normal(f'epsilon_{year}', mu=0, sigma=epsilon_sigma)
                alpha = pm.Deterministic(
                    f'alpha_{year}', 
                    alphas[year-1] * pm.math.exp(pm.math.dot(X.loc[year], betas) + epsilon)
                )
                alphas[year] = alpha

            # Functions for the custom distribution (truncated Weibull)
            def weibull_logp(value, gamma, alpha):
                return -np.log(alpha) + (gamma-1)*np.log(value) + (1-value**gamma)/(alpha*gamma)

            def weibull_random(gamma, alpha, rng=None, size=None):
                u = rng.uniform(size=size)
                return (1-np.log(1-u)*alpha*gamma)**(1/gamma)
            
            y_obs = []
            for year in years:
                y_obs.append(
                    pm.CustomDist(
                        f'y_{year}', 
                        gamma, 
                        alphas[year],
                        logp=weibull_logp,
                        random=weibull_random,
                        observed=self.df_train.loc[year][y_column]
                    )
                )

    def fit(
        self,
        draws: int=2000,
        tune: int=2000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='pymc'
    ):
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler
            )

    def get_trace(self):
        return self.trace

    def prior_predictive(
        self,
        samples: int=1000
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior_pm = pm.sample_prior_predictive(samples=samples)
        prior = prior_pm['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = prior_pm['prior_predictive']
        prior_df = pd.DataFrame()
        for var in list(prior_pred.data_vars):
            vals = prior_pred[var].values.flatten()
            # select 10000 random samples from the prior
            prior_df[var] = np.random.choice(vals, 10000)
        merged_prior = pd.concat([prior, prior_df], axis=1)
        return merged_prior

    def posterior_predictive(
        self
    ):
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(self.trace)
        post_pred = pm_post['posterior_predictive']
        post_df = pd.DataFrame()
        for var in list(post_pred.data_vars):
            vals = post_pred[var].values.flatten()
            # select 10000 random samples from the posterior
            post_df[var] = np.random.choice(vals, 10000)
        merged_post = pd.concat([pm_trace, post_df], axis=1)
        return merged_post

    def predict(
        self
    ):
        def predict_row(row, df_test, X_columns, year):
            row[f'epsilon_{year}'] = np.random.normal(loc=0, scale=row['epsilon_sigma'])
            cum_sum = 0
            for X_column in X_columns:
                cum_sum += row[f'beta_{X_column}'] * df_test.loc[year, X_column]
            cum_sum += row[f'epsilon_{year}']
            row[f'alpha_{year}'] = row[f'alpha_{year - 1}'] * np.exp(cum_sum)
            row[f'y_{year}'] = Weibull(gamma=row['gamma'], alpha=row[f'alpha_{year}']).sample(1)[0]
            return row

        trace_df = self.trace['posterior'].to_dataframe().reset_index(drop=True)

        # Apply this to the trace_df
        for year in self.df_test.index:
            trace_df = trace_df.apply(
                lambda row: predict_row(
                    row=row, 
                    df_test=self.df_test, 
                    X_columns=self.X_columns,
                    year=year
                ),
            axis=1)

        return trace_df
        

class Univariate_GeneralisedPareto_TimeSeries(ABM):
    def __init__(
        self,
        panel_df: pd.DataFrame,
        train_until: int=None,
        y_column: str='net_worth',
        X_columns: list[str]=['constant', 'log_change_gdp_pc', 'log_change_MSCI'],
        hyperprior_gamma_alpha: float=33, # Estimated from the data
        hyperprior_gamma_beta: float=30, # Estimated from the data
        hyperprior_sigma_alpha: float=1, 
        hyperprior_sigma_beta: float=1 
    ):
        """ Initialize the model """
        self.y_column = y_column
        self.X_columns = X_columns

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
        self.df = panel_df.set_index('year')
        if train_until is not None:
            self.df_train = self.df.loc[:train_until]
            self.df_test = self.df.loc[train_until+1:]
        else:
            self.df_train = self.df
            self.df_test = None

        # Check how long the panel is
        MAX_LENGTH = 15
        if len(self.df_train) > MAX_LENGTH:
            print(f"Panel is too long, truncating to {MAX_LENGTH} years")
            self.df_train = self.df_train.tail(MAX_LENGTH)

        # Check that there are at least len(X_columns)+2 observations
        assert len(self.df_train) >= len(X_columns)+2, "Not enough observations in the panel"

        self.model = pm.Model()

        with self.model:
            betas = []
            for cov in X_columns:
                beta_variance = pm.Gamma(f'beta_variance_{cov}', alpha=1, beta=1)
                betas.append(pm.Normal(f'beta_{cov}', mu=0, sigma=beta_variance))

            years = self.df_train.index.values
            X = self.df_train[X_columns]

            initial_sigma = pm.Gamma(
                f'sigma_{years[0]}', 
                alpha=hyperprior_sigma_alpha, 
                beta=hyperprior_sigma_beta
            ) 
            sigmas = {years[0]: initial_sigma}

            epsilon_sigma = pm.Gamma('epsilon_sigma', alpha=1, beta=1)
            for year in years[1:]:
                epsilon = pm.Normal(f'epsilon_{year}', mu=0, sigma=epsilon_sigma)
                sigma = pm.Deterministic(
                    f'sigma_{year}', 
                    sigmas[year-1] * pm.math.exp(pm.math.dot(X.loc[year], betas) + epsilon)
                )
                sigmas[year] = sigma

            gamma_false_loc = pm.Gamma(
                'gamma_false_loc', 
                alpha=hyperprior_gamma_alpha, 
                beta=hyperprior_gamma_beta
            )
            gamma = pm.Deterministic('gamma', gamma_false_loc - 0.5) # From Example 2.8, Dombry, Padoan and Rizelli (2023)

            mu=1 # Mu is fixed at 1

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

            y_obs = []
            for year in years:
                y_obs.append(
                    pm.CustomDist(
                        f'y_{year}', 
                        mu,
                        gamma, 
                        sigmas[year],
                        logp=gpd_logp,
                        random=gpd_random,
                        observed=self.df_train.loc[year][y_column]
                    )
                )

    def fit(
        self,
        draws: int=2000,
        tune: int=2000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='nutpie'
    ):
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler
            )

    def get_trace(self):
        return self.trace

    def prior_predictive(
        self,
        samples: int=1000
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior_pm = pm.sample_prior_predictive(samples=samples)
        prior = prior_pm['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = prior_pm['prior_predictive']
        prior_df = pd.DataFrame()
        for var in list(prior_pred.data_vars):
            vals = prior_pred[var].values.flatten()
            # select 10000 random samples from the prior
            prior_df[var] = np.random.choice(vals, 10000)
        merged_prior = pd.concat([prior, prior_df], axis=1)
        return merged_prior

    def posterior_predictive(
        self
    ):
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(self.trace)
        post_pred = pm_post['posterior_predictive']
        post_df = pd.DataFrame()
        for var in list(post_pred.data_vars):
            vals = post_pred[var].values.flatten()
            # select 10000 random samples from the posterior
            post_df[var] = np.random.choice(vals, 10000)
        merged_post = pd.concat([pm_trace, post_df], axis=1)
        return merged_post

    def predict(
        self
    ):
        def predict_row(row, df_test, X_columns, year):
            row[f'epsilon_{year}'] = np.random.normal(loc=0, scale=row['epsilon_sigma'])
            cum_sum = 0
            for X_column in X_columns:
                cum_sum += row[f'beta_{X_column}'] * df_test.loc[year, X_column]
            cum_sum += row[f'epsilon_{year}']
            row[f'sigma_{year}'] = row[f'sigma_{year - 1}'] * np.exp(cum_sum)
            row[f'y_{year}'] = GeneralisedPareto(gamma=row['gamma'], sigma=row[f'sigma_{year}'], mu=1.0).sample(1)[0]
            return row

        trace_df = self.trace['posterior'].to_dataframe().reset_index(drop=True)

        # Apply this to the trace_df
        for year in self.df_test.index:
            trace_df = trace_df.apply(
                lambda row: predict_row(
                    row=row, 
                    df_test=self.df_test, 
                    X_columns=self.X_columns,
                    year=year
                ),
            axis=1)

        return trace_df

class Univariate_Pareto_TimeSeries_NoCovariates(ABM):
    def __init__(
        self, 
        panel_df: pd.DataFrame,
        train_until: int=None,
        y_column: str='net_worth', 
        hyperprior_alpha: float=41, # Estimated from the data
        hyperprior_beta: float=44 # Estimated from the data
    ):
        """ Initialize the model """

        # Assert that the passed hyperpriors are positive
        assert hyperprior_alpha > 0, "hyperprior_alpha must be positive"
        assert hyperprior_beta > 0, "hyperprior_beta must be positive"
        
        self.hyperprior_alpha = hyperprior_alpha
        self.hyperprior_beta = hyperprior_beta

        self.y_column = y_column

        self.trace = None
        self.df = panel_df.set_index('year')
        if train_until is not None:
            self.df_train = self.df.loc[:train_until]
            self.df_test = self.df.loc[train_until+1:]
        else:
            self.df_train = self.df
            self.df_test = None

        # Check how long the panel is
        # MAX_LENGTH = 15
        # if len(self.df_train) > MAX_LENGTH:
        #     print(f"Panel is too long, truncating to {MAX_LENGTH} years")
        #     self.df_train = self.df_train.tail(MAX_LENGTH) TODO: Verify this check is indeed not needed in all models without covariates

        self.model = pm.Model()
        with self.model:    
            years = self.df_train.index.values 

            one_over_alphas = {}
            alphas = {}
            for year in years:
                one_over_alpha = pm.Gamma(
                    f'one_over_alpha_{year}', 
                    alpha=self.hyperprior_alpha,
                    beta=self.hyperprior_beta
                )
                alpha = pm.Deterministic(f'alpha_{year}', 1 / one_over_alpha)
                one_over_alphas[year] = one_over_alpha
                alphas[year] = alpha

            y_obs = []
            for year in years:
                y_obs.append(
                    pm.Pareto(
                        f'y_{year}', 
                        alpha=one_over_alphas[year], 
                        m = 1.0,
                        observed=self.df_train.loc[year][y_column])
                )

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
                nuts_sampler=nuts_sampler
            )

    def get_trace(self):
        """ Return the trace """
        return self.trace

    def prior_predictive(
        self,
        samples: int=1000
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior_pm = pm.sample_prior_predictive(samples=samples)
        prior = prior_pm['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = prior_pm['prior_predictive']
        prior_df = pd.DataFrame()
        for var in list(prior_pred.data_vars):
            vals = prior_pred[var].values.flatten()
            # select 10000 random samples from the prior
            prior_df[var] = np.random.choice(vals, 10000)
        merged_prior = pd.concat([prior, prior_df], axis=1)
        return merged_prior

    def posterior_predictive(
        self
    ):
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(self.trace)
        post_pred = pm_post['posterior_predictive']
        post_df = pd.DataFrame()
        for var in list(post_pred.data_vars):
            vals = post_pred[var].values.flatten()
            # select 10000 random samples from the posterior
            post_df[var] = np.random.choice(vals, 10000)
        merged_post = pd.concat([pm_trace, post_df], axis=1)
        return merged_post
        
    def predict(
        self
    ):
        pass

        
class Univariate_Weibull_TimeSeries_NoCovariates(ABM):
    def __init__(
        self,
        panel_df: pd.DataFrame,
        train_until: int=None,
        y_column: str='net_worth',
        hyperprior_gamma_alpha: float=1.7, # Estimated from the data
        hyperprior_gamma_beta: float=5.9, # Estimated from the data
        hyperprior_alpha_alpha: float=1.0,
        hyperprior_alpha_beta: float=1.0
    ):
        """ Initialize the model """
        self.y_column = y_column

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
        self.df = panel_df.set_index('year')
        if train_until is not None:
            self.df_train = self.df.loc[:train_until]
            self.df_test = self.df.loc[train_until+1:]
        else:
            self.df_train = self.df
            self.df_test = None

        # Check how long the panel is
        # MAX_LENGTH = 15
        # if len(self.df_train) > MAX_LENGTH:
        #     print(f"Panel is too long, truncating to {MAX_LENGTH} years")
        #     self.df_train = self.df_train.tail(MAX_LENGTH)

        self.model = pm.Model()

        with self.model:
            years = self.df_train.index.values

            gamma = pm.InverseGamma(
                'gamma', 
                alpha=hyperprior_gamma_alpha, 
                beta=hyperprior_gamma_beta
            )
            alphas = {}

            for year in years:
                alpha = pm.Gamma(
                    f'alpha_{year}', 
                    alpha=hyperprior_alpha_alpha, 
                    beta=hyperprior_alpha_beta
                    )
                alphas[year] = alpha

            # Functions for the custom distribution (truncated Weibull)
            def weibull_logp(value, gamma, alpha):
                return -np.log(alpha) + (gamma-1)*np.log(value) + (1-value**gamma)/(alpha*gamma)

            def weibull_random(gamma, alpha, rng=None, size=None):
                u = rng.uniform(size=size)
                return (1-np.log(1-u)*alpha*gamma)**(1/gamma)
            
            y_obs = []
            for year in years:
                y_obs.append(
                    pm.CustomDist(
                        f'y_{year}', 
                        gamma, 
                        alphas[year],
                        logp=weibull_logp,
                        random=weibull_random,
                        observed=self.df_train.loc[year][y_column]
                    )
                )

    def fit(
        self,
        draws: int=2000,
        tune: int=2000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='pymc'
    ):
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler
            )

    def get_trace(self):
        return self.trace

    def prior_predictive(
        self,
        samples: int=1000
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior_pm = pm.sample_prior_predictive(samples=samples)
        prior = prior_pm['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = prior_pm['prior_predictive']
        prior_df = pd.DataFrame()
        for var in list(prior_pred.data_vars):
            vals = prior_pred[var].values.flatten()
            # select 10000 random samples from the prior
            prior_df[var] = np.random.choice(vals, 10000)
        merged_prior = pd.concat([prior, prior_df], axis=1)
        return merged_prior

    def posterior_predictive(
        self
    ):
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(self.trace)
        post_pred = pm_post['posterior_predictive']
        post_df = pd.DataFrame()
        for var in list(post_pred.data_vars):
            vals = post_pred[var].values.flatten()
            # select 10000 random samples from the posterior
            post_df[var] = np.random.choice(vals, 10000)
        merged_post = pd.concat([pm_trace, post_df], axis=1)
        return merged_post

    def predict(
        self
    ):
        pass
        

class Univariate_GeneralisedPareto_TimeSeries_NoCovariates(ABM):
    def __init__(
        self,
        panel_df: pd.DataFrame,
        train_until: int=None,
        y_column: str='net_worth',
        hyperprior_gamma_alpha: float=33, # Estimated from the data
        hyperprior_gamma_beta: float=30, # Estimated from the data
        hyperprior_sigma_alpha: float=1, 
        hyperprior_sigma_beta: float=1 
    ):
        """ Initialize the model """
        self.y_column = y_column

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
        self.df = panel_df.set_index('year')
        if train_until is not None:
            self.df_train = self.df.loc[:train_until]
            self.df_test = self.df.loc[train_until+1:]
        else:
            self.df_train = self.df
            self.df_test = None

        # Check how long the panel is
        # MAX_LENGTH = 15
        # if len(self.df_train) > MAX_LENGTH:
        #     print(f"Panel is too long, truncating to {MAX_LENGTH} years")
        #     self.df_train = self.df_train.tail(MAX_LENGTH)

        self.model = pm.Model()

        with self.model:
            years = self.df_train.index.values

            sigmas = {}
            for year in years:
                sigma = pm.Gamma(
                    f'sigma_{year}',
                    alpha=hyperprior_sigma_alpha,
                    beta=hyperprior_sigma_beta
                )
                sigmas[year] = sigma

            gamma_false_loc = pm.Gamma(
                'gamma_false_loc', 
                alpha=hyperprior_gamma_alpha, 
                beta=hyperprior_gamma_beta
            )
            gamma = pm.Deterministic('gamma', gamma_false_loc - 0.5) # From Example 2.8, Dombry, Padoan and Rizelli (2023)

            mu=1 # Mu is fixed at 1

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

            y_obs = []
            for year in years:
                y_obs.append(
                    pm.CustomDist(
                        f'y_{year}', 
                        mu,
                        gamma, 
                        sigmas[year],
                        logp=gpd_logp,
                        random=gpd_random,
                        observed=self.df_train.loc[year][y_column]
                    )
                )

    def fit(
        self,
        draws: int=2000,
        tune: int=2000,
        chains: int=4,
        cores: int=4,
        target_accept: float=0.975,
        nuts_sampler: str='nutpie'
    ):
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler
            )

    def get_trace(self):
        return self.trace

    def prior_predictive(
        self,
        samples: int=1000
    ):
        """ Generate prior predictive samples """
        with self.model:
            prior_pm = pm.sample_prior_predictive(samples=samples)
        prior = prior_pm['prior'].to_dataframe().reset_index(drop=True)
        prior_pred = prior_pm['prior_predictive']
        prior_df = pd.DataFrame()
        for var in list(prior_pred.data_vars):
            vals = prior_pred[var].values.flatten()
            # select 10000 random samples from the prior
            prior_df[var] = np.random.choice(vals, 10000)
        merged_prior = pd.concat([prior, prior_df], axis=1)
        return merged_prior

    def posterior_predictive(
        self
    ):
        """ Generate posterior predictive samples """
        pm_trace = self.trace['posterior'].to_dataframe().reset_index(drop=True)
        with self.model:
            pm_post = pm.sample_posterior_predictive(self.trace)
        post_pred = pm_post['posterior_predictive']
        post_df = pd.DataFrame()
        for var in list(post_pred.data_vars):
            vals = post_pred[var].values.flatten()
            # select 10000 random samples from the posterior
            post_df[var] = np.random.choice(vals, 10000)
        merged_post = pd.concat([pm_trace, post_df], axis=1)
        return merged_post

    def predict(
        self
    ):
        def predict_row(row, df_test, X_columns, year):
            row[f'epsilon_{year}'] = np.random.normal(loc=0, scale=row['epsilon_sigma'])
            cum_sum = 0
            for X_column in X_columns:
                cum_sum += row[f'beta_{X_column}'] * df_test.loc[year, X_column]
            cum_sum += row[f'epsilon_{year}']
            row[f'sigma_{year}'] = row[f'sigma_{year - 1}'] * np.exp(cum_sum)
            row[f'y_{year}'] = GeneralisedPareto(gamma=row['gamma'], sigma=row[f'sigma_{year}'], mu=1.0).sample(1)[0]
            return row

        trace_df = self.trace['posterior'].to_dataframe().reset_index(drop=True)

        # Apply this to the trace_df
        for year in self.df_test.index:
            trace_df = trace_df.apply(
                lambda row: predict_row(
                    row=row, 
                    df_test=self.df_test, 
                    X_columns=self.X_columns,
                    year=year
                ),
            axis=1)

        return trace_df