# Internal libraries
from thesis_tools.models.bayesian_univariate_cross_sectional import Pareto_One_Stage, Weibull_One_Stage, GeneralisedPareto_One_Stage
from thesis_tools.models.bayesian_univariate_time_series import Univariate_Pareto_TimeSeries, Univariate_Weibull_TimeSeries, Univariate_GeneralisedPareto_TimeSeries

# External libraries
import numpy as np
import pandas as pd
import dill
import os

def train_or_retrieve_cross_sectional_model(
    panel_df:pd.DataFrame, 
    group:str, 
    year:int, 
    model_type:str, 
    retrain_if_saved=False,
    folder_path:str="../../Stored_Models/bayesian_univariate_cross_sectional/"
):
    model_name = f"{model_type}_{group}_{year}"
    model_path = folder_path + model_name + ".pkl"
    if os.path.exists(model_path) and not retrain_if_saved:
        with open(model_path, "rb") as f:
            model = dill.load(f)
    else:
        y_data = np.array(panel_df[(panel_df['year'] == year) & (panel_df['group'] == group)]['net_worth'].iloc[0])
        if model_type == 'Pareto':
            model = Pareto_One_Stage(y_data)
        elif model_type == 'Weibull':
            model = Weibull_One_Stage(y_data)
        elif model_type == 'GeneralisedPareto':
            model = GeneralisedPareto_One_Stage(y_data)
        model.fit()
        with open(model_path, "wb") as f:
            dill.dump(model, f)
    return model

def train_or_retrieve_time_series_model(
    panel_df:pd.DataFrame, 
    group:str, 
    covariates:list[str],
    start_year:int,
    end_year:int, 
    model_type:str, 
    retrain_if_saved=False,
    folder_path:str="../../Stored_Models/bayesian_univariate_time_series/"
):
    model_name = f"{group}_{model_type}_{start_year}_{end_year}"
    for covariate in covariates:
        model_name += f"_{covariate}"
    model_path = f"../../Stored_Models/bayesian_univariate_time_series/{model_name}.pkl"
    if os.path.exists(model_path) and not retrain_if_saved:
        with open(model_path, "rb") as f:
            model = dill.load(f)
    else:
        data = copy.deepcopy(panel_df[panel_df['group'] == group])
        data = data[data['year'] >= start_year]
        if model_type == 'Pareto':
            model = Univariate_Pareto_TimeSeries(
                panel_df=data,
                train_until=end_year,
                X_columns=covariates
            )
        elif model_type == 'Weibull':
            model = Univariate_Weibull_TimeSeries(
                panel_df=data,
                train_until=end_year,
                X_columns=covariates
            )
        elif model_type == 'GeneralisedPareto':
            model = Univariate_GeneralisedPareto_TimeSeries(
                panel_df=data,
                train_until=end_year,
                X_columns=covariates
            )
        model.fit(target_accept=0.99, nuts_sampler='pymc')
        with open(model_path, "wb") as f:
            dill.dump(model, f)
    return model