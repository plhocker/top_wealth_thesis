import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from thesis_tools.models.frequentist import Pareto, Exponential, Weibull, Gompertz, GeneralisedPareto, ExponentiatedGeneralisedPareto

class HazardRateAnalysis:
    def __init__(
        self, 
        data: np.ndarray,
        country: str,
        models: list[str] = ['Exponential', 'Gompertz', 'ExponentiatedGeneralisedPareto']
    ):
        self.country = country
        self.data = data
        self.models = {}
        for model in models:
            if model == 'Exponential':
                temp_model = Exponential()
                temp_model.fit(data)
                self.models[model] = temp_model
            elif model == 'Gompertz':
                temp_model = Gompertz()
                temp_model.fit(data)
                self.models[model] = temp_model
            elif model == 'ExponentiatedGeneralisedPareto':
                temp_model = ExponentiatedGeneralisedPareto()
                temp_model.fit(data)
                self.models[model] = temp_model
            elif model == 'Pareto':
                temp_model = Pareto()
                temp_model.fit(data)
                self.models[model] = temp_model
            elif model == 'Weibull':
                temp_model = Weibull()
                temp_model.fit(data)
                self.models[model] = temp_model
            elif model == 'GeneralisedPareto':
                temp_model = GeneralisedPareto()
                temp_model.fit(data)
                self.models[model] = temp_model
            else:
                raise ValueError(f"Model {model} not supported")

        events = np.ones_like(data)
        sorted_data = np.sort(data)
        kp_df = pd.DataFrame({'Log Wealth': sorted_data, 'billionaire': events})
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(kp_df['Log Wealth'], event_observed=kp_df['billionaire'])

    def plot_cumulative_hazard(
        self,
        title: str = 'Cumulative Hazard Rate',
        xlabel: str = 'Log Wealth',
        ylabel: str = 'Cumulative Hazard Rate',
        models: list[str] = None,
        ax: plt.Axes = None
    ):
        survival_prob = self.kmf.survival_function_
        kmf_cumulative_hazard = -np.log(survival_prob)
        
        if ax is None:
            max_val = self.data.max()
            if models is None:
                models = self.models.keys()
            for model in models:
                x_values = np.linspace(0, max_val, 1000)
                y_values = self.models[model].cumulative_hazard(x_values)
                plt.plot(x_values, y_values, label=model)
            plt.step(kmf_cumulative_hazard.index, kmf_cumulative_hazard['KM_estimate'], where='post', label='True')
            plt.legend()
            plt.title(f'{title}')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
        else:
            max_val = self.data.max()
            if models is None:
                models = self.models.keys()
            for model in models:
                x_values = np.linspace(0, max_val, 1000)
                y_values = self.models[model].cumulative_hazard(x_values)
                ax.plot(x_values, y_values, label=model)
            ax.step(kmf_cumulative_hazard.index, kmf_cumulative_hazard['KM_estimate'], where='post', label='Empirical')
            ax.legend(fontsize='small')
            ax.set_title(f'{title}', fontsize=14)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)