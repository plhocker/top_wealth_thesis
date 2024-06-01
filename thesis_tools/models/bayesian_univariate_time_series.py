import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from .abstract_bayesian import AbstractBayesianModel as ABM

class Univariate_Pareto_TimeSeries(ABM):
    def __init__(
        self, 
        data, 
        n_samples=1000, 
        n_tune=1000
    ):
        pass