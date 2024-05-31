from abc import ABC, abstractmethod

class AbstractBayesianModel(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def prior_predictive(self):
        pass

    @abstractmethod
    def posterior_predictive(self):
        pass