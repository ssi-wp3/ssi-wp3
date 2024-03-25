from typing import Any
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin


class Model(BaseEstimator, ClassifierMixin, ABC):
    """ The Model class is a super class for different classifiers. It wraps the classifier in
     a scikit learn compatible API. In this way, we can use scikit-learn classifiers, the hi-class classifier, 
     as well as PyTorch models using the same scikit-learn compatible API. In addition, this class provides a way to 
    read the training and evaluation data in a form suitable to the model type.
    """

    def __init__(self, model: Any):
        self.__model = model

    @property
    def model(self) -> Any:
        return self.__model

    @abstractmethod
    def load_data(self, **kwargs) -> Any:
        pass
