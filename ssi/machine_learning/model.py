from typing import Dict, Any
from abc import ABC, abstractmethod, abstractproperty
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class ModelSettings:
    def __init__(self, model: 'Model'):
        self.__model = model
        self.__settings_dict = dict()

    @property
    def model(self) -> 'Model':
        return self.__model

    @property
    def settings_dict(self) -> Dict[str, Any]:
        return self.__settings_dict

    def __getitem__(self, key: str) -> Any:
        return self.__settings_dict[key]

    def __setitem__(self, key: str, value: Any):
        self.__settings_dict[key] = value

    def add(self, key: str, value: Any) -> 'ModelSettings':
        self.__settings_dict[key] = value
        return self


class Model(BaseEstimator, ClassifierMixin, ABC):
    """ The Model class is a super class for different classifiers. It wraps the classifier in
     a scikit learn compatible API. In this way, we can use scikit-learn classifiers, the hi-class classifier, 
     as well as PyTorch models using the same scikit-learn compatible API. In addition, this class provides a way to 
    read the training and evaluation data in a form suitable to the model type.
    """

    def __init__(self, model: Any, classes: np.ndarray = None):
        self.__model_class = model
        self.__model = None
        self.__classes = classes

    @property
    def model(self) -> Any:
        if self.__model is None:
            self.__model = self.__model_class()
        return self.__model

    @property
    def classes_(self) -> np.ndarray:
        return self.__classes

    @classes_.setter
    def classes_(self, value: np.ndarray):
        self.__classes = value

    @abstractmethod
    def load_data(self, filename: str, **kwargs) -> Any:
        pass
