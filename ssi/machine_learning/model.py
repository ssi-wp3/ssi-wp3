from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin


class Model(BaseEstimator, ClassifierMixin, ABC):
    """ The Model class is a wrapper object around different classifiers. In this way,
    we can use scikit-learn classifiers, the hi-class classifier, as well as PyTorch models
    using the same scikit-learn compatible API. In addition, this class provides a way to 
    read the training and evaluation data in a form suitable to the model type.
    """

    def __init__(self, model_type: str, model, **model_kwargs):
        self.__model_type = model_type
        self.__model = model

    @property
    def model_type(self) -> str:
        return self.__model_type

    @abstractmethod
    def load_data(self, **kwargs) -> Any:
        pass

    @property
    def model(self):
        return self.__model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)
