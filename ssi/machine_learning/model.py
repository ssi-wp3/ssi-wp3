from typing import Dict, Any, Callable
from abc import ABC, abstractmethod, abstractproperty
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import numpy as np


class ModelSettings(ABC):
    def __init__(self):
        self.__settings_dict = dict()

    @property
    def settings_dict(self) -> Dict[str, Any]:
        return self.__settings_dict

    def __getitem__(self, key: str) -> Any:
        return self.__settings_dict[key]

    def __setitem__(self, key: str, value: Any):
        self.__settings_dict[key] = value

    @abstractmethod
    def check_settings_key_exists(self, key: str) -> bool:
        pass

    def add(self, key: str, value: Any) -> 'ModelSettings':
        if not self.check_settings_key_exists(key):
            raise ValueError(f"Key {key} does not exist in the model settings")

        self.__settings_dict[key] = value
        return self


class SklearnModelSettings(ModelSettings):
    def __init__(self, model: 'Model'):
        super().__init__()
        self.__model = model

    @property
    def model(self) -> 'Model':
        return self.__model

    def check_settings_key_exists(self, key: str) -> bool:
        return key in self.__model.model.get_params().keys()


class HuggingFaceModelSettings(ModelSettings):
    def __init__(self, output_directory: str):
        super().__init__()
        self.__output_directory = output_directory
        self.add("output_dir", output_directory)

    @property
    def output_directory(self) -> str:
        return self.__output_directory

    @property
    def training_args(self) -> TrainingArguments:
        return TrainingArguments(**self.settings_dict)

    def check_settings_key_exists(self, key: str) -> bool:
        return hasattr(self.__training_args, key)


class Model(BaseEstimator, ClassifierMixin, ABC):
    """ The Model class is a super class for different classifiers. It wraps the classifier in
     a scikit learn compatible API. In this way, we can use scikit-learn classifiers, the hi-class classifier, 
     as well as PyTorch models using the same scikit-learn compatible API. In addition, this class provides a way to 
    read the training and evaluation data in a form suitable to the model type.
    """

    def __init__(self, model_create_function: Callable[[ModelSettings], Any], model_settings: ModelSettings, classes: np.ndarray = None):
        self.__model = None
        self.__model_create_function = model_create_function
        self.__model_settings = model_settings
        self.__classes = classes

    @property
    def model(self) -> Any:
        if not self.__model:
            self.__model = self.__model_create_function(self.model_settings)
        return self.__model

    @property
    def model_settings(self) -> ModelSettings:
        return self.__model_settings

    @property
    def classes_(self) -> np.ndarray:
        return self.__classes

    @classes_.setter
    def classes_(self, value: np.ndarray):
        self.__classes = value

    @abstractmethod
    def fit(self, X, y) -> 'Model':
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        pass


class SklearnModel(Model):
    def __init__(self, model: ClassifierMixin, model_settings: SklearnModelSettings, classes: np.ndarray = None):
        super().__init__(lambda sk_model_settings: model(
            sk_model_settings.settings_dict), model_settings, classes)

    def fit(self, X, y) -> 'Model':
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)


class HuggingFaceModel(Model):
    def __init__(self, model_name: str, model_settings: HuggingFaceModelSettings, classes: np.ndarray = None):

        super().__init__(lambda model_settings: self._create_model(
            model_name, model_settings), model_settings, classes)

    def _create_model(self, model_name: str, model_settings: HuggingFaceModelSettings) -> Trainer:
        number_of_categories = len(
            np.unique(classes)) if classes is not None else None
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=number_of_categories)

        return Trainer(model=model, args=model_settings.training_args)

    def fit(self, X, y) -> 'Model':
        self.model.train()
        return self

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        return self.model.predict_proba(X)
