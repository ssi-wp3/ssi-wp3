from typing import Dict, Any, Callable
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from ssi.machine_learning.data_loaders.data_provider import DataProvider
import numpy as np


class ModelSettings(ABC):
    """ The ModelSettings class is a super class for the different model settings classes. """

    def __init__(self, **kwargs):
        """ Initialize the model settings.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            The model settings
        """
        self.__settings_dict = dict()
        for key, value in kwargs.items():
            self.add(key, value)

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """ Get the settings dictionary.

        Returns
        -------
        Dict[str, Any]
            The settings dictionary
        """
        return self.__settings_dict

    def __getitem__(self, key: str) -> Any:
        """ Get the value for a given key.

        Parameters
        ----------
        key : str
            The key

        Returns
        -------
        Any
            The value
        """
        return self.__settings_dict[key]

    def __setitem__(self, key: str, value: Any):
        """ Set the value for a given key.

        Parameters
        ----------
        key : str
            The key
        value : Any
            The value
        """
        self.__settings_dict[key] = value

    @abstractmethod
    def check_settings_key_exists(self, key: str) -> bool:
        """ Check if a given key exists in the settings dictionary.

        Parameters
        ----------
        key : str
            The key

        Returns
        -------
        bool
            True if the key exists, False otherwise
        """
        pass

    def add(self, key: str, value: Any) -> 'ModelSettings':
        """ Add a key-value pair to the settings dictionary.

        Parameters
        ----------
        key : str
            The key
        value : Any
            The value

        Returns
        -------
        ModelSettings
            The model settings
        """
        if not self.check_settings_key_exists(key):
            raise ValueError(f"Key {key} does not exist in the model settings")

        self.__settings_dict[key] = value
        return self


class SklearnModelSettings(ModelSettings):
    """ The SklearnModelSettings class is a model settings class for sklearn models. """

    def __init__(self, model: 'Model', **kwargs):
        """ Initialize the sklearn model settings.

        Parameters
        ----------
        model : Model
            The model
        kwargs : Dict[str, Any]
            The model settings
        """
        super().__init__(**kwargs)
        self.__model = model

    @property
    def model(self) -> 'Model':
        """ Get the model.

        Returns
        -------
        Model
            The model
        """
        return self.__model

    def check_settings_key_exists(self, key: str) -> bool:
        """ Check if a given key exists in the settings dictionary.

        Parameters
        ----------
        key : str
            The key

        Returns
        -------
        bool
            True if the key exists, False otherwise
        """
        return key in self.__model.model.get_params().keys()


class HuggingFaceModelSettings(ModelSettings):
    """ The HuggingFaceModelSettings class is a model settings class for HuggingFace models. """

    def __init__(self, **kwargs):
        """ Initialize the huggingface model settings.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            The model settings
        """
        super().__init__(**kwargs)

    @property
    def training_args(self) -> TrainingArguments:
        """ Get the training arguments.

        Returns
        -------
        TrainingArguments
            The training arguments
        """
        return TrainingArguments(**self.settings_dict)

    def check_settings_key_exists(self, key: str) -> bool:
        """ Check if a given key exists in the settings dictionary.

        Parameters
        ----------
        key : str
            The key

        Returns
        -------
        bool
            True if the key exists, False otherwise
        """
        return hasattr(self.training_args, key)


class Model(BaseEstimator, ClassifierMixin, ABC):
    """ The Model class is a super class for different classifiers. It wraps the classifier in
     a scikit learn compatible API. In this way, we can use scikit-learn classifiers, the hi-class classifier, 
     as well as PyTorch models using the same scikit-learn compatible API. In addition, this class provides a way to 
    read the training and evaluation data in a form suitable to the model type.
    """

    def __init__(self, model_create_function: Callable[[ModelSettings], Any], model_settings: ModelSettings):
        """ Initialize the model.

        Parameters
        ----------
        model_create_function : Callable[[ModelSettings], Any]
            The function to create the model
        model_settings : ModelSettings
            The model settings
        """
        self.__model = None
        self.__model_create_function = model_create_function
        self.__model_settings = model_settings

    @property
    def model(self) -> Any:
        """ Get the model.

        Returns
        -------
        Any
            The model
        """
        if not self.__model:
            self.__model = self.__model_create_function(self.model_settings)
        return self.__model

    @property
    def model_settings(self) -> ModelSettings:
        """ Get the model settings.

        Returns
        -------
        ModelSettings
            The model settings
        """
        return self.__model_settings

    @abstractmethod
    def fit(self, training_dataset: DataProvider) -> 'Model':
        """ Fit the model to the training data.

        Parameters
        ----------
        training_dataset : DataProvider
            The training dataset

        Returns
        -------
        Model
            The model
        """
        pass

    @abstractmethod
    def predict(self, test_dataset: DataProvider) -> np.ndarray:
        """ Predict the labels for the test data.

        Parameters
        ----------
        test_dataset : DataProvider
            The test dataset

        Returns
        -------
        np.ndarray
            The predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, test_dataset: DataProvider) -> np.ndarray:
        """ Predict the probabilities for the test data.

        Parameters
        ----------
        test_dataset : DataProvider
            The test dataset

        Returns
        -------
        np.ndarray
            The predicted probabilities
        """
        pass


class SklearnModel(Model):
    def __init__(self, model: ClassifierMixin, **kwargs):
        """ Initialize the sklearn model.

        Parameters
        ----------
        model : ClassifierMixin
            The sklearn model
        kwargs : Dict[str, Any]
            The model settings
        """
        model_settings = SklearnModelSettings(**kwargs)
        super().__init__(lambda sk_model_settings: model(
            sk_model_settings.settings_dict), model_settings)

    def fit(self, training_dataset: DataProvider) -> 'Model':
        """ Fit the model to the training data.

        Parameters
        ----------
        training_dataset : DataProvider
            The training dataset

        Returns
        -------
        Model
            The model
        """
        self.model.fit(training_dataset.features, training_dataset.labels)
        return self

    def predict(self, test_dataset: DataProvider) -> np.ndarray:
        """ Predict the labels for the test data.

        Parameters
        ----------
        test_dataset : DataProvider
            The test dataset

        Returns
        -------
        np.ndarray
            The predicted labels
        """
        return self.model.predict(test_dataset.features)

    def predict_proba(self, test_dataset: DataProvider) -> np.ndarray:
        """ Predict the probabilities for the test data.

        Parameters
        ----------
        test_dataset : DataProvider
            The test dataset

        Returns
        -------
        np.ndarray
            The predicted probabilities
        """
        return self.model.predict_proba(test_dataset.features)


class HuggingFaceModel(Model):
    """ The HuggingFaceModel class is a model class for HuggingFace models. """

    def __init__(self, model_name: str, **kwargs):
        """ Initialize the huggingface model.

        Parameters
        ----------
        model_name : str
            The name of the model
        kwargs : Dict[str, Any]
            The model settings
        """
        model_settings = HuggingFaceModelSettings(**kwargs)
        super().__init__(lambda model_settings: self._create_model(
            model_name, model_settings), model_settings)

    def _create_model(self, model_name: str, model_settings: HuggingFaceModelSettings, number_of_categories: int) -> Trainer:
        """ Create the model.

        Parameters
        ----------
        model_name : str
            The name of the model
        model_settings : HuggingFaceModelSettings
            The model settings
        number_of_categories : int
            The number of categories

        Returns
        -------
        Trainer
            The model
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=number_of_categories)
        return Trainer(model=model, args=model_settings.training_args)

    def fit(self, training_dataset: DataProvider) -> 'Model':
        """ Fit the model to the training data.

        Parameters
        ----------
        training_dataset : DataProvider
            The training dataset

        Returns
        -------
        Model
            The model
        """
        self.model.train(training_dataset.features, training_dataset.labels)
        return self

    def predict(self, test_dataset: DataProvider) -> np.ndarray:
        """ Predict the labels for the test data.

        Parameters
        ----------
        test_dataset : DataProvider
            The test dataset

        Returns
        -------
        np.ndarray
            The predicted labels
        """
        self.model.eval()
        return self.model.predict(test_dataset.features)

    def predict_proba(self, test_dataset: DataProvider) -> np.ndarray:
        """ Predict the probabilities for the test data.

        Parameters
        ----------
        test_dataset : DataProvider
            The test dataset

        Returns
        -------
        np.ndarray
            The predicted probabilities
        """
        self.model.eval()
        return self.model.predict_proba(test_dataset.features)
