from typing import Dict, List, Callable
from .model import Model, SklearnModel
from sklearn.utils import all_estimators
from sklearn.ensemble._voting import _BaseVoting
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.linear_model import LogisticRegression


class ModelFactory:
    """ Factory class to create models. """

    def __init__(self, type_filter: str = "classifier"):
        """ Initialize the model factory.

        Parameters
        ----------
        type_filter : str
            The type filter
        """
        self._models = None
        self._model_type_filter = type_filter

    @property
    def model_type_filter(self) -> str:
        """ Get the model type filter.

        Returns
        -------
        str
            The model type filter
        """
        return self._model_type_filter

    @property
    def model_names(self) -> List[str]:
        """ Get the model names.

        Returns
        -------
        List[str]
            The model names
        """
        return list(self.models.keys())

    @property
    def models(self) -> Dict[str, Callable[[Dict[str, object]], Model]]:
        """ Get the available models.

        Returns
        -------
        Dict[str, Callable[[Dict[str, object]], Model]]
            The models
        """
        # From: https://stackoverflow.com/questions/42160313/how-to-list-all-classification-regression-clustering-algorithms-in-scikit-learn
        if not self._models:
            self._models = {model_name: SklearnModel(model)
                            for model_name, model in all_estimators(type_filter=self.model_type_filter)
                            if not issubclass(model, _BaseVoting) and not issubclass(model, _BaseStacking)
                            }
            self._models = self._add_extra_models(self._models)

        return self._models

    def create_model(self, model_type: str, **model_kwargs) -> Model:
        """ Create a model.

        Parameters
        ----------
        model_type : str
            The model type
        **model_kwargs
            Additional keyword arguments to pass to the model
        """
        if model_type in self.models:
            return self.models[model_type]  # (**model_kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    @staticmethod
    def model_for(model_type: str, **model_kwargs) -> Model:
        """ Create a model.

        Parameters
        ----------
        model_type : str
            The model type
        **model_kwargs
            Additional keyword arguments to pass to the model
        """
        model_factory = ModelFactory()
        return model_factory.create_model(model_type, **model_kwargs)

    def _add_extra_models(self, models: Dict[str, Callable[[Dict[str, object]], object]]):
        """ Add extra (non-sklearn) models.

        Parameters
        ----------
        models : Dict[str, Callable[[Dict[str, object]], object]]
            The models
        """
        # TODO add new model classes.
        # (local_classifier=LogisticRegression(), verbose=1)
        # models["hiclass"] = HiClassModel(model=LogisticRegression)
        # models["pytorch_classifier"] = PytorchModel(
        #    model=TorchLogisticRegression)
        return models
