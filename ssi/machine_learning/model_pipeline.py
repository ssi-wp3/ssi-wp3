from typing import Any, Dict, List, Union, Optional
from ssi.machine_learning.data_loaders.data_provider import DataProvider
from .model import Model, SklearnModel, HuggingFaceModel, PyTorchModel
from .evaluate import ModelEvaluator
from .data_loaders.data_provider_factory import DataProviderFactory, DataProviderType
from sklearn.model_selection import BaseCrossValidator
import torch.nn as nn


class ModelPipeline:
    def __init__(self,
                 model: Model,
                 cross_validator: BaseCrossValidator,
                 model_evaluator: ModelEvaluator,
                 features_column: str,
                 label_column: str,
                 evaluation_metric: str = "balanced_accuracy"
                 ):
        self.__model = model
        self.__cross_validator = cross_validator
        self.__model_evaluator = model_evaluator
        self.__features_column = features_column
        self.__label_column = label_column
        self.__evaluation_metric = evaluation_metric
        self.__test_dataset = None
        self.__best_model_fold = None
        self.__best_model = None

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def data_provider_type(self) -> DataProviderType:
        if not self.model:
            raise ValueError(
                "Trying to get data provider type without a model set.")
        if isinstance(self.model, SklearnModel):
            return DataProviderType.DataFrame
        elif isinstance(self.model, HuggingFaceModel):
            return DataProviderType.HuggingFace
        elif isinstance(self.model, nn.Module):
            return DataProviderType.PyTorch

        raise ValueError(
            f"DataProvider not known for model type: {self.model}.")

    @ property
    def cross_validator(self) -> BaseCrossValidator:
        return self.__cross_validator

    @ property
    def model_evaluator(self) -> ModelEvaluator:
        return self.__model_evaluator

    @ model_evaluator.setter
    def model_evaluator(self, value: ModelEvaluator):
        self.__model_evaluator = value

    @ property
    def train_dataset(self) -> DataProvider:
        return self.__train_dataset

    @ train_dataset.setter
    def train_dataset(self, value: DataProvider):
        self.__train_dataset = value

    @ property
    def validation_dataset(self) -> DataProvider:
        return self.__validation_dataset

    @ validation_dataset.setter
    def validation_dataset(self, value: DataProvider):
        self.__validation_dataset = value

    @ property
    def test_dataset(self) -> DataProvider:
        return self.__test_dataset

    @ test_dataset.setter
    def test_dataset(self, value: DataProvider):
        self.__test_dataset = value

    @ property
    def features_column(self) -> str:
        return self.__features_column

    @ features_column.setter
    def features_column(self, value: str):
        self.__features_column = value

    @ property
    def label_column(self) -> str:
        return self.__label_column

    @ label_column.setter
    def label_column(self, value: str):
        self.__label_column = value

    @ property
    def evaluation_metric(self) -> str:
        return self.__evaluation_metric

    @ property
    def best_model_fold(self) -> int:
        return self.__best_model_fold

    @ best_model_fold.setter
    def best_model_fold(self, value: int):
        self.__best_model_fold = value

    @ property
    def best_model(self) -> Model:
        return self.__best_model

    @ best_model.setter
    def best_model(self, value: Model):
        self.__best_model = value

    @ staticmethod
    def pipeline_for(model: Union[str, Model]) -> "ModelPipeline":
        pass

    def with_model_evaluator(self, model_evaluator: ModelEvaluator) -> "ModelPipeline":
        self.model_evaluator = model_evaluator
        return self

    def with_features_column(self, features_column: str) -> "ModelPipeline":
        self.features_column = features_column
        return self

    def with_label_column(self, label_column: str) -> "ModelPipeline":
        self.label_column = label_column
        return self

    def with_evaluation_metric(self, evaluation_metric: str) -> "ModelPipeline":
        self.__evaluation_metric = evaluation_metric
        return self

    def with_train_dataset(self, train_dataset: Union[str, DataProvider]) -> "ModelPipeline":
        self.train_dataset = self.__get_data_provider(train_dataset)
        return self

    def with_validation_dataset(self, validation_dataset: Union[str, DataProvider]) -> "ModelPipeline":
        self.validation_dataset = self.__get_data_provider(validation_dataset)
        return self

    def with_test_dataset(self, test_dataset: Union[str, DataProvider]) -> "ModelPipeline":
        self.test_dataset = self.__get_data_provider(test_dataset)
        return self

    def with_test_size(self, test_size: float, random_state: Optional[int] = None, shuffle: bool = True) -> "ModelPipeline":
        # TODO implement this
        return self

    # def train_model(self,
    #                 data_loader: DataProvider,
    #                 model_file: str,
    #                 training_predictions_file: str,
    #                 training_evaluation_file: str,
    #                 test_predictions_file: str,
    #                 test_evaluation_file: str
    #                 ):
    #     # TODO do train test split here?
    #     # TODO where to fit the label encoder?
    #     pass

    def train_model(self) -> List[Dict[str, Any]]:
        if not self.train_dataset:
            raise ValueError("Train dataset must be set before calling train.")

        model_training_evaluations = []

        model_evaluation_score = -1
        # TODO Split needs X, y, and sometimes groups, how to provide them?
        for train_indices, validation_indices in self.cross_validator.split(train_data_loader.X,
                                                                            train_data_loader.y,
                                                                            train_data_loader.groups):
            train_data = self.train_dataset.get_subset(train_indices)
            validation_data = self.train_dataset.get_subset(
                validation_indices)

            self.model.fit(train_data)

            y_train_true, y_train_pred = self.model.predict(validation_data)

            model_evaluation = self.model_evaluator.evaluate(
                y_train_true, y_train_pred)
            model_evaluation["fold"] = model_training_evaluations
            model_evaluation["train_indices"] = train_indices
            model_evaluation["validation_indices"] = validation_indices

            # TODO check if evaluation_metric is implemented in ModelEvaluator
            if model_evaluation[self.model_evaluator.evaluation_metric] > model_evaluation_score:
                model_evaluation_score = model_evaluation[self.model_evaluator.evaluation_metric]
                self.best_model_fold = model_evaluation["fold"]
                self.best_model = self.model

            model_training_evaluations.append(model_evaluation)

    def predict(self) -> Dict[str, Any]:
        y_test_true, y_test_pred = self.best_model.predict(self.test_dataset)
        test_evaluation = self.model_evaluator.evaluate(
            y_test_true, y_test_pred)
        test_evaluation["fold"] = self.best_model_fold
        return test_evaluation

    def __get_data_provider(self, data_provider: Union[str, DataProvider]) -> DataProvider:
        if isinstance(data_provider, str):
            return DataProviderFactory.create_data_provider(self.data_provider_type,
                                                            filename=data_provider,
                                                            features_column=self.features_column,
                                                            label_column=self.label_column)
        return data_provider
