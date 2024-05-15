from typing import Any, Dict, List
from data_loaders.data import DataProvider
from .model import Model
from .evaluate import ModelEvaluator
from sklearn.model_selection import BaseCrossValidator


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
        self.__best_model_fold = None
        self.__best_model = None

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def cross_validator(self) -> BaseCrossValidator:
        return self.__cross_validator

    @property
    def model_evaluator(self) -> ModelEvaluator:
        return self.__model_evaluator

    @property
    def features_column(self) -> str:
        return self.__features_column

    @property
    def label_column(self) -> str:
        return self.__label_column

    @property
    def evaluation_metric(self) -> str:
        return self.__evaluation_metric

    @property
    def best_model_fold(self) -> int:
        return self.__best_model_fold

    @best_model_fold.setter
    def best_model_fold(self, value: int):
        self.__best_model_fold = value

    @property
    def best_model(self) -> Model:
        return self.__best_model

    @best_model.setter
    def best_model(self, value: Model):
        self.__best_model = value

    def fit(self, data_loader: DataProvider) -> List[Dict[str, Any]]:
        model_training_evaluations = []

        model_evaluation_score = -1
        for train_indices, test_indices in self.cross_validator.split(data_loader.X, data_loader.y, data_loader.groups):
            train_data = data_loader.get_subset(train_indices)
            test_data = data_loader.get_subset(test_indices)

            self.model.fit(train_data[self.features_column],
                           train_data[self.label_column])

            y_train_true = train_data[self.label_column]
            y_train_pred = self.model.predict(test_data[self.features_column])

            model_evaluation = self.model_evaluator.evaluate(
                y_train_true, y_train_pred)
            model_evaluation["fold"] = model_training_evaluations
            model_evaluation["train_indices"] = train_indices
            model_evaluation["test_indices"] = test_indices

            if model_evaluation[self.model_evaluator.evaluation_metric] > model_evaluation_score:
                model_evaluation_score = model_evaluation[self.model_evaluator.evaluation_metric]
                self.best_model_fold = model_evaluation["fold"]
                self.best_model = self.model

            model_training_evaluations.append(model_evaluation)
