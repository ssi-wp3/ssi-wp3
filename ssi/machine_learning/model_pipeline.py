from typing import Any, Dict, List, Union, Optional
from ssi.machine_learning.data_loaders.data_provider import DataProvider
from .model import Model, SklearnModel, HuggingFaceModel, PyTorchModel
from .evaluate import ModelEvaluator
from .data_loaders.data_provider_factory import DataProviderFactory, DataProviderType
from .model_factory import ModelFactory
from ..report import ReportFileManager, DefaultReportFileManager
from sklearn.model_selection import BaseCrossValidator
import torch.nn as nn


class ModelPipeline:
    def __init__(self,
                 model: Model,
                 cross_validator: BaseCrossValidator,
                 model_evaluator: ModelEvaluator,
                 features_column: str,
                 label_column: str,
                 report_file_manager: ReportFileManager = DefaultReportFileManager(),
                 evaluation_metric: str = "balanced_accuracy"
                 ):
        """ Initialize the model pipeline.

        Parameters
        ----------
        model : Model
            The model
        cross_validator : BaseCrossValidator
            The cross validator
        model_evaluator : ModelEvaluator
            The model evaluator
        """
        self.__model = model
        self.__cross_validator = cross_validator
        self.__model_evaluator = model_evaluator
        self.__features_column = features_column
        self.__label_column = label_column
        self.__report_file_manager = report_file_manager
        self.__evaluation_metric = evaluation_metric
        self.__train_dataset = None
        self.__validation_split_config = dict()
        self.__test_dataset = None
        self.__test_split_config = dict()
        self.__best_model_fold = None
        self.__best_model = None
        self.__training_predictions_filename = "training_predictions.parquet"
        self.__train_evaluation_filename = "train_evaluation.json"
        self.__test_predictions_filename = "test_predictions.parquet"
        self.__test_evaluation_filename = "test_evaluation.json"
        self.__model_filename = "model.joblib"

    @property
    def model(self) -> Model:
        """ Get the model.

        Returns
        -------
        Model
            The model
        """
        return self.__model

    @property
    def report_file_manager(self) -> ReportFileManager:
        """ Get the report file manager.

        Returns
        -------
        ReportFileManager
            The report file manager
        """
        return self.__report_file_manager

    @property
    def data_provider_type(self) -> DataProviderType:
        """ Get the data provider type.

        Returns
        -------
        DataProviderType
            The data provider type
        """
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

    @property
    def cross_validator(self) -> BaseCrossValidator:
        """ Get the cross validator.

        Returns
        -------
        BaseCrossValidator
            The cross validator
        """
        return self.__cross_validator

    @property
    def model_evaluator(self) -> ModelEvaluator:
        """ Get the model evaluator.

        Returns
        -------
        ModelEvaluator
            The model evaluator
        """
        return self.__model_evaluator

    @model_evaluator.setter
    def model_evaluator(self, value: ModelEvaluator):
        self.__model_evaluator = value

    @property
    def train_dataset(self) -> DataProvider:
        """ Get the train dataset.

        Returns
        -------
        DataProvider
            The train dataset
        """
        return self.__train_dataset

    @train_dataset.setter
    def train_dataset(self, value: DataProvider):
        """ Set the train dataset.

        Parameters
        ----------
        value : DataProvider
            The train dataset
        """
        self.__train_dataset = value

    @property
    def validation_split_config(self) -> Dict[str, Any]:
        """ Get the validation split config.

        Returns
        -------
        Dict[str, Any]
            The validation split config
        """
        return self.__validation_split_config

    @validation_split_config.setter
    def validation_split_config(self, value: Dict[str, Any]):
        """ Set the validation split config.

        Parameters
        ----------
        value : Dict[str, Any]
            The validation split config
        """
        self.__validation_split_config = value

    @property
    def validation_dataset(self) -> DataProvider:
        """ Get the validation dataset.

        Returns
        -------
        DataProvider
            The validation dataset
        """
        return self.__validation_dataset

    @validation_dataset.setter
    def validation_dataset(self, value: DataProvider):
        """ Set the validation dataset.

        Parameters
        ----------
        value : DataProvider
            The validation dataset
        """
        self.__validation_dataset = value

    @property
    def test_dataset(self) -> DataProvider:
        """ Get the test dataset.

        Returns
        -------
        DataProvider
            The test dataset
        """
        return self.__test_dataset

    @test_dataset.setter
    def test_dataset(self, value: DataProvider):
        """ Set the test dataset.

        Parameters
        ----------
        value : DataProvider
            The test dataset
        """
        self.__test_dataset = value

    @property
    def test_split_config(self) -> Dict[str, Any]:
        """ Get the test split config.

        Returns
        -------
        Dict[str, Any]
            The test split config
        """
        return self.__test_split_config

    @test_split_config.setter
    def test_split_config(self, value: Dict[str, Any]):
        """ Set the test split config.

        Parameters
        ----------
        value : Dict[str, Any]
            The test split config
        """
        self.__test_split_config = value

    @property
    def features_column(self) -> str:
        """ Get the features column.

        Returns
        -------
        str
            The features column
        """
        return self.__features_column

    @features_column.setter
    def features_column(self, value: str):
        """ Set the features column.

        Parameters
        ----------
        value : str
            The features column
        """
        self.__features_column = value

    @property
    def label_column(self) -> str:
        """ Get the label column.

        Returns
        -------
        str
            The label column
        """
        return self.__label_column

    @label_column.setter
    def label_column(self, value: str):
        """ Set the label column.

        Parameters
        ----------
        value : str
            The label column
        """
        self.__label_column = value

    @property
    def evaluation_metric(self) -> str:
        """ Get the evaluation metric.

        Returns
        -------
        str
            The evaluation metric
        """
        return self.__evaluation_metric

    @property
    def best_model_fold(self) -> int:
        """ Get the best model fold.

        Returns
        -------
        int
            The best model fold
        """
        return self.__best_model_fold

    @best_model_fold.setter
    def best_model_fold(self, value: int):
        """ Set the best model fold.

        Parameters
        ----------
        value : int
            The best model fold
        """
        self.__best_model_fold = value

    @property
    def best_model(self) -> Model:
        """ Get the best model.

        Returns
        -------
        Model
            The best model
        """
        return self.__best_model

    @best_model.setter
    def best_model(self, value: Model):
        """ Set the best model.

        Parameters
        ----------
        value : Model
            The best model
        """
        self.__best_model = value

    @property
    def training_predictions_filename(self) -> str:
        """ Get the training predictions filename.

        Returns
        -------
        str
            The training predictions filename
        """
        return self.__training_predictions_filename

    @training_predictions_filename.setter
    def training_predictions_filename(self, value: str):
        """ Set the training predictions filename.

        Parameters
        ----------
        value : str
            The training predictions filename
        """
        self.__training_predictions_filename = value

    @property
    def train_evaluation_filename(self) -> str:
        """ Get the train evaluation filename.

        Returns
        -------
        str
            The train evaluation filename
        """
        return self.__train_evaluation_filename

    @train_evaluation_filename.setter
    def train_evaluation_filename(self, value: str):
        """ Set the train evaluation filename.

        Parameters
        ----------
        value : str
            The train evaluation filename
        """
        self.__train_evaluation_filename = value

    @property
    def test_predictions_filename(self) -> str:
        """ Get the test predictions filename.

        Returns
        -------
        str
            The test predictions filename
        """
        return self.__test_predictions_filename

    @test_predictions_filename.setter
    def test_predictions_filename(self, value: str):
        """ Set the test predictions filename.

        Parameters
        ----------
        value : str
            The test predictions filename
        """
        self.__test_predictions_filename = value

    @property
    def test_evaluation_filename(self) -> str:
        """ Get the test evaluation filename.

        Returns
        -------
        str
            The test evaluation filename
        """
        return self.__test_evaluation_filename

    @test_evaluation_filename.setter
    def test_evaluation_filename(self, value: str):
        """ Set the test evaluation filename.

        Parameters
        ----------
        value : str
            The test evaluation filename
        """
        self.__test_evaluation_filename = value

    @property
    def model_filename(self) -> str:
        """ Get the model filename.

        Returns
        -------
        str
            The model filename
        """
        return self.__model_filename

    @model_filename.setter
    def model_filename(self, value: str):
        """ Set the model filename.

        Parameters
        ----------
        value : str
            The model filename
        """
        self.__model_filename = value

    @staticmethod
    def pipeline_for(model: Union[str, Model],
                     report_file_manager: ReportFileManager = ReportFileManager(),
                     **kwargs) -> "ModelPipeline":
        """ Create a model pipeline.

        Parameters
        ----------
        model : Union[str, Model]
            The model
        report_file_manager : ReportFileManager, optional
            The report file manager, by default ReportFileManager()
        kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to the model

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        if isinstance(model, str):
            model = ModelFactory.model_for(
                model, report_file_manager=report_file_manager, **kwargs)
        return ModelPipeline(model)

    def with_model_evaluator(self, model_evaluator: ModelEvaluator) -> "ModelPipeline":
        """ Set the model evaluator.

        Parameters
        ----------
        model_evaluator : ModelEvaluator
            The model evaluator

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.model_evaluator = model_evaluator
        return self

    def with_features_column(self, features_column: str) -> "ModelPipeline":
        """ Set the features column.

        Parameters
        ----------
        features_column : str
            The features column

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.features_column = features_column
        return self

    def with_label_column(self, label_column: str) -> "ModelPipeline":
        """ Set the label column.

        Parameters
        ----------
        label_column : str
            The label column

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.label_column = label_column
        return self

    def with_evaluation_metric(self, evaluation_metric: str) -> "ModelPipeline":
        """ Set the evaluation metric.

        Parameters
        ----------
        evaluation_metric : str
            The evaluation metric

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.__evaluation_metric = evaluation_metric
        return self

    def with_train_dataset(self, train_dataset: Union[str, DataProvider]) -> "ModelPipeline":
        """ Set the train dataset.

        Parameters
        ----------
        train_dataset : Union[str, DataProvider]
            The train dataset

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.train_dataset = self.__get_data_provider(train_dataset)
        return self

    def save_training_predictions_to(self, training_predictions_filename: str) -> "ModelPipeline":
        """ Set the training predictions filename.

        Parameters
        ----------
        training_predictions_filename : str
            The training predictions filename

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.__training_predictions_filename = training_predictions_filename
        return self

    def save_train_evaluation_to(self, train_evaluation_filename: str) -> "ModelPipeline":
        """ Set the train evaluation filename.

        Parameters
        ----------
        train_evaluation_filename : str
            The train evaluation filename

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.__train_evaluation_filename = train_evaluation_filename
        return self

    def with_validation_dataset(self, validation_dataset: Union[str, DataProvider]) -> "ModelPipeline":
        """ Set the validation dataset.

        Parameters
        ----------
        validation_dataset : Union[str, DataProvider]
            The validation dataset

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.validation_dataset = self.__get_data_provider(validation_dataset)
        return self

    def with_validation_split_size(self, test_size: float, random_state: Optional[int] = None, shuffle: bool = True) -> "ModelPipeline":
        """ Set the validation split size.

        Parameters
        ----------
        test_size : float
            The test size
        random_state : Optional[int], optional
            The random state, by default None
        shuffle : bool, optional
            Whether to shuffle the data, by default True

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.validation_split_config = {"test_size": test_size,
                                        "random_state": random_state, "shuffle": shuffle}
        return self

    def with_test_dataset(self, test_dataset: Union[str, DataProvider]) -> "ModelPipeline":
        """ Set the test dataset.

        Parameters
        ----------
        test_dataset : Union[str, DataProvider]
            The test dataset

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.test_dataset = self.__get_data_provider(test_dataset)
        return self

    def with_test_size(self, test_size: float, random_state: Optional[int] = None, shuffle: bool = True) -> "ModelPipeline":
        """ Set the test split size.

        Parameters
        ----------
        test_size : float
            The test size
        random_state : Optional[int], optional
            The random state, by default None
        shuffle : bool, optional
            Whether to shuffle the data, by default True

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.test_split_config = {"test_size": test_size,
                                  "random_state": random_state, "shuffle": shuffle}
        return self

    def save_test_predictions_to(self, test_predictions_filename: str) -> "ModelPipeline":
        """ Set the test predictions filename.

        Parameters
        ----------
        test_predictions_filename : str
            The test predictions filename

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.__test_predictions_filename = test_predictions_filename
        return self

    def save_test_evaluation_to(self, test_evaluation_filename: str) -> "ModelPipeline":
        """ Set the test evaluation filename.

        Parameters
        ----------
        test_evaluation_filename : str
            The test evaluation filename

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.__test_evaluation_filename = test_evaluation_filename
        return self

    def save_model_to(self, model_filename: str) -> "ModelPipeline":
        """ Set the model filename.

        Parameters
        ----------
        model_filename : str
            The model filename

        Returns
        -------
        ModelPipeline
            The model pipeline
        """
        self.__model_filename = model_filename
        return self

    def train_model(self) -> List[Dict[str, Any]]:
        """ Train the model.

        Returns
        -------
        List[Dict[str, Any]]
            The model training evaluations
        """
        if not self.train_dataset:
            raise ValueError("Train dataset must be set before calling train.")

        model_training_evaluations = []

        model_evaluation_score = -1
        # TODO Split needs X, y, and sometimes groups, how to provide them?
        groups = []

        for train_indices, validation_indices in self.cross_validator.split(list(range(len(self.train_dataset))),
                                                                            self.train_dataset.label_column,
                                                                            groups):
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
        """ Run predictions on the test dataset.

        Returns
        -------
        Dict[str, Any]
            The test evaluation
        """
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
