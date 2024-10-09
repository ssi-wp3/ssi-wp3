from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from .evaluate import ConfusionMatrixEvaluator
from .trainer import ModelTrainer
from ..feature_extraction.feature_extraction import FeatureExtractorType
import pandas as pd
import luigi
import os


class TrainModelTask(luigi.Task, ABC):
    """ Task to train a model. """
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()

    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    label_column = luigi.Parameter()
    test_size = luigi.FloatParameter(default=0.2)
    batch_size = luigi.IntParameter(default=1000)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        """ Initialize the task. """
        super().__init__(*args, **kwargs)
        self.__model_trainer = None
        self.__start_date_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.__train_label_mapping = None
        self.__test_label_mapping = None

    @property
    def model_trainer(self) -> ModelTrainer:
        """ Get the model trainer.

        Returns
        -------
        ModelTrainer
            The model trainer
        """
        if not self.__model_trainer:
            model_evaluator = ConfusionMatrixEvaluator()
            self.__model_trainer = ModelTrainer(
                model_evaluator=model_evaluator,
                features_column=self.features_column,
                label_column=self.label_column,
                batch_predict_size=self.batch_size,
                parquet_engine=self.parquet_engine
            )
        return self.__model_trainer

    @property
    def start_date_time(self) -> str:
        """ Get the start date time.

        Returns
        -------
        str
            The start date time
        """
        return self.__start_date_time

    @property
    def training_predictions_key(self) -> str:
        """ Get the training predictions key.

        Returns
        -------
        str
            The training predictions key
        """
        return "training_predictions"

    @property
    def training_evaluation_key(self) -> str:
        """ Get the training evaluation key.

        Returns
        -------
        str
            The training evaluation key
        """
        return "training_evaluation"

    @property
    def predictions_key(self) -> str:
        """ Get the predictions key.

        Returns
        -------
        str
            The predictions key
        """
        return "predictions"

    @property
    def model_key(self) -> str:
        """ Get the model key.

        Returns
        -------
        str
            The model key
        """
        return f"model"

    @property
    def evaluation_key(self) -> str:
        """ Get the evaluation key.

        Returns
        -------
        str
            The evaluation key
        """
        return "evaluation"

    @property
    def model_directory(self) -> str:
        """ Get the model directory.

        Returns
        -------
        str
            The model directory
        """
        # Create directory for our models
        model_directory = os.path.join(self.output_directory, self.model_type)
        return os.path.join(model_directory, self.start_date_time)

    @property
    def train_label_mapping(self) -> OrderedDict[str, int]:
        """ Get the train label mapping.

        Returns
        -------
        OrderedDict[str, int]
            The train label mapping
        """
        return self.__train_label_mapping

    @train_label_mapping.setter
    def train_label_mapping(self, value: OrderedDict[str, int]):
        """ Set the train label mapping.

        Parameters
        ----------
        value : OrderedDict[str, int]
            The train label mapping
        """
        self.__train_label_mapping = value

    @property
    def test_label_mapping(self) -> OrderedDict[str, int]:
        """ Get the test label mapping.

        Returns
        -------
        OrderedDict[str, int]
            The test label mapping
        """
        return self.__test_label_mapping

    @test_label_mapping.setter
    def test_label_mapping(self, value: OrderedDict[str, int]):
        """ Set the test label mapping.

        Parameters
        ----------
        value : OrderedDict[str, int]
            The test label mapping
        """
        self.__test_label_mapping = value

    @property
    @abstractmethod
    def training_predictions_filename(self) -> str:
        """ Get the training predictions filename.

        Returns
        -------
        str
            The training predictions filename
        """
        pass

    @property
    @abstractmethod
    def training_evaluation_filename(self) -> str:
        """ Get the training evaluation filename.

        Returns
        -------
        str
            The training evaluation filename
        """
        pass

    @property
    @abstractmethod
    def predictions_filename(self) -> str:
        """ Get the predictions filename.

        Returns
        -------
        str
            The predictions filename
        """
        pass

    @property
    @abstractmethod
    def model_filename(self) -> str:
        """ Get the model filename.

        Returns
        -------
        str
            The model filename
        """
        pass

    @property
    @abstractmethod
    def evaluation_filename(self) -> str:
        """ Get the evaluation filename.

        Returns
        -------
        str
            The evaluation filename
        """
        pass

    @abstractmethod
    def prepare_data(self) -> pd.DataFrame:
        """ Prepare the data.

        Returns
        -------
        pd.DataFrame
            The prepared data
        """
        pass

    def retrieve_label_mappings(self, train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame, label_column: str):
        """ Retrieve the label mappings.

        Parameters
        ----------
        train_dataframe : pd.DataFrame
            The training dataframe
        test_dataframe : pd.DataFrame
            The test dataframe
        label_column : str
            The label column
        """
        self.train_label_mapping = OrderedDict([(original_label, index)
                                                for index, original_label in enumerate(train_dataframe[self.label_column].unique())])

        print(f"Training label mapping: {self.train_label_mapping}")
        # Test dataset can have more categories than the training dataset, add them add the end of the mapping
        # In this way, we preserve the original label->index mapping for the training dataset
        self.test_label_mapping = self.train_label_mapping
        for label in test_dataframe[label_column].unique():
            if label not in self.test_label_mapping:
                self.test_label_mapping[label] = len(self.test_label_mapping)

        print(f"Test label mapping: {self.test_label_mapping}")

    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Split the data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to split
        test_size : float
            The test size

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training and test dataframes
        """
        train_df, test_df = train_test_split(
            dataframe, test_size=test_size, stratify=dataframe[self.label_column])
        self.retrieve_label_mappings(train_df, test_df, self.label_column)

        train_df[f"{self.label_column}_index"] = train_df[self.label_column].map(
            self.train_label_mapping)
        test_df[f"{self.label_column}_index"] = test_df[self.label_column].map(
            self.test_label_mapping)
        return train_df, test_df

    @abstractmethod
    def train_model(self, train_dataframe: pd.DataFrame, training_predictions_file):
        """ Train the model.

        Parameters
        ----------
        train_dataframe : pd.DataFrame
            The training dataframe
        training_predictions_file : file
            The file to write the training predictions to
        """
        pass

    def predict(self, predictions_dataframe: pd.DataFrame, predictions_file):
        """ Predict the labels for the test dataframe and write the predictions to a file.

        Parameters
        ----------
        predictions_dataframe : pd.DataFrame
            The dataframe to predict the labels for
        predictions_file : file
            The file to write the predictions to
        """
        self.model_trainer.predict(predictions_dataframe,
                                   predictions_file,
                                   label_mapping=self.test_label_mapping,
                                   )

    def output(self):
        """ Get the output.

        Returns
        -------
        Dict[str, luigi.LocalTarget]
            The output
        """
        return {
            self.model_key: luigi.LocalTarget(self.model_filename, format=luigi.format.Nop),
            self.training_predictions_key: luigi.LocalTarget(self.training_predictions_filename, format=luigi.format.Nop),
            self.training_evaluation_key: luigi.LocalTarget(self.training_evaluation_filename),
            self.predictions_key: luigi.LocalTarget(self.predictions_filename, format=luigi.format.Nop),
            self.evaluation_key: luigi.LocalTarget(
                self.evaluation_filename)
        }

    def run(self):
        """ Run the task. """
        print("Preparing data")
        dataframe = self.prepare_data()

        print("Splitting data")
        train_dataframe, test_dataframe = self.split_data(
            dataframe, test_size=self.test_size)

        print("Training model & writing training predictions to disk")
        with self.output()[self.training_predictions_key].open("w") as training_predictions_file:
            self.train_model(train_dataframe, training_predictions_file)

        print("Writing training evaluation to disk")
        with self.output()[self.training_evaluation_key].open("w") as evaluation_file:
            self.model_trainer.write_training_evaluation(
                training_predictions_file.path, evaluation_file)

        print("Writing test predictions to disk")
        with self.output()[self.predictions_key].open("w") as predictions_file:
            self.predict(test_dataframe,
                         predictions_file)

        print("Writing model to disk")
        with self.output()[self.model_key].open("w") as model_file:
            self.model_trainer.write_model(model_file)

        print("Writing evaluation to disk")
        with self.output()[self.evaluation_key].open("w") as evaluation_file:
            self.model_trainer.write_evaluation(
                predictions_file.path, evaluation_file)
