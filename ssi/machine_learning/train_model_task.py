from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple
from sklearn.model_selection import train_test_split
from .evaluate import ConfusionMatrixEvaluator
from .trainer import ModelTrainer
from ..feature_extraction.feature_extraction import FeatureExtractorType
import pandas as pd
import luigi


class TrainModelTask(luigi.Task, ABC):
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()

    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    test_size = luigi.FloatParameter(default=0.2)
    batch_predict_size = luigi.IntParameter(default=1000)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    @property
    def model_trainer(self) -> ModelTrainer:
        model_evaluator = ConfusionMatrixEvaluator()
        return ModelTrainer(
            model_evaluator=model_evaluator,
            batch_predict_size=self.batch_predict_size,
            parquet_engine=self.parquet_engine
        )

    @property
    def training_predictions_key(self) -> str:
        return "training_predictions"

    @property
    def predictions_key(self) -> str:
        return "predictions"

    @property
    def model_key(self) -> str:
        return f"model"

    @property
    def evaluation_key(self) -> str:
        return "evaluation"

    @abstractproperty
    def training_predictions_filename(self) -> str:
        pass

    @abstractproperty
    def predictions_filename(self) -> str:
        pass

    @abstractproperty
    def model_filename(self) -> str:
        pass

    @abstractproperty
    def evaluation_filename(self) -> str:
        pass

    @abstractmethod
    def prepare_data(self) -> pd.DataFrame:
        pass

    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(dataframe, test_size=test_size)

    @abstractmethod
    def train_model(self, train_dataframe: pd.DataFrame, training_predictions_file):
        pass

    def output(self):
        return {
            self.model_key: luigi.LocalTarget(self.model_filename, format=luigi.format.Nop),
            self.training_predictions_key: luigi.LocalTarget(self.training_predictions_filename, format=luigi.format.Nop),
            self.predictions_key: luigi.LocalTarget(self.predictions_filename, format=luigi.format.Nop),
            self.evaluation_key: luigi.LocalTarget(
                self.evaluation_filename)
        }

    def run(self):
        print("Preparing data")
        dataframe = self.prepare_data()

        print("Splitting data")
        train_dataframe, test_dataframe = self.split_data(
            dataframe, test_size=self.test_size)

        print("Training model & writing training predictions to disk")
        with self.output()[self.training_predictions_key].open("w") as training_predictions_file:
            self.train_model(train_dataframe, training_predictions_file)

        print("Writing test predictions to disk")
        with self.output()[self.predictions_key].open("w") as predictions_file:
            self.model_trainer.predict(lambda: test_dataframe,
                                       predictions_file)

        print("Writing model to disk")
        with self.output()[self.model_key].open("w") as model_file:
            self.model_trainer.write_model(model_file)

        print("Writing evaluation to disk")
        with self.output()[self.evaluation_key].open("w") as evaluation_file:
            self.model_trainer.write_evaluation(evaluation_file)
