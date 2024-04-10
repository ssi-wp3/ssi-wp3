from typing import Any, Callable, Dict, Union, Tuple, List
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from ..files import batched_writer
from .evaluate import ModelEvaluator
import torch
import pandas as pd
import simplejson
import joblib
import tqdm
import numpy as np


class ModelTrainer:
    def __init__(self,
                 model_evaluator: ModelEvaluator,
                 features_column: str,
                 label_column: str,
                 prediction_column: str = "y_pred",
                 batch_predict_size: int = 1000,
                 parquet_engine: str = "pyarrow"):
        self._train_evaluation_dict = {}
        self._evaluation_dict = {}
        self._pipeline = None
        self._model_evaluator = model_evaluator
        self._features_column = features_column
        self._label_column = label_column
        self._prediction_column = prediction_column
        self._batch_predict_size = batch_predict_size
        self._parquet_engine = parquet_engine

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        self._pipeline = value

    @property
    def model_evaluator(self) -> ModelEvaluator:
        return self._model_evaluator

    @property
    def features_column(self) -> str:
        return self._features_column

    @property
    def label_column(self) -> str:
        return self._label_column

    @property
    def prediction_column(self) -> str:
        return self._prediction_column

    @property
    def batch_predict_size(self) -> int:
        return self._batch_predict_size

    @batch_predict_size.setter
    def batch_predict_size(self, value: int):
        self._batch_predict_size = value

    @property
    def parquet_engine(self) -> str:
        return self._parquet_engine

    @property
    def train_evaluation_dict(self) -> Dict[str, Any]:
        return self._train_evaluation_dict

    @train_evaluation_dict.setter
    def train_evaluation_dict(self, value: Dict[str, Any]):
        self._train_evaluation_dict = value

    @property
    def evaluation_dict(self) -> Dict[str, Any]:
        return self._evaluation_dict

    @evaluation_dict.setter
    def evaluation_dict(self, value: Dict[str, Any]):
        self._evaluation_dict = value

    def get_kwargs_with_prefix(self, prefix: str, **kwargs) -> Dict[str, Any]:
        return {key: value for key, value in kwargs.items() if key.startswith(prefix)}

    def fit(self,
            training_data: pd.DataFrame,
            training_function: Callable[[pd.DataFrame, str, str, str], Any],
            training_predictions_file: str,
            label_mapping: Dict[str, int],
            ** training_kwargs
            ):
        self.pipeline = training_function(
            training_data, **training_kwargs)
        self.train_evaluation_dict = self.batch_predict(training_data,
                                                        training_predictions_file,
                                                        self.batch_predict_size,
                                                        label_mapping,
                                                        lambda dataframe, label_mapping: self.model_evaluator.evaluate_training(
                                                            dataframe, label_mapping)
                                                        )

    def predict(self,
                predictions_data: pd.DataFrame,
                predictions_file: Dict[str, int],
                label_mapping: LabelEncoder
                ):
        self.evaluation_dict = self.batch_predict(predictions_data,
                                                  predictions_file,
                                                  self.batch_predict_size,
                                                  label_mapping,
                                                  lambda dataframe, label_mapping: self.model_evaluator.evaluate(
                                                      dataframe, label_mapping)
                                                  )

    def batch_statistics(self, dataframe: pd.DataFrame, label_column: str, predicted_label_column: str) -> pd.DataFrame:
        y_true = dataframe[f"{label_column}_index"]
        y_pred = dataframe[predicted_label_column]

        print(f"y_true: {y_true.nunique()}")
        print(f"y_pred: {y_pred.nunique()}")
        return pd.DataFrame(confusion_matrix(y_true, y_pred))

    def batch_predict(self,
                      predictions_data: pd.DataFrame,
                      predictions_file: str,
                      batch_size: int,
                      label_mapping: Dict[str, int],
                      evaluation_function: Callable[[
                          pd.DataFrame], Dict[str, Any]]
                      ) -> Dict[str, Any]:
        batch_statistics = batched_writer(predictions_file,
                                          predictions_data,
                                          batch_size,
                                          process_batch=self.__predict,
                                          batch_statistics=lambda dataframe: self.batch_statistics(
                                              dataframe=dataframe,
                                              label_column=self.label_column,
                                              predicted_label_column=self._prediction_column),
                                          pipeline=self.pipeline,
                                          feature_column=self.features_column,
                                          label_mapping=label_mapping,
                                          prediction_column=self.prediction_column)
        return evaluation_function(batch_statistics, label_mapping)

    def __predict(self,
                  batch_dataframe: Union[pd.DataFrame, Tuple[torch.Tensor, torch.Tensor]],
                  progress_bar: tqdm.tqdm,
                  pipeline,
                  feature_column: str,
                  label_mapping: Dict[str, int],
                  probability_column_prefix: str = "y_proba",
                  prediction_column: str = "y_pred") -> pd.DataFrame:

        batch_dataframe, X = self.get_features(
            batch_dataframe, feature_column, label_mapping=label_mapping)

        progress_bar.set_description("Predicting probabilities")
        probabilities = pipeline.predict_proba(X)

        probability_dict = defaultdict(list)
        for probability_vector in probabilities:
            for class_label, probability_value in zip(label_mapping.keys(), probability_vector):
                probability_dict[f"{probability_column_prefix}_{class_label}"].append(
                    probability_value)

        for column, values in probability_dict.items():
            batch_dataframe[column] = values

        y_pred = pipeline.predict(X)
        print("Prediction size: ", np.unique(y_pred))
        batch_dataframe[prediction_column] = y_pred
        return batch_dataframe

    def get_features(self, batch_dataframe, feature_column: str, label_mapping: Dict[str, int]) -> Tuple[pd.DataFrame, pd.Series]:
        if isinstance(batch_dataframe, pd.DataFrame):
            batch_dataframe = batch_dataframe.copy()
            X = batch_dataframe[feature_column]
            return batch_dataframe, X.values.tolist()

        X = [batch[0].numpy() for batch in batch_dataframe]
        y = [batch[1].numpy() for batch in batch_dataframe]

        dataframe = pd.DataFrame({
            feature_column: X,
            f"{self.label_column}_index": y,
            self.label_column: self.inverse_transform(y, label_mapping)
        })
        return dataframe, np.vstack(X)

    def inverse_transform(self, labels: np.ndarray, label_mapping: Dict[str, int]) -> np.ndarray:
        return np.array([list(label_mapping.keys())[label] for label in labels])

    def write_model(self, model_file):
        joblib.dump(self.pipeline, model_file)

    def write_training_evaluation(self, evaluation_file):
        simplejson.dump(self.train_evaluation_dict,
                        evaluation_file, indent=4, sort_keys=True, ignore_nan=True)

    def write_evaluation(self, evaluation_file):
        simplejson.dump(self.evaluation_dict, evaluation_file,
                        indent=4, sort_keys=True, ignore_nan=True)
