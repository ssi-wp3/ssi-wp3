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
    """Train a model.

    Parameters
    ----------
    model_evaluator : ModelEvaluator
        The model evaluator to use
    features_column : str
        The column name containing the features
    label_column : str
        The column name containing the labels
    prediction_column : str, optional
        The column name containing the predictions, by default "y_pred"
    batch_predict_size : int, optional
        The number of rows to predict at a time, by default 1000
    parquet_engine : str, optional
        The parquet engine to use, by default "pyarrow"
    """

    def __init__(self,
                 model_evaluator: ModelEvaluator,
                 features_column: str,
                 label_column: str,
                 prediction_column: str = "y_pred",
                 batch_predict_size: int = 1000,
                 parquet_engine: str = "pyarrow"):
        self._pipeline = None
        self._model_evaluator = model_evaluator
        self._features_column = features_column
        self._label_column = label_column
        self._prediction_column = prediction_column
        self._batch_predict_size = batch_predict_size
        self._parquet_engine = parquet_engine

    @property
    def pipeline(self):
        """The pipeline to use for training and prediction.

        Returns
        -------
        Pipeline
            The pipeline to use for training and prediction
        """
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        """Set the pipeline to use for training and prediction.

        Parameters
        ----------
        value : Pipeline
            The pipeline to use for training and prediction
        """
        self._pipeline = value

    @property
    def model_evaluator(self) -> ModelEvaluator:
        """The model evaluator to use.

        Returns
        -------
        ModelEvaluator
            The model evaluator to use
        """
        return self._model_evaluator

    @property
    def features_column(self) -> str:
        """The column name containing the features.

        Returns
        -------
        str
            The column name containing the features
        """
        return self._features_column

    @property
    def label_column(self) -> str:
        """The column name containing the labels.

        Returns
        -------
        str
            The column name containing the labels
        """
        return self._label_column

    @property
    def prediction_column(self) -> str:
        """The column name containing the predictions.

        Returns
        -------
        str
            The column name containing the predictions
        """
        return self._prediction_column

    @property
    def batch_predict_size(self) -> int:
        """The number of rows to predict at a time.

        Returns
        -------
        int
            The number of rows to predict at a time
        """
        return self._batch_predict_size

    @batch_predict_size.setter
    def batch_predict_size(self, value: int):
        """Set the number of rows to predict at a time.

        Parameters
        ----------
        value : int
            The number of rows to predict at a time
        """
        self._batch_predict_size = value

    @property
    def parquet_engine(self) -> str:
        """The parquet engine to use.

        Returns
        -------
        str
            The parquet engine to use
        """
        return self._parquet_engine

    def get_kwargs_with_prefix(self, prefix: str, **kwargs) -> Dict[str, Any]:
        """Get the kwargs with the given prefix.

        Parameters
        ----------
        prefix : str
            The prefix to filter the kwargs
        kwargs : Dict[str, Any]
            The kwargs to filter

        Returns
        -------
        Dict[str, Any]
            The kwargs with the given prefix
        """
        return {key: value for key, value in kwargs.items() if key.startswith(prefix)}

    def fit(self,
            training_data: pd.DataFrame,
            training_function: Callable[[pd.DataFrame, str, str, str], Any],
            training_predictions_file: str,
            label_mapping: Dict[str, int],
            ** training_kwargs
            ):
        """Fit the model.

        Parameters
        ----------
        training_data : pd.DataFrame
            The training data
        training_function : Callable[[pd.DataFrame, str, str, str], Any]
            The training function
        training_predictions_file : str
            The file to write the training predictions to
        label_mapping : Dict[str, int]
            The label mapping
        training_kwargs : Dict[str, Any]
            The training kwargs
        """
        self.pipeline = training_function(
            training_data, **training_kwargs)
        self.batch_predict(training_data,
                           training_predictions_file,
                           self.batch_predict_size,
                           label_mapping
                           )

    def predict(self,
                predictions_data: pd.DataFrame,
                predictions_file: Dict[str, int],
                label_mapping: LabelEncoder
                ):
        """Predict the labels for the given data.

        Parameters
        ----------
        predictions_data : pd.DataFrame
            The data to predict the labels for
        predictions_file : str
            The file to write the predictions to
        label_mapping : LabelEncoder
        """
        self.batch_predict(predictions_data,
                           predictions_file,
                           self.batch_predict_size,
                           label_mapping
                           )

    def batch_statistics(self, dataframe: pd.DataFrame, label_column: str, predicted_label_column: str) -> pd.DataFrame:
        """Calculate the batch statistics.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to calculate the statistics for
        label_column : str
            The column name containing the labels
        predicted_label_column : str
            The column name containing the predicted labels

        Returns
        -------
        pd.DataFrame
            The batch statistics
        """
        y_true = dataframe[f"{label_column}_index"]
        y_pred = dataframe[f"{predicted_label_column}_index"]
        return pd.DataFrame(confusion_matrix(y_true, y_pred))

    def batch_predict(self,
                      predictions_data: pd.DataFrame,
                      predictions_file: str,
                      batch_size: int,
                      label_mapping: Dict[str, int]
                      ) -> List[pd.DataFrame]:
        """Predict the labels for the given data.

        Parameters
        ----------
        predictions_data : pd.DataFrame
            The data to predict the labels for
        predictions_file : str
            The file to write the predictions to
        label_mapping : Dict[str, int]
            The label mapping
        batch_size : int
            The number of rows to predict at a time

        Returns
        -------
        List[pd.DataFrame]
            The batch statistics
        """
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
        return batch_statistics

    def __predict(self,
                  batch_dataframe: Union[pd.DataFrame, Tuple[torch.Tensor, torch.Tensor]],
                  progress_bar: tqdm.tqdm,
                  pipeline,
                  feature_column: str,
                  label_mapping: Dict[str, int],
                  probability_column_prefix: str = "y_proba",
                  prediction_column: str = "y_pred") -> pd.DataFrame:
        """Predict the labels for the given data.

        Parameters
        ----------
        batch_dataframe : Union[pd.DataFrame, Tuple[torch.Tensor, torch.Tensor]]
            The dataframe to predict the labels for
        progress_bar : tqdm.tqdm
            The progress bar to update
        pipeline : Pipeline
            The pipeline to use for prediction
        feature_column : str
            The column name containing the features
        label_mapping : Dict[str, int]
            The label mapping
        probability_column_prefix : str, optional
            The prefix for the probability columns, by default "y_proba"
        prediction_column : str, optional
            The column name containing the predictions, by default "y_pred"

        Returns
        -------
        pd.DataFrame
            The dataframe with the predictions
        """
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
        batch_dataframe[f"{prediction_column}_index"] = y_pred
        batch_dataframe[prediction_column] = self.inverse_transform(
            y_pred, label_mapping)
        return batch_dataframe

    def get_features(self, batch_dataframe, feature_column: str, label_mapping: Dict[str, int]) -> Tuple[pd.DataFrame, pd.Series]:
        """Get the features from the given dataframe.

        Parameters
        ----------
        batch_dataframe : pd.DataFrame
            The dataframe to get the features from
        feature_column : str
            The column name containing the features
        label_mapping : Dict[str, int]
            The label mapping

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            The dataframe with the features and the features
        """
        if isinstance(batch_dataframe, pd.DataFrame):
            batch_dataframe = batch_dataframe.copy()
            X = batch_dataframe[feature_column]
            return batch_dataframe, X.values.tolist()

        X = [batch[0].numpy() for batch in batch_dataframe]
        y = [batch[1].item() for batch in batch_dataframe]

        # TODO this concats the series into the wrong direction, fix this
        addition_columns_df = pd.concat(
            [batch[2].to_frame().transpose() for batch in batch_dataframe]).reset_index(drop=True)

        dataframe = pd.DataFrame({
            feature_column: X,
        })

        # merge dataframes
        dataframe = pd.concat([dataframe, addition_columns_df], axis=1)

        dataframe[f"{self.label_column}_index"] = y
        dataframe[self.label_column] = self.inverse_transform(y, label_mapping)
        return dataframe, np.vstack(X)

    def inverse_transform(self, labels: np.ndarray, label_mapping: Dict[str, int]) -> np.ndarray:
        """Inverse transform the labels.

        Parameters
        ----------
        labels : np.ndarray
            The labels to inverse transform
        label_mapping : Dict[str, int]
            The label mapping

        Returns
        -------
        np.ndarray
            The inverse transformed labels
        """
        keys = list(label_mapping.keys())
        return [keys[label] for label in labels]

    def write_model(self, model_file):
        """Write the model to the given file.

        Parameters
        ----------
        model_file : str
            The file to write the model to
        """
        joblib.dump(self.pipeline, model_file)

    def write_training_evaluation(self, training_predictions_file, evaluation_file):
        """Write the training evaluation to the given file.

        Parameters
        ----------
        training_predictions_file : str
            The file containing the training predictions
        evaluation_file : str
            The file to write the evaluation to
        """
        train_evaluation_dict = self.model_evaluator.evaluate_training(
            training_predictions_file, self.label_column, self.prediction_column)
        simplejson.dump(train_evaluation_dict,
                        evaluation_file, indent=4, sort_keys=True, ignore_nan=True)

    def write_evaluation(self, predictions_file, evaluation_file):
        """Write the evaluation to the given file.

        Parameters
        ----------
        predictions_file : str
            The file containing the predictions
        evaluation_file : str
            The file to write the evaluation to
        """
        evaluation_dict = self.model_evaluator.evaluate(predictions_file,
                                                        self.label_column, self.prediction_column)

        simplejson.dump(evaluation_dict, evaluation_file,
                        indent=4, sort_keys=True, ignore_nan=True)
