from ..feature_extraction.feature_extraction import FeatureExtractorType, FeatureExtractorFactory
from ..label_extractor import LabelExtractor
from ..constants import Constants
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.discovery import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble._voting import _BaseVoting
from sklearn.ensemble._stacking import _BaseStacking
from hiclass import LocalClassifierPerParentNode
from typing import List, Dict, Callable, Any, Optional
from enum import Enum
import pandas as pd
import numpy as np
import tqdm
import joblib
import os
import json


class ModelFactory:
    def __init__(self, type_filter: str = "classifier"):
        self._models = None
        self._model_type_filter = type_filter

    @property
    def model_type_filter(self) -> str:
        return self._model_type_filter

    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    @property
    def models(self) -> Dict[str, Callable[[Dict[str, object]], object]]:
        # From: https://stackoverflow.com/questions/42160313/how-to-list-all-classification-regression-clustering-algorithms-in-scikit-learn
        if not self._models:
            self._models = {model_name: model
                            for model_name, model in all_estimators(type_filter=self.model_type_filter)
                            if not issubclass(model, _BaseVoting) and not issubclass(model, _BaseStacking)
                            }
            self._models = self._add_extra_models(self._models)

        return self._models

    def create_model(self, model_type: str, **model_kwargs):
        if model_type in self.models:
            return self.models[model_type](**model_kwargs)
        else:
            raise ValueError("Invalid model type: {model_type}")

    def _add_extra_models(self, models: Dict[str, Callable[[Dict[str, object]], object]]):
        # (local_classifier=LogisticRegression(), verbose=1)
        models["hiclass"] = LocalClassifierPerParentNode
        return models


def evaluate(y_true: np.array, y_pred: np.array, suffix: str = "") -> Dict[str, object]:
    return {
        f"accuracy{suffix}": accuracy_score(y_true, y_pred),
        f"precision{suffix}": precision_score(y_true, y_pred, average="macro"),
        f"recall{suffix}": recall_score(y_true, y_pred, average="macro"),
        f"f1{suffix}": f1_score(y_true, y_pred, average="macro"),
        f"classification_report{suffix}": classification_report(y_true, y_pred),
        f"confusion_matrix{suffix}": confusion_matrix(y_true, y_pred).tolist()
    }


def get_model(model_type: str, number_of_jobs: int = -1, verbose: bool = False):
    model_factory = ModelFactory()
    if model_type == "hiclass":
        model = model_factory.create_model(
            model_type, local_classifier=LogisticRegression(), verbose=verbose)
    else:
        model = model_factory.create_model(
            model_type)  # , n_jobs=number_of_jobs)
    return model


def fit_pipeline(train_dataframe: pd.DataFrame,
                 pipeline: Pipeline,
                 receipt_text_column: str,
                 label_column: str,
                 ) -> Pipeline:
    pipeline.fit(train_dataframe[receipt_text_column].values.tolist(),
                 train_dataframe[label_column].values.tolist())

    return pipeline


def train_model(train_dataframe: pd.DataFrame,
                model_type: str,
                feature_column: str,
                label_column: str,
                verbose: bool = False
                ) -> Pipeline:
    model = get_model(model_type, verbose=verbose)
    pipeline = Pipeline([
        ('classifier', model)
    ], verbose=verbose)
    return fit_pipeline(train_dataframe, pipeline, feature_column, label_column)


def train_model_with_feature_extractors(train_dataframe: pd.DataFrame,
                                        model_type: str,
                                        feature_extractor: FeatureExtractorType,
                                        receipt_text_column: str,
                                        label_column: str,
                                        verbose: bool = False
                                        ) -> Pipeline:
    model = get_model(model_type, verbose=verbose)
    pipeline = Pipeline([
        ('features', feature_extractor),
        ('classifier', model)
    ], verbose=verbose)
    return fit_pipeline(train_dataframe, pipeline, receipt_text_column, label_column)


def drop_labels_with_few_samples(dataframe: pd.DataFrame, label_column: str, min_samples: int = 10) -> pd.DataFrame:
    label_counts = dataframe[label_column].value_counts()
    labels_to_drop = label_counts[label_counts < min_samples].index
    return dataframe[~dataframe[label_column].isin(labels_to_drop)]


def train_and_evaluate_model(dataframe: pd.DataFrame,
                             feature_column: str,
                             label_column: str,
                             model_type: str,
                             feature_extractor: Optional[FeatureExtractorType] = None,
                             test_size: float = 0.2,
                             evaluation_function: Callable[[
                                 np.array, np.array], Dict[str, object]] = evaluate,
                             number_of_jobs: int = -1,
                             verbose: bool = False):
    # TODO check in training scripts whether the function signature should be updated to this new one.
    """ Trains and evaluates a model pipeline

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe to use for training and testing

    feature_column : str
        The column name containing the features, this can be a column containing the
        already extracted feature vectors, or the column containing the receipt text.

    label_column : str
        The column name containing the labels

    model_type : str
        The model to use, one of the classification models provided by the ModelFactory.
        At this moment, all scikit-learn classifiers are available, as well as, the hiclass hierarchical
        classifier.

    feature_extractor : FeatureExtractorType, optional
        The feature extractor to use, by default None. If None, the feature_column is assumed to contain
        the feature vectors.

    test_size : float, optional
        The test size for the train/test split, by default 0.2

    evaluation_function : Callable[[np.array, np.array], Dict[str, object]], optional
        The evaluation function to use, by default evaluate

    number_of_jobs : int, optional
        The number of jobs to use, by default -1

    verbose : bool, optional
        Specifies this if you want verbose output from the classifier, by default False

    Returns
    -------
    Pipeline
        The trained model pipeline

    Dict[str, object]
        The evaluation dictionary containing the evaluation metrics
    """
    dataframe = drop_labels_with_few_samples(
        dataframe, label_column, min_samples=1)
    train_df, test_df = train_test_split(
        dataframe, test_size=test_size, stratify=dataframe[label_column])
    if not feature_extractor:
        pipeline = train_model(train_df,
                               model_type,
                               feature_column,
                               label_column,
                               verbose)
    else:
        pipeline = train_model_with_feature_extractors(train_df,
                                                       model_type,
                                                       feature_extractor,
                                                       feature_column,
                                                       label_column,
                                                       verbose)

    evaluation_dict = evaluate_model(pipeline, test_df,
                                     feature_column, label_column, evaluation_function)

    return pipeline, evaluation_dict


def evaluate_model(pipeline: Pipeline,
                   test_df: pd.DataFrame,
                   feature_column: str,
                   label_column: str,
                   evaluation_function: Callable[[np.array, np.array], Dict[str, object]]) -> Dict[str, object]:
    """Evaluate a model pipeline

    Parameters
    ----------
    pipeline : Pipeline
        The trained model pipeline

    test_df : pd.DataFrame
        The dataframe to use for testing

    feature_column : str
        The column name containing the features, this can be a column containing the
        already extracted feature vectors, or the column containing the receipt text.

    label_column : str
        The column name containing the labels

    evaluation_function : Callable[[np.array, np.array], Dict[str, object]]
        The evaluation function to use

    Returns
    -------
    Dict[str, object]
        The evaluation dictionary containing the evaluation metrics
    """
    y_true = test_df[label_column]
    y_pred = pipeline.predict(test_df[feature_column].values.tolist())

    evaluation_dict = evaluation_function(y_true, y_pred)
    return evaluation_dict


def evaluate_hiclass(y_true: np.array, y_pred: np.array) -> Dict[str, Any]:
    evaluation_dict = []
    for i, coicop_level in enumerate(Constants.COICOP_LEVELS_COLUMNS[::-1]):
        y_true_level = [y[i] for y in y_true]
        y_pred_level = [y[i] for y in y_pred]
        evaluation_dict.append(
            evaluate(y_true_level, y_pred_level, f"_{coicop_level}"))
    return evaluation_dict


def train_model_with_feature_extractors(input_filename: str,
                                        receipt_text_column: str,
                                        coicop_column: str,
                                        label_extractor: LabelExtractor,
                                        feature_extractors: List[FeatureExtractorType],
                                        model_type: str,
                                        test_size: float,
                                        output_path: str,
                                        number_of_jobs: int = -1,
                                        verbose: bool = False
                                        ):
    dataframe = pd.read_parquet(input_filename, engine="pyarrow")
    extracted_label_column = "label"
    dataframe[extracted_label_column] = label_extractor.get_labels(
        dataframe[coicop_column])

    progress_bar = tqdm.tqdm(feature_extractors)

    evaluation_function = evaluate_hiclass if model_type == "hiclass" else evaluate
    for feature_extractor in progress_bar:
        progress_bar.set_description(
            f"Training model {model_type} with {feature_extractor}")
        trained_pipeline, evaluate_dict = train_and_evaluate_model(dataframe, receipt_text_column,
                                                                   extracted_label_column, feature_extractor, model_type, test_size, evaluation_function, number_of_jobs, verbose)

        model_path = os.path.join(
            output_path, f"{model_type.lower()}_{feature_extractor}.pipeline")
        progress_bar.set_description(
            f"Saving model {model_type.lower()} with {feature_extractor} to {model_path}")
        joblib.dump(trained_pipeline, model_path)

        evaluation_path = os.path.join(
            output_path, f"{model_type.lower()}_{feature_extractor}.evaluation.json")
        progress_bar.set_description(
            f"Saving evaluation {model_type.lower()} with {feature_extractor} to {evaluation_path}")
        with open(evaluation_path, "w") as evaluation_file:
            json.dump(evaluate_dict, evaluation_file)


def train_models(input_filename: str,
                 receipt_text_column: str,
                 coicop_column: str,
                 label_extractor: LabelExtractor,
                 feature_extractors: List[FeatureExtractorType],
                 model_types: List[str],
                 test_size: float,
                 output_path: str,
                 number_of_jobs: int = -1,
                 verbose: bool = False):
    progress_bar = tqdm.tqdm(model_types)
    for model_type in progress_bar:
        progress_bar.set_description(f"Training model {model_type}")
        train_model_with_feature_extractors(input_filename,
                                            receipt_text_column,
                                            coicop_column,
                                            label_extractor,
                                            feature_extractors,
                                            model_type,
                                            test_size,
                                            output_path,
                                            number_of_jobs,
                                            verbose)
