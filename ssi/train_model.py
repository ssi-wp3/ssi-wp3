from .feature_extraction import FeatureExtractorType, FeatureExtractorFactory
from .label_extractor import LabelExtractor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.discovery import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble._voting import _BaseVoting
from sklearn.ensemble._stacking import _BaseStacking
from hiclass import LocalClassifierPerParentNode
from typing import List, Dict, Callable
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
        models["hiclass"] = LocalClassifierPerParentNode #(local_classifier=LogisticRegression(), verbose=1)       
        return models


def evaluate(y_true: np.array, y_pred: np.array) -> Dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def train_model(dataframe: pd.DataFrame,
                receipt_text_column: str,
                coicop_column: str,
                label_extractor: LabelExtractor,
                feature_extractor: FeatureExtractorType,
                model_type: str,
                test_size: float,
                number_of_jobs: int = -1,
                verbose: bool = False):
    model_factory = ModelFactory()
    if model_type == "hiclass":
        model = model_factory.create_model(model_type, local_classifier=LogisticRegression(), verbose=1)
    else:
        model = model_factory.create_model(model_type)  # , n_jobs=number_of_jobs)

    feature_extractor_factory = FeatureExtractorFactory()
    feature_extractor = feature_extractor_factory.create_feature_extractor(
        feature_extractor)

    pipeline = Pipeline([
        ('features', feature_extractor),
        ('classifier', model)
    ], verbose=verbose)

    train_df, test_df = train_test_split(
        dataframe, test_size=test_size, stratify=dataframe[coicop_column])

    y_train = label_extractor.get_labels(train_df)
    pipeline.fit(train_df[receipt_text_column],
                 y_train)

    y_true = label_extractor.get_labels(test_df)
    y_pred = pipeline.predict(test_df[receipt_text_column])
    evaluation_dict = evaluate(y_true, y_pred)

    return pipeline, evaluation_dict


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

    progress_bar = tqdm.tqdm(feature_extractors)
    for feature_extractor in progress_bar:
        progress_bar.set_description(
            f"Training model {model_type} with {feature_extractor}")
        trained_pipeline, evaluate_dict = train_model(dataframe, receipt_text_column,
                                                      coicop_column, label_extractor, feature_extractor, model_type, test_size, number_of_jobs, verbose)

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
