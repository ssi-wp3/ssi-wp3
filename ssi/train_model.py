from .feature_extraction import FeatureExtractorType, FeatureExtractorFactory
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from typing import List, Dict
from enum import Enum
import pandas as pd
import numpy as np
import tqdm
import joblib
import os


class ModelType(Enum):
    logistic_regression = "logistic_regression"


class ModelFactory:
    def __init__(self):
        self._models = None

    @property
    def models(self) -> Dict[ModelType, object]:
        if not self._models:
            self._models = {
                ModelType.logistic_regression: LogisticRegression()
            }
        return self._models

    def create_model(self, model_type: ModelType):
        if model_type in self.models:
            return self.models[model_type]
        else:
            raise ValueError("Invalid model type: {model_type}")


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
                feature_extractor: FeatureExtractorType,
                model_type: ModelType,
                test_size: float):
    model_factory = ModelFactory()
    model = model_factory.create_model(model_type)

    feature_extractor_factory = FeatureExtractorFactory()
    feature_extractor = feature_extractor_factory.create_feature_extractor(
        feature_extractor)

    pipeline = Pipeline([
        ('features', feature_extractor),
        ('classifier', model)
    ])

    train_df, test_df = train_test_split(
        dataframe, test_size=test_size, stratify=dataframe[coicop_column])

    pipeline.fit(train_df[receipt_text_column], test_df[coicop_column])

    y_true = test_df[coicop_column]
    y_pred = pipeline.predict(test_df[receipt_text_column])
    evaluation_dict = evaluate(y_true, y_pred)

    return pipeline, evaluation_dict


def train_model_with_feature_extractors(input_filename: str,
                                        receipt_text_column: str,
                                        coicop_column: str,
                                        feature_extractors: List[FeatureExtractorType],
                                        model_type: ModelType,
                                        test_size: float,
                                        output_path: str):
    dataframe = pd.read_parquet(input_filename, engine="pyarrow")

    progress_bar = tqdm.tqdm(feature_extractors)
    for feature_extractor in progress_bar:
        progress_bar.set_description(
            f"Training model {model_type} with {feature_extractor}")
        trained_pipeline, evaluate_dict = train_model(dataframe, receipt_text_column,
                                                      coicop_column, feature_extractor, model_type, test_size)

        model_path = os.path.join(
            output_path, f"{model_type.value}_{feature_extractor}.pipeline")
        progress_bar.set_description(
            f"Saving model {model_type.value} with {feature_extractor} to {model_path}")
        joblib.dump(trained_pipeline, model_path)

        evaluation_path = os.path.join(
            output_path, f"{model_type.value}_{feature_extractor}.evaluation.json")
        progress_bar.set_description(
            f"Saving evaluation {model_type.value} with {feature_extractor} to {evaluation_path}")
        pd.DataFrame(evaluate_dict, index=[0]).to_json(
            evaluation_path, orient="columns")


def train_models(input_filename: str,
                 receipt_text_column: str,
                 coicop_column: str,
                 feature_extractors: List[FeatureExtractorType],
                 model_types: List[ModelType],
                 test_size: float,
                 output_path: str):
    progress_bar = tqdm.tqdm(model_types)
    for model_type in progress_bar:
        progress_bar.set_description(f"Training model {model_type}")
        train_model_with_feature_extractors(input_filename,
                                            receipt_text_column,
                                            coicop_column,
                                            feature_extractors,
                                            model_type,
                                            test_size,
                                            output_path)
