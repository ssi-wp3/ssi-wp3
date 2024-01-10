from .feature_extraction import FeatureExtractorType, FeatureExtractorFactory
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from typing import List, Dict
from enum import Enum
import pandas as pd
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


def train_model(dataframe: pd.DataFrame,
                receipt_text_column: str,
                coicop_column: str,
                feature_extractor: FeatureExtractorType,
                model_type: ModelType):
    model_factory = ModelFactory()
    model = model_factory.create_model(model_type)

    feature_extractor_factory = FeatureExtractorFactory()
    feature_extractor = feature_extractor_factory.create_feature_extractor(
        feature_extractor)

    pipeline = Pipeline([
        ('features', feature_extractor),
        ('classifier', model)
    ])

    pipeline.fit(dataframe[receipt_text_column], dataframe[coicop_column])
    return pipeline


def train_model_with_feature_extractors(input_filename: str,
                                        receipt_text_column: str,
                                        coicop_column: str,
                                        feature_extractors: List[FeatureExtractorType],
                                        model_type: ModelType,
                                        output_path: str):
    dataframe = pd.read_parquet(input_filename, engine="pyarrow")

    progress_bar = tqdm.tqdm(feature_extractors)
    for feature_extractor in progress_bar:
        progress_bar.set_description(
            f"Training model {model_type} with {feature_extractor}")
        trained_pipeline = train_model(dataframe, receipt_text_column,
                                       coicop_column, feature_extractor, model_type)

        model_path = os.path.join(
            output_path, f"{model_type.value}_{feature_extractor}.pipeline")
        progress_bar.set_description(
            f"Saving model {model_type.value} with {feature_extractor} to {model_path}")
        joblib.dump(trained_pipeline, model_path)


def train_models(input_filename: str,
                 receipt_text_column: str,
                 coicop_column: str,
                 feature_extractors: List[FeatureExtractorType],
                 model_types: List[ModelType],
                 output_path: str):
    progress_bar = tqdm.tqdm(model_types)
    for model_type in progress_bar:
        progress_bar.set_description(f"Training model {model_type}")
        train_model_with_feature_extractors(input_filename,
                                            receipt_text_column,
                                            coicop_column,
                                            feature_extractors,
                                            model_type,
                                            output_path)
