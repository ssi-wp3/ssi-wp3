from .feature_extraction import FeatureExtractorType, FeatureExtractorFactory
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from typing import List, Dict
from enum import Enum
import pandas as pd
import tqdm
import joblib
import os


class ModelTypes(Enum):
    logistic_regression = "logistic_regression"


class ModelFactory:
    def __init__(self):
        self._models = None

    @property
    def models(self) -> Dict[ModelTypes, object]:
        if not self._models:
            self._models = {
                ModelTypes.logistic_regression: LogisticRegression()
            }
        return self._models

    def create_model(self, model_type: ModelTypes):
        return self.models[model_type]


def train_model(dataframe: pd.DataFrame,
                receipt_text_column: str,
                coicop_column: str,
                feature_extractor: FeatureExtractorType,
                model_name: str):
    model_factory = ModelFactory()
    model = model_factory.create_model(ModelTypes[model_name])

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
                                        model_name: str,
                                        output_path: str):
    dataframe = pd.read_parquet(input_filename, engine="pyarrow")

    progress_bar = tqdm.tqdm(feature_extractors)
    for feature_extractor in progress_bar:
        progress_bar.set_description(
            f"Training model {model_name} with {feature_extractor}")
        trained_pipeline = train_model(dataframe, receipt_text_column,
                                       coicop_column, feature_extractor, model_name)

        model_path = os.path.join(
            output_path, f"{model_name}_{feature_extractor}.pipeline")
        progress_bar.set_description(
            f"Saving model {model_name} with {feature_extractor} to {model_path}")
        joblib.dump(trained_pipeline, model_path)
