from .feature_extraction import FeatureExtractorType, FeatureExtractorFactory
from typing import List
import pandas as pd
import tqdm
import joblib
import os


def train_model(dataframe: pd.DataFrame,
                receipt_text_column: str,
                coicop_column: str,
                feature_extractor: FeatureExtractorType,
                model_name: str):
    pass


def train_models(input_filename: str,
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
            output_path, f"{model_name}_{feature_extractor}.model")
        progress_bar.set_description(
            f"Saving model {model_name} with {feature_extractor} to {model_path}")
        joblib.dump(trained_pipeline, model_path)
