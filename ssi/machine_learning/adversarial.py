from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any
from .train_model import train_and_evaluate_model
from ..feature_extraction.feature_extraction import FeatureExtractorType
import pandas as pd
import numpy as np


def create_combined_dataframe(store1_dataframe: pd.DataFrame,
                              store2_dataframe: pd.DataFrame,
                              store_id_column: str,
                              receipt_text_column: str,
                              features_column: str
                              ) -> pd.DataFrame:
    """Combine two store dataframes into one dataframe for adversarial validation.
    The two dataframes are combined by concatenating the receipt text columns and the store id columns.
    The machine learning model can then be trained to predict the store id based on the receipt text column.

    Parameters
    ----------
    store1_dataframe : pd.DataFrame
        The first store dataframe

    store2_dataframe : pd.DataFrame
        The second store dataframe

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text    
    """
    return pd.concat([
        store1_dataframe[[store_id_column,
                          receipt_text_column, features_column]],
        store2_dataframe[[store_id_column,
                          receipt_text_column, features_column]]
    ])


def filter_unique_combinations(dataframe: pd.DataFrame,
                               store_id_column: str,
                               receipt_text_column: str,
                               features_column: str
                               ) -> pd.DataFrame:
    """Filter the dataframe to only contain unique combinations of store id and receipt text.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to filter

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text   

    features_column : str
        The column name containing the features

    Returns
    -------
    pd.DataFrame
        The dataframe containing only unique combinations of store id, receipt text, and features
    """
    return dataframe.drop_duplicates(subset=[store_id_column, receipt_text_column, features_column])


def train_adversarial_model(store1_dataframe: pd.DataFrame,
                            store2_dataframe: pd.DataFrame,
                            store_id_column: str,
                            receipt_text_column: str,
                            features_column: str,
                            model_type: str,
                            test_size: float = 0.2,
                            number_of_jobs: int = -1,
                            verbose: bool = False
                            ) -> Pipeline:
    """Train an adversarial model to predict the store id based on the receipt text column.

    Parameters
    ----------
    store1_dataframe : pd.DataFrame
        The first store dataframe

    store2_dataframe : pd.DataFrame
        The second store dataframe

    receipt_text_column : str
        The column name containing the receipt text

    features_column : str
        The column name containing the features

    store_id_column : str
        The column name containing the store id

    model_type : str
        The model to use

    test_size : float, optional
        The test size for the train/test split, by default 0.2

    number_of_jobs : int, optional
        The number of jobs to use, by default -1

    verbose : bool, optional
        Verbose output, by default False

    Returns
    -------
    object
        The trained model
    """
    print("Creating combined dataframe")
    combined_dataframe = create_combined_dataframe(
        store1_dataframe, store2_dataframe, store_id_column, receipt_text_column, features_column)
    print("Filtering unique combinations")
    unique_dataframe = filter_unique_combinations(
        combined_dataframe, store_id_column, receipt_text_column, features_column)
    print("Training and evaluating model")

    pipeline, evaluation_dict = train_and_evaluate_model(unique_dataframe,
                                                         receipt_text_column,
                                                         store_id_column,
                                                         feature_extractor,
                                                         model_type,
                                                         test_size,
                                                         evaluate_adversarial_pipeline,
                                                         verbose)
    return pipeline, evaluation_dict


def evaluate_adversarial_pipeline(y_true: np.array,
                                  y_pred: np.array,
                                  ) -> Dict[str, Any]:
    """
    Evaluate the adversarial pipeline.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
