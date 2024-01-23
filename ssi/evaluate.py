import pandas as pd

def get_labels_and_predictions(dataframe: pd.DataFrame, label_column: str, column_prefix: str = "predict_") -> pd.DataFrame:
    y_true = dataframe[label_column]
    y_pred = dataframe[f"{column_prefix}{label_column}"]
    return y_true, y_pred

def evaluate(dataframe: pd.DataFrame, label_column: str, column_prefix: str = "predict_") -> pd.DataFrame:
    y_true, y_pred = get_labels_and_predictions(dataframe, label_column, column_prefix)

    return dataframe
