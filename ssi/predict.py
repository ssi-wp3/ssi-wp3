import pandas as pd

def predict(pipeline, dataframe: pd.DataFrame, label_column: str, column_prefix: str = "predict_") -> pd.DataFrame:
    y_true = dataframe[label_column]
    dataframe[f"{column_prefix}{label_column}"] = pipeline.predict(y_true)
    return dataframe