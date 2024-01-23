import pandas as pd

def predict(pipeline, dataframe: pd.DataFrame, label_column: str, column_prefix: str = "predict_") -> pd.DataFrame:
    y_true = dataframe[label_column]
    dataframe[f"{column_prefix}{label_column}"] = pipeline.predict(y_true)
    return dataframe


def predict_from_file(pipeline, input_filename: str, output_filename: str, label_column: str, column_prefix: str = "predict_"):
    dataframe = pd.read_parquet(input_filename, engine="pyarrow")
    dataframe = predict(pipeline, dataframe, label_column, column_prefix)
    dataframe.to_parquet(output_filename, engine="pyarrow")