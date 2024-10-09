import pandas as pd


def predict(pipeline, dataframe: pd.DataFrame, receipt_text_column: str, label_column: str, column_prefix: str = "predict_") -> pd.DataFrame:
    """ Predict the labels for the test data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to predict on
    receipt_text_column : str
        The column with the receipt text
    label_column : str
        The column with the label
    column_prefix : str
        The prefix for the new columns

    Returns
    -------
    pd.DataFrame
        The dataframe with the predicted labels
    """
    dataframe[f"{column_prefix}{label_column}"] = pipeline.predict(
        dataframe[receipt_text_column])
    return dataframe


def predict_from_file(pipeline, input_filename: str, output_filename: str, label_column: str, column_prefix: str = "predict_"):
    """ Predict the labels for the test data.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to use for prediction
    input_filename : str
        The filename of the input file
    output_filename : str
        The filename of the output file
    label_column : str
        The column with the label
    column_prefix : str
        The prefix for the new columns
    """
    dataframe = pd.read_parquet(input_filename, engine="pyarrow")
    dataframe = predict(pipeline, dataframe, label_column, column_prefix)
    dataframe.to_parquet(output_filename, engine="pyarrow")
