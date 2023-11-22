import pandas as pd


def log_coicop_lengths(dataframe: pd.DataFrame, coicop_column: str) -> pd.DataFrame:
    return dataframe[coicop_column].str.len().value_counts().reset_index()


def log_coicop_value_counts_per_length(dataframe: pd.DataFrame, coicop_column: str, log_directory: str, delimiter: str = ";") -> pd.DataFrame:
    coicop_lengths = log_coicop_lengths(dataframe, coicop_column)
    for coicop_length in coicop_lengths.index:
        counts = dataframe[dataframe[coicop_column].str.len(
        ) == coicop_length][coicop_column].value_counts()
        counts.to_csv(os.path.join(
            log_directory, f"value_counts_coicop_{coicop_length}.csv"), delimiter=delimiter)


def log_number_of_unique_coicops_per_length(dataframe: pd.DataFrame, coicop_column: str, log_directory: str, delimiter: str = ";") -> pd.DataFrame:
    coicop_lengths = log_coicop_lengths(dataframe, coicop_column)

    coicop_lengths = dict()
    for coicop_length in coicop_lengths.index:
        coicop_lengths[coicop_length] = dataframe[dataframe[coicop_column].str.len(
        ) == coicop_length][coicop_column].nunique()

    coicop_length_df = pd.DataFrame(coicop_lengths, index=[0])
    coicop_length_df.to_csv(os.path.join(
        log_directory, "unique_coicops_per_length.csv"), delimiter=delimiter)
