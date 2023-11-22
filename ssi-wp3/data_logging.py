from typing import List
import pandas as pd


class DataLogger:
    def __init__(self, log_directory: str, delimiter: str = ";"):
        self.__log_directory = log_directory
        self.__delimiter = delimiter

    @property
    def log_directory(self) -> str:
        return self.__log_directory

    @property
    def delimiter(self) -> str:
        return self.__delimiter

    @staticmethod
    def log_dataframe_description(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.describe(include="all")

    @staticmethod
    def log_coicop_lengths(dataframe: pd.DataFrame, coicop_column: str) -> pd.DataFrame:
        return dataframe[coicop_column].str.len().value_counts().reset_index()

    @staticmethod
    def log_coicop_value_counts_per_length(dataframe: pd.DataFrame, coicop_column: str) -> List[pd.DataFrame]:
        coicop_lengths = log_coicop_lengths(dataframe, coicop_column)
        coicop_value_dfs = []
        for coicop_length in coicop_lengths.index:
            counts = dataframe[dataframe[coicop_column].str.len(
            ) == coicop_length][coicop_column].value_counts()
            coicop_value_dfs.append(counts)
        return coicop_value_dfs

    @staticmethod
    def log_number_of_unique_coicops_per_length(dataframe: pd.DataFrame, coicop_column: str) -> pd.DataFrame:
        coicop_lengths = log_coicop_lengths(dataframe, coicop_column)

        coicop_lengths = dict()
        for coicop_length in coicop_lengths.index:
            coicop_lengths[coicop_length] = dataframe[dataframe[coicop_column].str.len(
            ) == coicop_length][coicop_column].nunique()

        return pd.DataFrame(coicop_lengths, index=[0])

    @staticmethod
    def log_number_of_coicop_with_leading_zero(dataframe: pd.DataFrame, coicop_column: str, length: int = 5) -> int:
        return dataframe[dataframe[coicop_column].str.len(
        ) == length][coicop_column].str.startswith("0").sum()

    @staticmethod
    def log_unique_products_per_coicop_level(dataframe: pd.DataFrame, coicop_level_column: str, product_id_column: str) -> pd.DataFrame:
        coicop_level_dict = dict()
        for coicop_level in coicop_level_columns:
            coicop_level_dict[coicop_level] = dataframe.groupby(
                by=coicop_level_column)[product_id_column].nunique()
        return pd.DataFrame(coicop_level_dict, index=[0])

    def log(self, dataframe: pd.DataFrame, coicop_column: str, coicop_level_columns: List[str], product_id_column: str):
        log_dataframe_description(dataframe).to_csv(os.path.join(
            self.log_directory, "dataframe_description.csv"), delimiter=self.delimiter)
        log_coicop_lengths(dataframe, coicop_column).to_csv(os.path.join(
            self.log_directory, "coicop_lengths.csv"), delimiter=self.delimiter)

        for counts in log_coicop_value_counts_per_length(dataframe, coicop_column):
            counts.to_csv(os.path.join(
                self.log_directory, f"value_counts_coicop_{coicop_length}.csv"), delimiter=self.delimiter)

        coicop_length_df = log_number_of_unique_coicops_per_length(
            dataframe, coico_column)
        coicop_length_df.to_csv(os.path.join(
            self.log_directory, "unique_coicops_per_length.csv"), delimiter=self.delimiter)

        # number_of_coicop_with_leading_zero = log_number_of_coicop_with_leading_zero(
        #    dataframe, coicop_column)
        unique_products_per_coicop_level = log_unique_products_per_coicop_level(
            dataframe, coicop_level_columns, product_id_column)
        unique_products_per_coicop_level.to_csv(os.path.join(
            self.log_directory, "unique_products_per_coicop_level.csv"), delimiter=self.delimiter)
