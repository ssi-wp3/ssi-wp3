from typing import Dict, List
import pandas as pd
import os


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

    def log_dataframe(self, dataframe: pd.DataFrame, filename: str, index: bool = False):
        csv_directory = os.path.join(self.log_directory, "csv")
        if not os.path.exists(csv_directory):
            print(f"Creating log directory {csv_directory}")
            os.makedirs(csv_directory)

        html_directory = os.path.join(self.log_directory, "html")
        if not os.path.exists(html_directory):
            print(f"Creating log directory {html_directory}")
            os.makedirs(html_directory)
        
        dataframe.to_csv(os.path.join(
            csv_directory, filename + ".csv"), sep=self.delimiter, index=index)

        if isinstance(dataframe, pd.Series):
            dataframe = pd.DataFrame(dataframe)

        dataframe.to_html(os.path.join(html_directory, filename + ".html"))
        

    @staticmethod
    def log_dataframe_description(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.describe(include="all")

    @staticmethod
    def log_coicop_lengths(dataframe: pd.DataFrame, coicop_column: str) -> pd.DataFrame:
        return dataframe[coicop_column].str.len().value_counts().reset_index()

    @staticmethod
    def log_coicop_value_counts_per_length(dataframe: pd.DataFrame, coicop_column: str) -> Dict[int, pd.DataFrame]:
        coicop_lengths = DataLogger.log_coicop_lengths(
            dataframe, coicop_column)
        coicop_value_dfs = dict()
        for coicop_length in coicop_lengths.index:
            counts = dataframe[dataframe[coicop_column].str.len(
            ) == coicop_length][coicop_column].value_counts()
            coicop_value_dfs[coicop_length] = counts
        return coicop_value_dfs

    @staticmethod
    def log_number_of_unique_coicops_per_length(dataframe: pd.DataFrame, coicop_column: str) -> pd.DataFrame:
        coicop_lengths_df = DataLogger.log_coicop_lengths(
            dataframe, coicop_column)

        coicop_lengths = dict()
        for coicop_length in coicop_lengths_df.index:
            coicop_lengths[coicop_length] = dataframe[dataframe[coicop_column].str.len(
            ) == coicop_length][coicop_column].nunique()

        return pd.DataFrame(coicop_lengths, index=[0])

    @staticmethod
    def log_number_of_coicop_with_leading_zero(dataframe: pd.DataFrame, coicop_column: str, length: int = 5) -> int:
        return dataframe[dataframe[coicop_column].str.len(
        ) == length][coicop_column].str.startswith("0").sum()

    @staticmethod
    def log_unique_products_per_coicop_level(dataframe: pd.DataFrame, coicop_level_columns: str, product_id_column: str) -> pd.DataFrame:
        coicop_level_dict = dict()
        for coicop_level in coicop_level_columns:
            coicop_level_dict[coicop_level] = dataframe.groupby(
                by=coicop_level)[product_id_column].nunique()
        return pd.DataFrame(coicop_level_dict, index=[0])

    def log_before_preprocessing(self, dataframe: pd.DataFrame, coicop_column: str, filename_prefix: str = "before"):
        self.log_dataframe(DataLogger.log_dataframe_description(dataframe), f"{filename_prefix}_dataframe_description")
        self.log_dataframe(DataLogger.log_coicop_lengths(dataframe, coicop_column), f"{filename_prefix}_coicop_lengths")

        for coicop_length, counts in DataLogger.log_coicop_value_counts_per_length(dataframe, coicop_column).items():
            self.log_dataframe(counts, f"{filename_prefix}_value_counts_coicop_{coicop_length}")

        coicop_length_df = DataLogger.log_number_of_unique_coicops_per_length(
            dataframe, coicop_column)
        self.log_dataframe(coicop_length_df, f"{filename_prefix}_unique_coicops_per_length")
        # number_of_coicop_with_leading_zero = log_number_of_coicop_with_leading_zero(
        #    dataframe, coicop_column)

    def log_after_preprocessing(self, dataframe: pd.DataFrame, coicop_column: str, coicop_level_columns: List[str], product_id_column: str, filename_prefix: str = "after"):
        self.log_before_preprocessing(
            dataframe, coicop_column, filename_prefix=filename_prefix)
        unique_products_per_coicop_level = DataLogger.log_unique_products_per_coicop_level(
            dataframe, coicop_level_columns, product_id_column)
        self.log_dataframe(unique_products_per_coicop_level, f"{filename_prefix}_unique_products_per_coicop_level")
