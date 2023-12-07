from typing import List
from .plots import sunburst_coicop_levels
import pandas as pd
import os
import tqdm
from wordcloud import WordCloud


def filter_coicop_level(dataframe: pd.DataFrame, coicop_level_column: str, coicop_level_value: str) -> pd.DataFrame:
    return dataframe[dataframe[coicop_level_column] == coicop_level_value]


def write_filtered_coicop_level_files(dataframe: pd.DataFrame, coicop_level_columns: List[str], data_directory: str, supermarket_name: str):
    number_of_files = len(coicop_level_columns) * \
        len(dataframe[coicop_level_columns[0]].unique())
    with tqdm.tqdm(total=number_of_files) as progress_bar:
        for coicop_level_column in coicop_level_columns:
            for coicop_level_value in dataframe[coicop_level_column].unique():
                progress_bar.set_description(
                    f"Writing {supermarket_name}_{coicop_level_value}.parquet")
                filtered_dataframe = filter_coicop_level(
                    dataframe, coicop_level_column, coicop_level_value)
                filtered_dataframe.to_parquet(
                    os.path.join(data_directory, f"{supermarket_name}_{coicop_level_value}.parquet"), engine="pyarrow", index=False)
                progress_bar.update(1)


class ProductAnalysis:
    def __init__(self, data_directory: str, plot_directory: str, supermarket_name: str, coicop_level_columns: List[str]):
        self.__data_directory = data_directory
        self.__plot_directory = plot_directory
        self.__supermarket_name = supermarket_name
        self.__coicop_level_columns = coicop_level_columns

    @property
    def data_directory(self):
        return self.__data_directory

    @property
    def plot_directory(self):
        return self.__plot_directory

    @property
    def wordcloud_plot_directory(self):
        return os.path.join(self.plot_directory, "wordclouds")

    @property
    def supermarket_name(self):
        return self.__supermarket_name

    @property
    def coicop_level_columns(self):
        return self.__coicop_level_columns

    def plot_sunburst(self, dataframe: pd.DataFrame, amount_column: str):
        sunburst_filename = os.path.join(
            self.plot_directory, f"products_{self.supermarket_name}_sunburst.html")
        sunburst_coicop_levels(
            dataframe, self.coicop_level_columns, amount_column, sunburst_filename)

    def plot_wordcloud(self, dataframe: pd.DataFrame, product_description_column: str, filename: str):
        wordcloud = WordCloud()
        product_descriptions = " ".join(
            dataframe[product_description_column].tolist())
        wordcloud.generate_from_text(product_descriptions).to_file(filename)

    def perform_product_analysis_per_coicop_level(self, dataframe: pd.DataFrame, coicop_level: str, product_description_column: str = "ean_name"):
        coicop_level_values = dataframe[coicop_level].unique()

        self.plot_wordcloud(dataframe, product_description_column, os.path.join(self.wordcloud_plot_directory,
                                                                                f"products_{self.supermarket_name}_{coicop_level}_all_wordcloud.png"))
        for coicop_level_value in coicop_level_values:
            coicop_level_value_df = filter_coicop_level(
                dataframe, coicop_level, coicop_level_value)

            wordcloud_filename = os.path.join(
                self.wordcloud_plot_directory, f"products_{self.supermarket_name}_{coicop_level}_{coicop_level_value}_wordcloud.png")
            self.plot_wordcloud(coicop_level_value_df,
                                product_description_column, wordcloud_filename)

    def perform_product_level_analysis(self, dataframe: pd.DataFrame):
        for coicop_level in self.coicop_level_columns:
            self.perform_product_analysis_per_coicop_level(
                dataframe, coicop_level)

    def analyze_products(self, dataframe: pd.DataFrame):
        self.plot_sunburst(dataframe, amount_column="count")
        self.perform_product_level_analysis(dataframe)
