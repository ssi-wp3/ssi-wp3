from typing import List
from .preprocess_data import split_month_year_column
from .plots import sunburst_coicop_levels
from .data_utils import export_dataframe
from wordcloud import WordCloud
import pandas as pd
import os
import tqdm
import matplotlib.pyplot as plt
import plotly.express as px


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


def get_product_counts_per_time(dataframe: pd.DataFrame, time_column: str, product_id_column: str, agg_column_name: str = "count") -> pd.DataFrame:
    dataframe = dataframe.groupby(
        [time_column])[product_id_column].nunique()
    return pd.DataFrame(dataframe).rename(columns={product_id_column: agg_column_name})


def get_product_counts_per_category_and_time(dataframe: pd.DataFrame, time_column: str, product_id_column: str, coicop_level_column: str, agg_column_name: str = "count") -> pd.DataFrame:
    dataframe = dataframe.groupby(
        [time_column, coicop_level_column])[product_id_column].nunique()
    return pd.DataFrame(dataframe).rename(columns={product_id_column: agg_column_name})


class ProductAnalysis:
    def __init__(self,
                 data_directory: str,
                 plot_directory: str,
                 supermarket_name: str,
                 coicop_level_columns: List[str],
                 year_column: str = "year",
                 year_month_column: str = "year_month",
                 product_id_columns: List[str] = ["ean_number", "product_id"]):
        self.__data_directory = data_directory
        self.__plot_directory = plot_directory
        self.__supermarket_name = supermarket_name
        self.__coicop_level_columns = coicop_level_columns
        self.__year_column = year_column
        self.__year_month_column = year_month_column
        self.__product_id_columns = product_id_columns

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

    @property
    def year_column(self):
        return self.__year_column

    @property
    def year_month_column(self):
        return self.__year_month_column

    @property
    def product_id_columns(self):
        return self.__product_id_columns

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

    def retrieve_product_counts(self, dataframe: pd.DataFrame, coicop_level: str):
        self.export_product_counts(dataframe, coicop_level)
        # self.plot_product_counts(coicop_level)

    def export_product_counts(self, dataframe, coicop_level):
        for product_id_column in self.product_id_columns:
            self.export_product_counts_per_category_and_time(
                dataframe, coicop_level, product_id_column)

    def export_product_counts_per_category_and_time(self, dataframe, coicop_level, product_id_column):
        product_counts_per_category_per_year_df = get_product_counts_per_category_and_time(
            dataframe, self.year_column, product_id_column, coicop_level)
        export_dataframe(product_counts_per_category_per_year_df, self.data_directory,
                         f"products_{self.supermarket_name}_{coicop_level}_{product_id_column}_counts_per_category_per_year")

        product_counts_per_category_per_year_month_df = get_product_counts_per_category_and_time(
            dataframe, self.year_month_column, product_id_column, coicop_level)
        export_dataframe(product_counts_per_category_per_year_month_df, self.data_directory,
                         f"products_{self.supermarket_name}_{coicop_level}_{product_id_column}_counts_per_category_per_year_month")

    def plot_product_counts(self, dataframe: pd.DataFrame, filename: str, time_column: str, count_column: str = "count"):
        # TODO use index as x-axis
        fig = px.bar(dataframe, x=dataframe.index, y=count_column,
                     title=f"Number of products per {time_column}")
        fig.write_html(filename)

    def export_products_counts_per_time_unit(self, dataframe: pd.DataFrame, supermarket_name: str, product_id_column: str, time_column: str):
        product_counts_per_time_df = get_product_counts_per_time(
            dataframe, time_column, product_id_column)

        counts_per_time_filename = f"products_{supermarket_name}_{product_id_column}_counts_per_{time_column}"
        export_dataframe(product_counts_per_time_df, self.data_directory,
                         counts_per_time_filename)
        self.plot_product_counts(product_counts_per_time_df, os.path.join(
            self.plot_directory, f"{counts_per_time_filename}.html"), time_column)

    def export_all_product_counts_per_time_unit(self, dataframe, product_id_column):
        self.export_products_counts_per_time_unit(
            dataframe, self.supermarket_name, product_id_column, self.year_column)
        self.export_products_counts_per_time_unit(
            dataframe, self.supermarket_name, product_id_column, self.year_month_column)

    def perform_product_analysis_per_coicop_level(self, dataframe: pd.DataFrame, coicop_level: str, product_description_column: str = "ean_name"):
        coicop_level_values = dataframe[coicop_level].unique()
        os.makedirs(self.wordcloud_plot_directory, exist_ok=True)
        self.plot_wordcloud(dataframe, product_description_column, os.path.join(self.wordcloud_plot_directory,
                                                                                f"products_{self.supermarket_name}_{coicop_level}_all_wordcloud.png"))

        self.retrieve_product_counts(dataframe, coicop_level)
        for coicop_level_value in coicop_level_values:
            coicop_level_value_df = filter_coicop_level(
                dataframe, coicop_level, coicop_level_value)

            wordcloud_filename = os.path.join(
                self.wordcloud_plot_directory, f"products_{self.supermarket_name}_{coicop_level}_{coicop_level_value}_wordcloud.png")
            self.plot_wordcloud(coicop_level_value_df,
                                product_description_column, wordcloud_filename)

    def perform_product_level_analysis(self, dataframe: pd.DataFrame):
        # Export product counts per time unit based on unique occurrences of product_id
        for product_id_column in self.product_id_columns:
            self.export_all_product_counts_per_time_unit(
                dataframe, product_id_column)

        for coicop_level in self.coicop_level_columns:
            self.perform_product_analysis_per_coicop_level(
                dataframe, coicop_level)

    def analyze_products(self, dataframe: pd.DataFrame):
        self.plot_sunburst(dataframe, amount_column="count")
        self.perform_product_level_analysis(dataframe)
