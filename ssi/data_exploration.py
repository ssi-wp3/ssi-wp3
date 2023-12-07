from typing import List
from .preprocess_data import split_month_year_column
from .plots import sunburst_coicop_levels
import pandas as pd
import os
import tqdm
from wordcloud import WordCloud


def export_dataframe(dataframe: pd.DataFrame, path: str, filename: str, index: bool = False, delimiter: str = ";"):
    csv_directory = os.path.join(path, "csv")

    print(f"Creating log directory {csv_directory}")
    os.makedirs(csv_directory, exist_ok=True)

    html_directory = os.path.join(path, "html")
    print(f"Creating log directory {html_directory}")
    os.makedirs(html_directory, exist_ok=True)

    dataframe.to_csv(os.path.join(
        csv_directory, filename + ".csv"), sep=delimiter, index=index)

    if isinstance(dataframe, pd.Series):
        dataframe = pd.DataFrame(dataframe)

    dataframe.to_html(os.path.join(html_directory, filename + ".html"))


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


def get_product_counts_per_time(dataframe: pd.DataFrame, time_column: str, year_month_column: str, product_id_column: str, agg_column_name: str = "count") -> pd.DataFrame:
    dataframe = split_month_year_column(dataframe, year_month_column)
    dataframe = dataframe.groupby(
        [time_column])[product_id_column].nunique()
    return pd.DataFrame(dataframe).rename(columns={product_id_column: agg_column_name})


def get_product_counts_per_category_and_time(dataframe: pd.DataFrame, time_column: str, year_month_column: str, product_id_column: str, coicop_level_column: str, agg_column_name: str = "count") -> pd.DataFrame:
    dataframe = split_month_year_column(dataframe, year_month_column)
    dataframe = dataframe.groupby(
        [time_column, coicop_level_column])[product_id_column].nunique()
    return pd.DataFrame(dataframe).rename(columns={product_id_column: agg_column_name})


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

    def retrieve_product_counts(self, dataframe: pd.DataFrame, coicop_level: str):
        product_counts_per_year_df = get_product_counts_per_time(
            dataframe, "year", "year_month", "product_id")
        product_counts_per_year_df.to_csv(os.path.join(
            self.data_directory, f"products_{self.supermarket_name}_{coicop_level}_product_counts_per_year.csv"))

        product_counts_per_year_month_df = get_product_counts_per_time(
            dataframe, "year_month", "year_month", "product_id")
        product_counts_per_year_month_df.to_csv(os.path.join(
            self.data_directory, f"products_{self.supermarket_name}_{coicop_level}_product_counts_per_year_month.csv"))

        product_counts_per_category_per_year_df = get_product_counts_per_category_and_time(
            dataframe, "year", "year_month", "product_id", coicop_level)
        product_counts_per_category_per_year_df.to_csv(os.path.join(
            self.data_directory, f"products_{self.supermarket_name}_{coicop_level}_product_counts_per_category_per_year.csv"))

    def perform_product_analysis_per_coicop_level(self, dataframe: pd.DataFrame, coicop_level: str, product_description_column: str = "ean_name"):
        coicop_level_values = dataframe[coicop_level].unique()

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
        for coicop_level in self.coicop_level_columns:
            self.perform_product_analysis_per_coicop_level(
                dataframe, coicop_level)

    def analyze_products(self, dataframe: pd.DataFrame):
        self.plot_sunburst(dataframe, amount_column="count")
        self.perform_product_level_analysis(dataframe)
