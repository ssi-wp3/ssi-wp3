from typing import Dict, Callable, Any
from .utils import unpivot
from .files import get_combined_revenue_files_in_directory
from .products import *
from .text_analysis import string_length_histogram
from ..preprocessing.files import get_store_name_from_combined_filename
from ..plots import PlotEngine
import pandas as pd
import luigi
import os


class StoreFile(luigi.ExternalTask):
    input_filename = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename, format=luigi.format.Nop)


class StoreProductAnalysis(luigi.Task):
    """ This task analyzes the store product inventory and dynamics.

    """
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    parquet_engine = luigi.Parameter(default="pyarrow")

    store_name = luigi.Parameter()
    period_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    coicop_column = luigi.Parameter()

    @property
    def product_analysis_functions(self) -> Dict[str, Callable]:
        value_columns = [self.product_id_column, self.receipt_text_column]
        return {
            "unique_column_values": lambda dataframe: unique_column_values(dataframe, value_columns),
            "unique_coicop_values_per_coicop": lambda dataframe: unique_column_values_per_coicop(dataframe, self.coicop_column, value_columns),
            "unique_column_values_per_period": lambda dataframe: unique_column_values_per_period(dataframe, self.period_column, value_columns),
            "texts_per_ean_histogram": lambda dataframe: texts_per_ean_histogram(dataframe, self.receipt_text_column, self.product_id_column),
            "log_texts_per_ean_histogram": lambda dataframe: log_texts_per_ean_histogram(dataframe, self.receipt_text_column, self.product_id_column),
            "compare_products_per_period": lambda dataframe: compare_products_per_period(dataframe, self.period_column, value_columns),
            "compare_products_per_period_coicop_level": lambda dataframe: compare_products_per_period_coicop_level(dataframe, self.period_column, self.coicop_column, value_columns),

            "receipt_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.receipt_text_column),
            "ean_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.product_id_column),
        }

    def requires(self):
        return StoreFile(self.input_filename)

    def output(self):
        return {function_name:
                luigi.LocalTarget(os.path.join(
                    self.output_directory, self._get_store_analysis_filename(function_name)), format=luigi.format.Nop)
                for function_name in self.product_analysis_functions.keys()}

    def _get_store_analysis_filename(self, function_name: str):
        return f"{self.store_name}_{self.period_column}_{self.coicop_column}_analysis_{function_name}.parquet"

    def run(self):
        with self.input().open("r") as input_file:
            dataframe = pd.read_parquet(
                input_file, engine=self.parquet_engine)
            for function_name, function in self.product_analysis_functions.items():
                print(
                    f"Running {function_name} for {self.store_name}, period column: {self.period_column}, coicop column: {self.coicop_column}")
                result_df = function(dataframe)
                with self.output()[function_name].open("w") as output_file:
                    result_df.to_parquet(
                        output_file, engine=self.parquet_engine)


class PlotResults(luigi.Task):
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    parquet_engine = luigi.Parameter(default="pyarrow")

    store_name = luigi.Parameter()
    period_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    coicop_column = luigi.Parameter()

    def requires(self):
        return StoreProductAnalysis(
            input_filename=self.input_filename,
            output_directory=self.output_directory,
            parquet_engine=self.parquet_engine,
            store_name=self.store_name,
            period_column=self.period_column,
            receipt_text_column=self.receipt_text_column,
            product_id_column=self.product_id_column,
            coicop_column=self.coicop_column
        )

    @property
    def plot_engine(self) -> PlotEngine:
        return PlotEngine()

    @property
    def plot_settings(self) -> Dict[str, Any]:
        return {
            # "unique_column_values": lambda file, dataframe: dataframe.to_latex(file),
            "unique_coicop_values_per_coicop": {
                "pivot": True,
                "filename": f"{self.store_name}_{self.period_column}_{self.coicop_column}_unique_receipt_values_per_coicop.png",
                "plot_settings": {
                    "plot_type": "bar_chart",
                    "x_column": self.coicop_column,
                    "y_column": "value",
                    "group_column": "group",
                    "title": f"Unique receipt texts per {self.coicop_column} for {self.store_name}"
                },
            },
            "unique_column_values_per_period": {
                "pivot": True,
                "filename": f"{self.store_name}_{self.period_column}_{self.coicop_column}_unique_receipt_values_per_{self.period_column}.png",
                "plot_settings": {
                    "plot_type": "line_chart",
                    "x_column": self.period_column,
                    "y_column": "value",
                    "group_column": "group",
                    "title": f"Unique receipt texts per {self.period_column} for {self.store_name}"
                },
            },
            "texts_per_ean_histogram": {
                "pivot": False,
                "filename": f"{self.store_name}_{self.period_column}_{self.coicop_column}_texts_per_ean_histogram.png",
                "plot_settings": {
                    "plot_type": "bar_chart",
                    "x_column": "receipt_text",
                    "y_column": "count",
                    "title": f"Unique receipt texts per EAN for {self.store_name}"
                },
            },
            "log_texts_per_ean_histogram": {
                "pivot": False,
                "filename": f"{self.store_name}_{self.period_column}_{self.coicop_column}_log_texts_per_ean_histogram.png",
                "plot_settings": {
                    "plot_type": "bar_chart",
                    "x_column": "receipt_text",
                    "y_column": "count",
                    "title": f"Unique receipt texts per EAN for {self.store_name}"
                },
            },
            # "compare_products_per_period_coicop_level": lambda dataframe: compare_products_per_period_coicop_level(dataframe, self.period_column, self.coicop_column, value_columns),

            # "receipt_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.receipt_text_column),
            # "ean_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.product_id_column),
        }

    def output(self):
        output_dict = dict()
        for function_name in self.input().keys():
            if function_name in self.plot_settings:
                plot_settings = self.plot_settings[function_name]
                if isinstance(plot_settings, list):
                    for settings in plot_settings:
                        output_dict[function_name] = luigi.LocalTarget(os.path.join(
                            self.output_directory, settings["filename"]), format=luigi.format.Nop)
                else:
                    output_dict[function_name] = luigi.LocalTarget(os.path.join(
                        self.output_directory, plot_settings["filename"]), format=luigi.format.Nop)

        return output_dict

    def run(self):
        value_columns = [self.product_id_column, self.receipt_text_column]
        for function_name, input in self.input().items():
            if function_name not in self.plot_settings:
                continue

            with input.open("r") as input_file:
                dataframe = pd.read_parquet(
                    input_file, engine=self.parquet_engine)

                if function_name not in self.plot_settings:
                    continue
                plot_settings = self.plot_settings[function_name]
                if isinstance(plot_settings, list):
                    for settings in plot_settings:
                        self.plot_with_settings(
                            function_name, dataframe, settings, value_columns)
                else:
                    self.plot_with_settings(
                        function_name, dataframe, plot_settings, value_columns)

    def plot_with_settings(self, function_name: str, dataframe: pd.DataFrame, plot_settings: Dict[str, Any], value_columns: List[str] = None):
        if "pivot" in plot_settings and plot_settings["pivot"]:
            dataframe = unpivot(dataframe, value_columns)

        with self.output()[function_name].open("w") as output_file:
            figure = self.plot_engine.plot_settings(
                dataframe, plot_settings["plot_settings"])
            figure.save(output_file)


class AllStoresAnalysis(luigi.WrapperTask):
    """ This task analyses the product inventory and dynamics for all stores in a certain
    directory.

    Parameters
    ----------

    input_directory : luigi.PathParameter
        The directory containing the store product inventory files.

    output_directory : luigi.PathParameter
        The directory to store the analysis results.

    project_prefix : luigi.Parameter
        The project prefix used in the revenue file names.
    """
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    project_prefix = luigi.Parameter(default="ssi")

    parquet_engine = luigi.Parameter(default="pyarrow")
    period_columns = luigi.ListParameter()
    receipt_text_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    coicop_columns = luigi.ListParameter()

    def requires(self):
        for filename in get_combined_revenue_files_in_directory(self.input_directory, project_prefix=self.project_prefix):
            for period_column in self.period_columns:
                for coicop_column in self.coicop_columns:
                    store_name = get_store_name_from_combined_filename(
                        filename)
                    output_directory = os.path.join(
                        self.output_directory, "products")
                    output_directory = os.path.join(
                        output_directory, store_name)
                    output_directory = os.path.join(
                        output_directory, coicop_column)
                    os.makedirs(output_directory, exist_ok=True)

                    yield PlotResults(
                        input_filename=filename,
                        output_directory=output_directory,
                        parquet_engine=self.parquet_engine,
                        store_name=store_name,
                        period_column=period_column,
                        receipt_text_column=self.receipt_text_column,
                        product_id_column=self.product_id_column,
                        coicop_column=coicop_column
                    )
