from abc import ABCMeta, abstractproperty
from typing import Dict, Callable, Any
from .utils import unpivot
from .files import get_combined_revenue_files_in_directory
from .products import *
from .revenue import *
from .text_analysis import string_length_histogram
from ..preprocessing.files import get_store_name_from_combined_filename
from ..constants import Constants
from ..plots import PlotEngine
import pandas as pd
import luigi
import os


class StoreFile(luigi.ExternalTask):
    input_filename = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename, format=luigi.format.Nop)


class BaseStoreAnalysisTask(luigi.Task, metaclass=ABCMeta):
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    parquet_engine = luigi.Parameter(default="pyarrow")

    store_name = luigi.Parameter()

    @abstractproperty
    def analysis_functions(self) -> Dict[str, Callable]:
        pass

    def get_store_analysis_filename(self, function_name: str) -> str:
        return f"{self.store_name}_analysis_{function_name}.parquet"

    def log_analysis_status(self, function_name):
        print(
            f"Running {function_name} for {self.store_name}")

    def requires(self):
        return StoreFile(self.input_filename)

    def output(self):
        return {function_name:
                luigi.LocalTarget(os.path.join(
                    self.output_directory, self.get_store_analysis_filename(function_name)), format=luigi.format.Nop)
                for function_name in self.analysis_functions.keys()}

    def run(self):
        with self.input().open("r") as input_file:
            dataframe = pd.read_parquet(
                input_file, engine=self.parquet_engine)
            for function_name, function in self.analysis_functions.items():
                self.log_analysis_status(function_name)
                result_df = function(dataframe)
                with self.output()[function_name].open("w") as output_file:
                    result_df.to_parquet(
                        output_file, engine=self.parquet_engine)


class TableAnalysis(BaseStoreAnalysisTask):
    """ This task analyzes the store product inventory and dynamics.

    """
    receipt_text_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    amount_column = luigi.Parameter()
    revenue_column = luigi.Parameter()
    coicop_columns = luigi.ListParameter(
        default=Constants.COICOP_LEVELS_COLUMNS)

    @property
    def analysis_functions(self) -> Dict[str, Callable]:
        value_columns = [self.product_id_column, self.receipt_text_column]
        return {
            # Whole table
            "unique_column_values": lambda dataframe: unique_column_values(dataframe, value_columns),
            "texts_per_ean_histogram": lambda dataframe: texts_per_ean_histogram(dataframe, self.receipt_text_column, self.product_id_column),
            "log_texts_per_ean_histogram": lambda dataframe: log_texts_per_ean_histogram(dataframe, self.receipt_text_column, self.product_id_column),
            # Receipt and EAN lengths
            "receipt_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.receipt_text_column),
            "ean_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.product_id_column),

            # Revenue
            "total_revenue": lambda dataframe: total_revenue(dataframe, self.amount_column, self.revenue_column),
            "total_revenue_per_product": lambda dataframe: total_revenue_per_product(dataframe, self.product_id_column, self.amount_column, self.revenue_column),
            "total_revenue_per_receipt_text": lambda dataframe: total_revenue_per_product(dataframe, self.receipt_text_column, self.amount_column, self.revenue_column),

            "revenue_for_coicop_hierarchy": lambda dataframe: revenue_for_coicop_hierarchy(dataframe, self.coicop_columns, self.amount_column, self.revenue_column),
        }

    def get_store_analysis_filename(self, function_name: str) -> str:
        return f"{self.store_name}_analysis_{function_name}.parquet"

    def log_analysis_status(self, function_name):
        print(
            f"Running {function_name} for {self.store_name}")


class PeriodAnalysis(BaseStoreAnalysisTask):
    """ This task analyzes the revenue data for a certain supermarket.
    """
    period_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    revenue_column = luigi.Parameter()
    amount_column = luigi.Parameter()

    @property
    def analysis_functions(self) -> Dict[str, Callable]:
        value_columns = [self.product_id_column, self.receipt_text_column]
        return {
            # Per period
            "compare_products_per_period": lambda dataframe: compare_products_per_period(dataframe, self.period_column, value_columns),
            "total_revenue_per_period": lambda dataframe: total_revenue_per_period(dataframe, self.period_column, self.amount_column, self.revenue_column),
            "unique_column_values_per_period": lambda dataframe: unique_column_values_per_period(dataframe, self.period_column, value_columns),
        }

    def get_store_analysis_filename(self, function_name: str) -> str:
        return f"{self.store_name}_{self.period_column}_analysis_{function_name}.parquet"

    def log_analysis_status(self, function_name):
        print(
            f"Running {function_name} for {self.store_name}, period column: {self.period_column}")


class CoicopAnalysis(BaseStoreAnalysisTask):
    coicop_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    amount_column = luigi.Parameter()
    revenue_column = luigi.Parameter()

    @property
    def analysis_functions(self) -> Dict[str, Callable]:
        value_columns = [self.product_id_column, self.receipt_text_column]
        return {
            # Grouped per coicop
            "unique_coicop_values_per_coicop": lambda dataframe: unique_column_values_per_coicop(dataframe, self.coicop_column, value_columns),

            # Revenue
            "total_revenue_per_coicop": lambda dataframe: total_revenue_per_coicop(dataframe, self.coicop_column, self.amount_column, self.revenue_column),
            "product_revenue_versus_lifetime": lambda dataframe: product_revenue_versus_lifetime(dataframe, self.coicop_column, self.product_id_column, self.amount_column, self.revenue_column),
        }

    def get_store_analysis_filename(self, function_name: str) -> str:
        return f"{self.store_name}_{self.coicop_column}_analysis_{function_name}.parquet"

    def log_analysis_status(self, function_name):
        print(
            f"Running {function_name} for {self.store_name}, coicop column: {self.coicop_column}")


class CoicopPeriodAnalysis(BaseStoreAnalysisTask):
    """ This task analyzes the revenue data for a certain supermarket.
    """
    period_column = luigi.Parameter()
    coicop_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    revenue_column = luigi.Parameter()
    amount_column = luigi.Parameter()

    @property
    def analysis_functions(self) -> Dict[str, Callable]:
        value_columns = [self.product_id_column, self.receipt_text_column]
        return {
            "total_revenue_per_coicop_and_period": lambda dataframe: total_revenue_per_coicop_and_period(dataframe, self.period_column, self.coicop_column, self.amount_column, self.revenue_column),
            "compare_products_per_period_coicop_level": lambda dataframe: compare_products_per_period_coicop_level(dataframe, self.period_column, self.coicop_column, value_columns),
        }

    def get_store_analysis_filename(self, function_name: str) -> str:
        return f"{self.store_name}_{self.period_column}_{self.coicop_column}_analysis_{function_name}.parquet"

    def log_analysis_status(self, function_name):
        print(
            f"Running {function_name} for {self.store_name}, period column: {self.period_column}, coicop column: {self.coicop_column}")


class PerStoreAnalysis(luigi.Task):
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    parquet_engine = luigi.Parameter(default="pyarrow")

    store_name = luigi.Parameter()
    period_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    amount_column = luigi.Parameter()
    revenue_column = luigi.Parameter()
    coicop_columns = luigi.ListParameter()
    coicop_column = luigi.Parameter()

    @property
    def directories(self) -> Dict[str, str]:
        store_directory = os.path.join(self.output_directory, self.store_name)
        return {
            "table": os.path.join(store_directory, "table"),
            "period": os.path.join(store_directory, "period"),
            "coicop": os.path.join(store_directory, "coicop"),
            "period_coicop": os.path.join(store_directory, "period_coicops")
        }

    def requires(self):
        return {
            "table": TableAnalysis(
                input_filename=self.input_filename,
                output_directory=self.directories["table"],
                parquet_engine=self.parquet_engine,
                store_name=self.store_name,
                receipt_text_column=self.receipt_text_column,
                product_id_column=self.product_id_column,
                amount_column=self.amount_column,
                revenue_column=self.revenue_column,
                coicop_columns=self.coicop_columns
            ),
            "period": PeriodAnalysis(
                input_filename=self.input_filename,
                output_directory=self.directories["period"],
                parquet_engine=self.parquet_engine,
                store_name=self.store_name,
                period_column=self.period_column,
                receipt_text_column=self.receipt_text_column,
                product_id_column=self.product_id_column,
                amount_column=self.amount_column,
                revenue_column=self.revenue_column
            ),
            "coicop": CoicopAnalysis(
                input_filename=self.input_filename,
                output_directory=self.directories["coicop"],
                parquet_engine=self.parquet_engine,
                store_name=self.store_name,
                coicop_column=self.coicop_column,
                product_id_column=self.product_id_column,
                receipt_text_column=self.receipt_text_column,
                amount_column=self.amount_column,
                revenue_column=self.revenue_column
            ),
            "period_coicop": CoicopPeriodAnalysis(
                input_filename=self.input_filename,
                output_directory=self.directories["period_coicop"],
                parquet_engine=self.parquet_engine,
                store_name=self.store_name,
                period_column=self.period_column,
                coicop_column=self.coicop_column,
                product_id_column=self.product_id_column,
                receipt_text_column=self.receipt_text_column,
                amount_column=self.amount_column,
                revenue_column=self.revenue_column
            )
        }


class CrossStoreAnalysis(luigi.Task):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    project_prefix = luigi.Parameter(default="ssi")
    parquet_engine = luigi.Parameter(default="pyarrow")


class PlotResults(luigi.Task):
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    parquet_engine = luigi.Parameter(default="pyarrow")

    store_name = luigi.Parameter()
    period_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    product_id_column = luigi.Parameter()
    amount_column = luigi.Parameter()
    revenue_column = luigi.Parameter()
    coicop_columns = luigi.ListParameter()
    coicop_column = luigi.Parameter()

    def requires(self):
        return PerStoreAnalysis(
            input_filename=self.input_filename,
            output_directory=self.output_directory,
            parquet_engine=self.parquet_engine,
            store_name=self.store_name,
            period_column=self.period_column,
            receipt_text_column=self.receipt_text_column,
            product_id_column=self.product_id_column,
            amount_column=self.amount_column,
            revenue_column=self.revenue_column,
            coicop_columns=self.coicop_columns,
            coicop_column=self.coicop_column
        )

    @property
    def plot_engine(self) -> PlotEngine:
        return PlotEngine()

    @property
    def plot_settings(self) -> Dict[str, Any]:
        # TODO split up in plots that need to be run once for the whole dataset, plots that needt to be run for each
        # COICOP level, and plots that need to be run for each period.
        # TODO Add sunburst with number of products (EAN/Receipt texts) per coicop
        # TODO Add sunburst with total products sold/revenue?
        return {
            # "unique_column_values": lambda file, dataframe: dataframe.to_latex(file),
            "unique_coicop_values_per_coicop": {
                "pivot": True,
                "value_columns": [self.product_id_column, self.receipt_text_column],
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
                "value_columns": [self.product_id_column, self.receipt_text_column],
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
                    "title": f"Unique receipt texts per EAN for {self.store_name} (Log scale)"
                },
            },
            "receipt_length_histogram": {
                "pivot": False,
                "filename": f"{self.store_name}_{self.period_column}_{self.coicop_column}_receipt_length_histogram.png",
                "plot_settings": {
                    "plot_type": "bar_chart",
                    "x_column": self.receipt_text_column,
                    "y_column": "count",
                    "title": f"Receipt text length histogram for {self.store_name}"
                },
            },
            "ean_length_histogram": {
                "pivot": False,
                "filename": f"{self.store_name}_{self.period_column}_{self.coicop_column}_ean_length_histogram.png",
                "plot_settings": {
                    "plot_type": "bar_chart",
                    "x_column": self.product_id_column,
                    "y_column": "count",
                    "title": f"EAN length histogram for {self.store_name}"
                },
            },
            "compare_products_per_period": {
                "pivot": True,
                "value_columns": [
                    f"number_{self.receipt_text_column}_introduced",
                    f"number_{self.receipt_text_column}_removed",
                    f"number_{self.product_id_column}_introduced",
                    f"number_{self.product_id_column}_removed",
                ],
                "filename": f"{self.store_name}_{self.period_column}_{self.coicop_column}_compare_products_per_period.png",
                "plot_settings": {
                    "plot_type": "line_chart",
                    "x_column": self.period_column,
                    "y_column": "value",
                    "group_column": "group",
                    "title": f"Number of products introduced/removed per {self.period_column} for {self.store_name}"
                },
            },
            # "compare_products_per_period_coicop_level":{


        }

    def output(self):
        print(f"Inputs: {self.input()}")
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
        print(f"Inputs: {self.input()}")
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
                            function_name, dataframe, settings)
                else:
                    self.plot_with_settings(
                        function_name, dataframe, plot_settings)

    def plot_with_settings(self, function_name: str, dataframe: pd.DataFrame, plot_settings: Dict[str, Any], value_columns: List[str] = None):
        if "pivot" in plot_settings and plot_settings["pivot"]:
            value_columns = plot_settings["value_columns"]
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
    amount_column = luigi.Parameter()
    revenue_column = luigi.Parameter()
    coicop_columns = luigi.ListParameter()

    def requires(self):
        for filename in get_combined_revenue_files_in_directory(self.input_directory, project_prefix=self.project_prefix):
            for period_column in self.period_columns:
                for coicop_column in self.coicop_columns:
                    store_name = get_store_name_from_combined_filename(
                        filename)
                    output_directory = os.path.join(
                        self.output_directory, "products")
                    os.makedirs(output_directory, exist_ok=True)

                    yield PlotResults(
                        input_filename=filename,
                        output_directory=output_directory,
                        parquet_engine=self.parquet_engine,
                        store_name=store_name,
                        period_column=period_column,
                        receipt_text_column=self.receipt_text_column,
                        product_id_column=self.product_id_column,
                        amount_column=self.amount_column,
                        revenue_column=self.revenue_column,
                        coicop_columns=self.coicop_columns,
                        coicop_column=coicop_column
                    )
