from abc import ABCMeta, abstractproperty
from typing import Dict, Callable, Any
from .files import get_combined_revenue_files_in_directory
from .preprocessing import Preprocessing
from .overlap import calculate_overlap_for_stores, jaccard_index, jaccard_similarity, dice_coefficient, overlap_coefficient, percentage_overlap, asymmetrical_overlap, compare_overlap_between_preprocessing_functions, compare_overlap_per_coicop_label
from .products import *
from .revenue import *
from .text_analysis import string_length_histogram
from ..report import ReportEngine
from ..preprocessing.files import get_store_name_from_combined_filename
from ..constants import Constants
from ..plots import PlotEngine
from ..settings import Settings
import pandas as pd
import luigi
import os
import tqdm


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
            "texts_per_product_id_histogram": lambda dataframe: texts_per_ean_histogram(dataframe, self.receipt_text_column, self.product_id_column),
            "log_texts_per_product_id_histogram": lambda dataframe: log_texts_per_ean_histogram(dataframe, self.receipt_text_column, self.product_id_column),
            # Receipt and EAN lengths
            "receipt_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.receipt_text_column),
            "product_id_length_histogram": lambda dataframe: string_length_histogram(dataframe, self.product_id_column),

            # Revenue
            "total_revenue": lambda dataframe: total_revenue(dataframe, self.amount_column, self.revenue_column),
            "total_revenue_per_product": lambda dataframe: total_revenue_per_product(dataframe, self.product_id_column, self.amount_column, self.revenue_column),
            "total_revenue_per_receipt_text": lambda dataframe: total_revenue_per_product(dataframe, self.receipt_text_column, self.amount_column, self.revenue_column),

            "revenue_for_coicop_hierarchy": lambda dataframe: revenue_for_coicop_hierarchy(dataframe, list(self.coicop_columns), self.amount_column, self.revenue_column),
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
            "product_revenue_versus_lifetime": lambda dataframe: product_revenue_versus_lifetime(dataframe, self.period_column, self.product_id_column, self.amount_column, self.revenue_column),
            "product_availability_in_period": lambda dataframe: product_availability_in_period(dataframe, self.product_id_column, self.period_column),
            "product_lifetime_in_periods": lambda dataframe: product_lifetime_in_periods(dataframe, self.period_column, self.product_id_column),
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
    plot_settings_filename = luigi.Parameter(default=os.path.join(
        os.path.dirname(__file__), "report_settings.yml"))
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
        return [
            TableAnalysis(
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
            PeriodAnalysis(
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
            CoicopAnalysis(
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
            CoicopPeriodAnalysis(
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
        ]

    @property
    def plot_engine(self) -> PlotEngine:
        return PlotEngine()

    @property
    def report_settings(self) -> Dict[str, Any]:
        # TODO split up in plots that need to be run once for the whole dataset, plots that needt to be run for each
        # COICOP level, and plots that need to be run for each period.
        # TODO Add sunburst with number of products (EAN/Receipt texts) per coicop
        # TODO Add sunburst with total products sold/revenue?
        settings = Settings.load(self.plot_settings_filename,
                                 "report_settings",
                                 True,
                                 store_name=self.store_name,
                                 period_column=self.period_column,
                                 receipt_text_column=self.receipt_text_column,
                                 product_id_column=self.product_id_column,
                                 amount_column=self.amount_column,
                                 revenue_column=self.revenue_column,
                                 coicop_column=self.coicop_column,
                                 coicop_columns=list(self.coicop_columns)
                                 )

        return settings

    @property
    def report_engine(self) -> ReportEngine:
        return ReportEngine(self.report_settings)

    def target_for(self, filename: str, binary_file: bool = True) -> luigi.LocalTarget:
        if binary_file:
            return luigi.LocalTarget(filename, format=luigi.format.Nop)
        return luigi.LocalTarget(filename)

    def output(self):
        return {report.output_filename: self.target_for(os.path.join(self.output_directory, report.output_filename), binary_file=report.needs_binary_file)
                for report in self.report_engine.flattened_reports}

    def run(self):
        for task in self.input():
            for function_name, input in task.items():
                if function_name not in self.report_settings:
                    continue

                with input.open("r") as input_file:
                    dataframe = pd.read_parquet(
                        input_file, engine=self.parquet_engine)

                    reports = self.report_engine.reports[function_name]
                    for report in reports:
                        with self.output()[report.output_filename].open("w") as output_file:
                            report.write_to_file(dataframe, output_file)


class CrossStoreAnalysis(luigi.Task):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    project_prefix = luigi.Parameter(default="ssi")
    parquet_engine = luigi.Parameter(default="pyarrow")

    product_id_columns = luigi.ListParameter(
        default=[Constants.PRODUCT_ID_COLUMN, Constants.RECEIPT_TEXT_COLUMN])
    store_name_column = luigi.Parameter(default=Constants.STORE_NAME_COLUMN)

    @property
    def combined_revenue_files(self) -> Dict[str, str]:
        return {
            get_store_name_from_combined_filename(filename): filename
            for filename in get_combined_revenue_files_in_directory(self.input_directory, project_prefix=self.project_prefix)
        }

    @property
    def overlap_functions(self) -> Dict[str, Callable]:
        return {
            "jaccard_similarity": jaccard_similarity,
            "jaccard_index": jaccard_index,
            "dice_coefficient": dice_coefficient,
            "overlap_coefficient": overlap_coefficient,
            "percentage_overlap": percentage_overlap,
            "asymmetrical_overlap": asymmetrical_overlap,
        }

    @property
    def overlap_preprocessing_functions(self) -> Dict[str, Dict[str, Callable[[pd.Series], pd.Series]]]:
        return Preprocessing().all_preprocessing_functions

    def target_for(self,
                   overlap_directory: str,
                   product_id_column: str,
                   preprocessing_function_name: str,
                   overlap_name: str) -> luigi.LocalTarget:
        return luigi.LocalTarget(os.path.join(overlap_directory,
                                              f"overlap_{product_id_column}_{preprocessing_function_name}_{overlap_name}.parquet"),
                                 format=luigi.format.Nop
                                 )

    def get_output_key(self,
                       product_id_column: str,
                       preprocessing_function_name: str,
                       overlap_name: str) -> str:
        return f"{product_id_column}_{preprocessing_function_name}_{overlap_name}"

    def requires(self):
        return {store_name: StoreFile(filename)
                for store_name, filename in self.combined_revenue_files.items()}

    def output(self):
        overlap_directory = os.path.join(self.output_directory, "overlap")
        return {self.get_output_key(product_id_column, preprocessing_function, overlap_name):
                self.target_for(overlap_directory,
                                product_id_column,
                                preprocessing_function,
                                overlap_name)

                for product_id_column in self.product_id_columns
                for overlap_name in self.overlap_functions.keys()
                for preprocessing_function in self.overlap_preprocessing_functions[product_id_column].keys()
                }

    def run(self):
        print("Input", self.input())
        store_dataframes = [self.read_store_file(input_file, self.store_name_column)
                            for store_name, input_file in self.input().items()]
        # TODO calculations give different values then those from the notebook
        for product_id_column in self.product_id_columns:
            for overlap_function_name, overlap_function in self.overlap_functions.items():
                for preprocessing_function_name, preprocessing_function in self.overlap_preprocessing_functions[product_id_column].items():
                    with tqdm.tqdm(total=len(store_dataframes) * (len(store_dataframes) - 1) // 2) as progress_bar:
                        overlap_matrix_df = calculate_overlap_for_stores(
                            store_dataframes,
                            store_id_column=self.store_name_column,
                            product_id_column=product_id_column,
                            overlap_function=overlap_function,
                            preprocess_function=preprocessing_function,
                            progress_bar=progress_bar,
                            calculate_all_cells=True if overlap_function_name == "asymmetrical_overlap" else False)

                        output_key = self.get_output_key(
                            product_id_column, preprocessing_function_name, overlap_function_name)
                        with self.output()[output_key].open("w") as output_file:
                            overlap_matrix_df.to_parquet(
                                output_file, engine=self.parquet_engine)

    def read_store_file(self, input_file, store_name_column: str) -> pd.DataFrame:
        with input_file.open("r") as input_parquet_file:
            dataframe = pd.read_parquet(
                input_parquet_file, engine=self.parquet_engine, columns=list(self.product_id_columns) + [self.store_name_column])
            dataframe[store_name_column] = dataframe[store_name_column].str.replace(
                "ah_franchise", "ah")
            return dataframe
            # return self.__add_store_name_column(dataframe, store_name, store_name_column)

    def __add_store_name_column(self,
                                store_dataframe: pd.DataFrame,
                                store_name: str,
                                store_name_column: str = "store_name") -> pd.DataFrame:
        store_dataframe[store_name_column] = store_name
        return store_dataframe


class OverlapPerPreprocessing(luigi.Task):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    project_prefix = luigi.Parameter(default="ssi")
    parquet_engine = luigi.Parameter(default="pyarrow")

    product_id_column = luigi.Parameter(
        default=Constants.RECEIPT_TEXT_COLUMN)
    store_name_column = luigi.Parameter(default=Constants.STORE_NAME_COLUMN)

    @property
    def combined_revenue_files(self) -> Dict[str, str]:
        return {
            get_store_name_from_combined_filename(filename): filename
            for filename in get_combined_revenue_files_in_directory(self.input_directory, project_prefix=self.project_prefix)
        }

    @property
    def preprocessing_functions(self) -> Dict[str, Dict[str, Callable[[pd.Series], pd.Series]]]:
        return Preprocessing().receipt_text_preprocessing_functions

    def requires(self):
        return {store_name: StoreFile(filename)
                for store_name, filename in self.combined_revenue_files.items()}

    def output(self):
        overlap_directory = os.path.join(self.output_directory, "overlap")
        return luigi.LocalTarget(os.path.join(overlap_directory, "mean_overlap_per_preprocessing_function.parquet"),
                                 format=luigi.format.Nop
                                 )

    def run(self):
        with tqdm.tqdm(total=len(self.preprocessing_functions)) as progress_bar:
            store_dataframes = [self.read_store_file(input_file, self.store_name_column)
                                for store_name, input_file in self.input().items()]
            dataframe = compare_overlap_between_preprocessing_functions(store_dataframes,
                                                                        self.store_name_column,
                                                                        self.product_id_column,
                                                                        self.preprocessing_functions,
                                                                        progress_bar=progress_bar)
            with self.output().open("w") as output_file:
                dataframe.to_parquet(
                    output_file, engine=self.parquet_engine)

    def read_store_file(self, input_file, store_name_column: str) -> pd.DataFrame:
        with input_file.open("r") as input_parquet_file:
            dataframe = pd.read_parquet(
                input_parquet_file, engine=self.parquet_engine, columns=[self.product_id_column, self.store_name_column])
            dataframe[store_name_column] = dataframe[store_name_column].str.replace(
                "ah_franchise", "ah")
            return dataframe

    def __add_store_name_column(self,
                                store_dataframe: pd.DataFrame,
                                store_name: str,
                                store_name_column: str = "store_name") -> pd.DataFrame:
        store_dataframe[store_name_column] = store_name
        return store_dataframe


class OverlapPerPreprocessingAndCoicop(luigi.Task):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    project_prefix = luigi.Parameter(default="ssi")
    parquet_engine = luigi.Parameter(default="pyarrow")

    product_id_column = luigi.Parameter(
        default=Constants.RECEIPT_TEXT_COLUMN)
    store_name_column = luigi.Parameter(default=Constants.STORE_NAME_COLUMN)
    coicop_column = luigi.Parameter(default="coicop_level_1")

    @property
    def combined_revenue_files(self) -> Dict[str, str]:
        return {
            get_store_name_from_combined_filename(filename): filename
            for filename in get_combined_revenue_files_in_directory(self.input_directory, project_prefix=self.project_prefix)
        }

    @property
    def preprocessing_functions(self) -> Dict[str, Dict[str, Callable[[pd.Series], pd.Series]]]:
        return Preprocessing().receipt_text_preprocessing_functions

    def requires(self):
        return {store_name: StoreFile(filename)
                for store_name, filename in self.combined_revenue_files.items()}

    def output(self):
        overlap_directory = os.path.join(self.output_directory, "overlap")
        return {preprocessing: luigi.LocalTarget(os.path.join(overlap_directory,
                                                              f"overlap_{preprocessing}_function_and_{self.coicop_column}.parquet"),
                                                 format=luigi.format.Nop
                                                 )
                for preprocessing in self.preprocessing_functions.keys()}

    def run(self):
        with tqdm.tqdm(total=len(self.preprocessing_functions)) as progress_bar:
            store_dataframes = [self.read_store_file(input_file, self.store_name_column)
                                for store_name, input_file in self.input().items()]

            for preprocessing_function_name, preprocessing_function in self.preprocessing_functions.items():
                dataframe = compare_overlap_per_coicop_label(store_dataframes,
                                                             self.store_name_column,
                                                             self.product_id_column,
                                                             self.coicop_column,
                                                             preprocessing_function,
                                                             progress_bar=progress_bar)
                progress_bar.update(1)
                with self.output()[preprocessing_function_name].open("w") as output_file:
                    dataframe.to_parquet(
                        output_file, engine=self.parquet_engine)

    def read_store_file(self, input_file, store_name_column: str) -> pd.DataFrame:
        with input_file.open("r") as input_parquet_file:
            dataframe = pd.read_parquet(
                input_parquet_file, engine=self.parquet_engine, columns=[self.product_id_column, self.store_name_column, self.coicop_column])
            dataframe[store_name_column] = dataframe[store_name_column].str.replace(
                "ah_franchise", "ah")
            return dataframe

    def __add_store_name_column(self,
                                store_dataframe: pd.DataFrame,
                                store_name: str,
                                store_name_column: str = "store_name") -> pd.DataFrame:
        store_dataframe[store_name_column] = store_name
        return store_dataframe


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

                    yield PerStoreAnalysis(
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
