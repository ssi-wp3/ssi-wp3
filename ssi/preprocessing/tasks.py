

from .files import get_revenue_files_in_folder
from .preprocess_data import preprocess_data, convert_ah_receipts
from .convert import ConvertCSVToParquet
import pandas as pd
import luigi
import os


class ConvertAHReceipts(luigi.Task):
    """ Convert an AH receipts file in Excel format to a parquet file.

    """

    input_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()
    coicop_sheets_prefix = luigi.Parameter(default="coi")

    def run(self):
        with self.input().open('r') as input_file:
            with self.output().open('w') as output_file:
                ah_receipts_df = convert_ah_receipts(
                    input_file, self.coicop_sheet_prefix)
                ah_receipts_df.to_parquet(output_file, engine="pyarrow")

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)


class CombineRevenueFiles(luigi.Task):
    """ This task combines revenue files that were prepared by ConvertCSVToParquet into a single parquet file.

    Parameters
    ----------
    input_directory : luigi.Parameter
        The directory where the revenue files are stored.

    output_filename : luigi.Parameter
        The output filename for the combined revenue file.

    """
    input_directory = luigi.Parameter()
    output_filename = luigi.Parameter()
    store_name = luigi.Parameter()
    filename_prefix = luigi.Parameter()
    parquet_engine = luigi.Parameter()
    sort_order = luigi.DictParameter(
        default={"bg_number": True, "month": True, "coicop_number": True})

    def requires(self):
        revenue_files = get_revenue_files_in_folder(
            self.input_directory,
            self.store_name,
            self.filename_prefix,
            ".csv")
        return [ConvertCSVToParquet(input_filename)
                for input_filename in revenue_files
                ]

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def run(self):
        combined_dataframe = None
        for input in self.input():
            with input.open("r") as input_file:
                revenue_file = pd.read_parquet(
                    input_file, engine=self.parquet_engine)
                if combined_dataframe is None:
                    combined_dataframe = revenue_file
                else:
                    combined_dataframe = pd.concat(
                        [combined_dataframe, revenue_file])

        combined_dataframe = combined_dataframe.sort_values(
            by=list(self.sort_order.keys()), ascending=list(self.sort_order.values())).reset_index(drop=True)

        with self.output().open('w') as output_file:
            combined_dataframe.to_parquet(
                output_file, engine=self.parquet_engine)


class PreprocessCombinedFile(luigi.Task):
    """ This task preprocesses the combined revenue file.

    Parameters
    ----------
    input_filename : luigi.Parameter
        The input filename of the combined revenue file.

    output_filename : luigi.Parameter
        The output filename for the preprocessed combined revenue file.

    coicop_column : luigi.Parameter
        The name of the column containing the coicop numbers.

    product_id_column : luigi.Parameter
        The name of the column containing the product ids.

    product_description_column : luigi.Parameter
        The name of the column containing the product descriptions.

    selected_columns : luigi.ListParameter
        Names of the columns to select.

    coicop_level_columns : luigi.ListParameter
        Names of the columns containing the coicop levels.

    column_mapping : luigi.DictParameter
        A dictionary containing the mapping of column names.

    """
    input_filename = luigi.Parameter()
    output_filename = luigi.Parameter()
    coicop_column = luigi.Parameter(default="coicop_number")
    product_id_column = luigi.Parameter(default="product_id")
    product_description_column = luigi.Parameter(default="ean_name")
    selected_columns = luigi.ListParameter(default=[])
    coicop_level_columns = luigi.ListParameter(default=[])
    column_mapping = luigi.DictParameter(default={})
    parquet_engine = luigi.Parameter(default="pyarrow")

    store_name = luigi.Parameter()

    def requires(self):
        return CombineRevenueFiles(output_filename=self.input_filename,
                                   store_name=self.store_name,
                                   parquet_engine=self.parquet_engine
                                   )

    def run(self):
        with self.input().open("r") as input_file:
            combined_df = pd.read_parquet(
                input_file, engine=self.parquet_engine)

        combined_df = preprocess_data(
            combined_df,
            columns=self.selected_columns,
            coicop_column=self.coicop_column,
            product_id_column=self.product_id_column,
            product_description_column=self.product_description_column,
            column_mapping=self.column_mapping
        )

        with self.output().open('w') as output_file:
            combined_df.to_parquet(output_file, engine=self.parquet_engine)

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)


class PreprocessAllFiles(luigi.WrapperTask):
    output_directory = luigi.PathParameter()
    stores = luigi.ListParameter()

    def requires(self):
        return [PreprocessCombinedFile(
                input_filename=f"combined_{store}.parquet",
                output_filename=os.path.join(
                    self.output_directory, f"preprocessed_{store}.parquet"),
                store_name=store
                )
                for store in self.stores]
