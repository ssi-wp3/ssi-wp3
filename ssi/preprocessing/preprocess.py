from .utils import ParquetFile
from .combine import CombineRevenueFiles
from .preprocess_data import preprocess_data
from .files import get_store_name_from_combined_filename
import pandas as pd
import luigi
import os


class PreprocessFile(luigi.Task):
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
    column_mapping = luigi.DictParameter(
        default={"bg_number": "store_id", "month": "year_month"})
    parquet_engine = luigi.Parameter(default="pyarrow")

    store_name = luigi.Parameter()
    combine_revenue_files = luigi.BoolParameter(default=False)

    def requires(self):
        if self.combine_revenue_files:
            return CombineRevenueFiles(output_filename=self.input_filename,
                                       store_name=self.store_name,
                                       parquet_engine=self.parquet_engine
                                       )
        return ParquetFile(input_filename=self.input_filename)

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
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    extension = luigi.Parameter(default=".parquet")

    def requires(self):
        return [PreprocessFile(
                input_filename=os.path.join(
                    self.input_directory, input_filename),
                output_filename=os.path.join(
                    self.output_directory, os.path.basename(input_filename)),
                store_name=get_store_name_from_combined_filename(
                    input_filename),
                )
                for input_filename in os.listdir(self.input_directory)
                if input_filename.endswith(self.extension)]
