from ..parquet_file import ParquetFile
from .combine_unique_values import drop_duplicates_and_empty_receipts
from .files import get_store_name_from_combined_filename, get_combined_revenue_files_in_folder
from .batch_copy import batch_copy
import pandas as pd
import luigi
import os


class DropDuplicatesTask(luigi.Task):
    input_filename = luigi.Parameter()
    output_filename = luigi.Parameter()
    key_columns = luigi.ListParameter(
        default=['receipt_text', 'coicop_number'])
    receipt_text_column = luigi.Parameter(default='receipt_text')
    drop_empty_receipt_texts = luigi.BoolParameter(default=True)
    batch_size = luigi.IntParameter(default=1024)
    engine = luigi.Parameter(default='pyarrow')

    def requires(self):
        return ParquetFile(self.input_filename)

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def run(self):
        with self.input().open('r') as input_file:
            dataframe = pd.read_parquet(
                input_file, columns=self.key_columns, engine=self.engine)
            # Drop duplicates and empty receipts and get the row indices that we need to write later
            dataframe = drop_duplicates_and_empty_receipts(dataframe,
                                                           self.key_columns,
                                                           self.receipt_text_column,
                                                           self.drop_empty_receipt_texts)

            with self.output().open('w') as output_file:
                batch_copy(input_filename=self.input_filename, row_indices_to_copy=dataframe,
                           output_filename=output_file, batch_size=self.batch_size)


class DropDuplicatesForAllFiles(luigi.WrapperTask):
    input_directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    filename_prefix = luigi.Parameter()
    key_columns = luigi.ListParameter(
        default=['receipt_text', 'coicop_number'])
    receipt_text_column = luigi.Parameter(default='receipt_text')
    drop_empty_receipt_texts = luigi.BoolParameter(default=True)
    engine = luigi.Parameter(default='pyarrow')

    def requires(self):
        for input_filename in get_combined_revenue_files_in_folder(
                self.input_directory, self.filename_prefix):
            store_name = get_store_name_from_combined_filename(input_filename)
            yield DropDuplicatesTask(
                input_filename=input_filename,
                output_filename=os.path.join(
                    self.output_directory, f"{self.filename_prefix}_{store_name}_unique_values.parquet"),
                key_columns=self.key_columns,
                receipt_text_column=self.receipt_text_column,
                drop_empty_receipt_texts=self.drop_empty_receipt_texts,
                engine=self.engine
            )
