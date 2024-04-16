from ..parquet_file import ParquetFile
from .parquet import convert_to_parquet
from .preprocess_data import convert_ah_receipts, convert_jumbo_receipts
from .clean import CleanCPIFile
import pandas as pd
import luigi
import os


class CsvFile(luigi.ExternalTask):
    input_filename = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename)


class ExcelFile(luigi.ExternalTask):
    input_filename = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename, format=luigi.format.Nop)


class ConvertAHReceipts(luigi.Task):
    """ Convert an AH receipts file in Excel format to a parquet file.

    """
    input_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()
    coicop_sheets_prefix = luigi.Parameter(default="coi")
    add_start_date = luigi.BoolParameter(default=True)

    def requires(self):
        return ExcelFile(input_filename=self.input_filename)

    def run(self):
        with self.input().open('r') as input_file:
            with self.output().open('w') as output_file:
                ah_receipts_df = convert_ah_receipts(
                    input_file, self.coicop_sheets_prefix)
                # The Plus receipt texts we got in February 2024.
                # When couple to the revenue files we saw that the first month that the EANs were available
                # was in January 2023. Since, we do not have a date column as with Plus, we add this column
                # artificially here.
                if self.add_start_date:
                    ah_receipts_df['start_date'] = "2023-01-01"

                ah_receipts_df.to_parquet(output_file, engine="pyarrow")

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)


class ConvertJumboReceipts(luigi.Task):
    input_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()
    delimiter = luigi.Parameter(default='|')
    year_month_column = luigi.Parameter(default='year_month')

    parquet_engine = luigi.Parameter(default="pyarrow")
    csv_encoding = luigi.Parameter(default="latin1")

    def requires(self):
        return CsvFile(input_filename=self.input_filename)

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def run(self):
        with self.input().open('r') as input_file:
            with self.output().open('w') as output_file:
                jumbo_receipts_df = convert_jumbo_receipts(
                    input_file, self.delimiter, self.year_month_column, self.csv_encoding)
                jumbo_receipts_df.to_parquet(
                    output_file, engine=self.parquet_engine)


class ConvertAllJumboReceipts(luigi.Task):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    output_filename = luigi.Parameter(default='jumbo_receipts.parquet')
    delimiter = luigi.Parameter(default='|')
    year_month_column = luigi.Parameter(default='year_month')

    parquet_engine = luigi.Parameter(default="pyarrow")
    csv_encoding = luigi.Parameter(default='latin1')

    def requires(self):
        return [ConvertJumboReceipts(input_filename=os.path.join(self.input_directory, input_filename),
                                     output_filename=os.path.join(
                                         self.output_directory, input_filename.replace('.csv', '.parquet')),
                                     delimiter=self.delimiter,
                                     year_month_column=self.year_month_column,
                                     parquet_engine=self.parquet_engine,
                                     csv_encoding=self.csv_encoding)
                for input_filename in os.listdir(self.input_directory)
                if input_filename.endswith('.csv')
                and self._is_correct_receipt_file(input_filename)
                ]

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_directory, self.output_filename), format=luigi.format.Nop)

    def run(self):
        for input_receipts in self.input():
            receipts_dfs = []
            with input_receipts.open('r') as input_file:
                receipts_dfs.append(pd.read_parquet(
                    input_file, engine=self.parquet_engine))

            with self.output().open('w') as output_file:
                pd.concat(receipts_dfs).to_parquet(
                    output_file, engine=self.parquet_engine)

    def _is_correct_receipt_file(self, input_filename: str) -> bool:
        df = pd.read_csv(os.path.join(self.input_directory,
                         input_filename), delimiter=self.delimiter, nrows=1, encoding=self.csv_encoding)
        jumbo_columns = ["NUM_ISO_JAARWEEK",
                         "NUM_VESTIGING", "NUM_ARTIKEL", "NAM_ARTIKEL"]
        return set(df.columns.values).intersection(set(jumbo_columns)) == set(jumbo_columns)


class ConvertCSVToParquet(luigi.Task):
    """ This task converts a CSV file to a parquet file.

    Parameters
    ----------
    input_filename : luigi.PathParameter
        The input filename of the original CSV file.

    output_filename : luigi.PathParameter
        The output filename for the parquet file.

    delimiter : luigi.Parameter
        The delimiter used in the CSV file.

    """
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    delimiter = luigi.Parameter(default=';')
    encoding = luigi.Parameter(default='utf-8')
    extension = luigi.Parameter(default='.csv')
    decimal = luigi.Parameter(default=',')
    clean_cpi = luigi.BoolParameter(default=False)

    def get_parquet_filename(self):
        return os.path.join(self.output_directory, os.path.basename(
            self.input_filename).replace(self.extension, ".parquet"))

    def requires(self):
        if self.clean_cpi:
            return CleanCPIFile(input_filename=self.input_filename, output_filename=self.get_parquet_filename())
        return CsvFile(input_filename=self.input_filename)

    def run(self):
        with self.input().open('r') as input_file:
            with self.output().open('w') as output_file:
                convert_to_parquet(self.input_filename,
                                   input_file,
                                   output_file,
                                   delimiter=self.delimiter,
                                   encoding=self.encoding,
                                   extension=self.extension,
                                   decimal=',')

    def output(self):
        return luigi.LocalTarget(self.get_parquet_filename(), format=luigi.format.Nop)


class ConvertAllCSVToParquet(luigi.WrapperTask):
    input_directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    delimiter = luigi.Parameter(default=';')
    encoding = luigi.Parameter(default='utf-8')
    extension = luigi.Parameter(default='.csv')
    decimal = luigi.Parameter(default=',')
    clean_cpi = luigi.BoolParameter(default=False)

    def requires(self):
        return [ConvertCSVToParquet(input_filename=os.path.join(self.input_directory, input_filename),
                                    output_directory=self.output_directory,
                                    delimiter=self.delimiter,
                                    encoding=self.encoding,
                                    extension=self.extension,
                                    decimal=self.decimal,
                                    clean_cpi=self.clean_cpi)
                for input_filename in os.listdir(self.input_directory)
                if input_filename.endswith(self.extension)
                ]
