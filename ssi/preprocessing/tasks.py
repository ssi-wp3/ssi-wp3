from luigi.contrib.external_program import ExternalProgramTask
from parquet import convert_to_parquet
from preprocess_data import get_revenue_files_in_folder, combine_revenue_files
import pandas as pd
import luigi


class CleanCPIFile(ExternalProgramTask):
    """ This task converts CBS CPI CSV files to CSV file format that can be read by python.

    The CBS CPI files use an unusual encoding. The shell script takes an input 
    file, reads the input_filename, skips the first 3 bytes (BOM), and converts 
    them from cp1252 to utf-8. Last, it filters out the carriage returns '\r' 
    and writes the result to a file with output_filename.

    Parameters
    ----------
    input_filename : luigi.PathParameter
        The input filename of the original CBS CPI file.

    output_filename : luigi.PathParameter
        The output filename for the cleaned CBS CPI file.

    """
    input_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()

    def program_args(self):
        return ['bash', './convert_cpi_file.sh', self.input_directory, self.output_directory]

    def output(self):
        """

        TODO: check if this doesn't cause the atomic writes problem mentioned here: 
        https://luigi.readthedocs.io/en/latest/luigi_patterns.html#atomic-writes-problem
        """
        return luigi.LocalTarget(self.output_filename)


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
    output_filename = luigi.PathParameter()
    delimiter = luigi.Parameter(default=';')
    encoding = luigi.Parameter(default='utf-8')
    extension = luigi.Parameter(default='.csv')
    decimal = luigi.Parameter(default=',')

    def requires(self):
        return CleanCPIFile(input_filename=self.input_filename, output_filename=self.output_filename)

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
        return luigi.LocalTarget(self.output_filename)


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
    filename_prefix = luigi.Parameter(default="Omzet")
    parquet_engine = luigi.Parameter(default="pyarrow")

    def requires(self):
        revenue_files = get_revenue_files_in_folder(
            self.input_directory,
            self.store_name,
            self.filename_prefix)
        return [ConvertCSVToParquet(input_filename)
                for input_filename in revenue_files
                ]

    def output(self):
        return luigi.LocalTarget(self.output_filename)

    def run(self):
        input_files = [input for input in self.input()]
        combined_revenue = combine_revenue_files()

        with self.output().open('w') as output_file:
            combined_revenue.to_parquet(
                output_file, engine=self.parquet_engine)
