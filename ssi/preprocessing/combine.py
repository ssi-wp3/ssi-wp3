
from .utils import ParquetFile
from .files import get_revenue_files_in_folder
from .convert import ConvertCSVToParquet
import pandas as pd
import luigi


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


class AddReceiptTexts(luigi.Task):
    """ This task adds the receipt texts to the combined revenue file.

    Parameters
    ----------
    input_filename : luigi.Parameter
        The input filename of the combined revenue file.

    output_filename : luigi.Parameter
        The output filename for the combined revenue file with receipt texts.

    receipt_texts_filename : luigi.Parameter
        The filename of the receipt texts file.

    """
    input_filename = luigi.Parameter()
    output_filename = luigi.Parameter()
    receipt_texts_filename = luigi.Parameter()

    parquet_engine = luigi.Parameter()

    def requires(self):
        return [ParquetFile(input_filename=self.input_filename),
                ParquetFile(input_filename=self.receipt_texts_filename)]

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def run(self):
        with self.input()[0].open("r") as input_file:
            combined_df = pd.read_parquet(input_file)
            with self.input()[1].open("r") as receipt_texts_file:
                receipt_texts = pd.read_parquet(receipt_texts_file)

                combined_df["receipt_text"] = combined_df["receipt_id"].map(
                    receipt_texts)

                with self.output().open("w") as output_file:
                    combined_df.to_parquet(
                        output_file, engine=self.parquet_engine)
