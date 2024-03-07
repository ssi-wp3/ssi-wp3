from .utils import ParquetFile
from .files import get_combined_revenue_files_in_folder, get_store_name_from_combined_filename, get_receipt_texts_for_store
import pandas as pd
import luigi
import tempfile
import duckdb
import os


class AddReceiptTextsWithDate(luigi.Task):
    """ This task adds the receipt texts to the combined revenue file. The receipt texts are coupled with the combined
    revenue file on the EAN number and the start date. The resulting dataframe is written to the output file.
    """
    input_filename = luigi.PathParameter()
    receipt_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()
    store_name = luigi.Parameter()
    receipt_text_column = luigi.Parameter(default="receipt_text")
    parquet_engine = luigi.Parameter()

    def output(self):
        return luigi.target.FileSystemTarget(self.output_filename)

    def couple_receipt_file(self):
        """ This method couples the receipt texts with the combined revenue file.
        Both parquet files are stored in a temporary duckdb database. The receipt texts
        are then joined with the combined revenue file on the EAN number and the start date.
        The resulting table is then written to the output file.

        Parameters
        ----------
        combined_df : pd.DataFrame
            The combined revenue dataframe.

        receipt_text_column : str
            The name of the receipt text column.

        parquet_engine : str
            The parquet engine to use.
        """
        with tempfile.TemporaryDirectory() as duck_db_temp_dir:
            con = duckdb.connect(f"ssi_{self.store_name}.db",
                                 config={
                                     "temp_directory": duck_db_temp_dir
                                 })
            con.sql(
                f"create table {self.store_name}_receipts as select * from read_parquet('{self.receipt_filename}')")
            con.sql(f"""drop table if exists {self.store_name}_revenue;
                    create table {self.store_name}_revenue as select 
                        date_trunc('day', strptime(year_month, '%Y%m')) as start_date, 
                        last_day(strptime(year_month, '%Y%m')) as end_date, * 
                    from read_parquet('{self.input_filename}');
                    """)
            con.sql(
                f"create index {self.store_name}_revenue_ean_idx on {self.store_name}_revenue (ean_number)")
            con.sql(f"""drop table if exists {self.store_name}_receipts;
                    create table {self.store_name}_receipts as select * from receipt_texts
                    """)
            con.sql(
                f"create index {self.store_name}_receipts_ean_idx on {self.store_name}_receipts (ean_number)")

            with self.output().temporary_path() as output_path:
                receipt_revenue_table = f"{self.store_name}_revenue_receipts"
                con.sql(f"""create table {receipt_revenue_table} as
                        select pr.*, pc.{self.receipt_text_column} from {self.store_name}_revenue as pr 
                        inner join {self.store_name}_receipts as pc on pr.ean_number = pc.ean_number 
                        where pc.start_date >= pr.start_date and pc.start_date <= pr.end_date
                    """)
                con.sql(
                    f"copy {receipt_revenue_table} to '{self.output_filename}' with (format 'parquet')")

    def run(self):
        self.couple_receipt_file()


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
    store_name = luigi.Parameter()
    receipt_text_column = luigi.Parameter(default="receipt_text")
    ean_name_column = luigi.Parameter(default="ean_name")
    parquet_engine = luigi.Parameter()

    def requires(self):
        if not self.receipt_texts_filename:
            return [ParquetFile(input_filename=self.input_filename)]

        return [ParquetFile(input_filename=self.input_filename),
                ParquetFile(input_filename=self.receipt_texts_filename)]

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def run(self):
        with self.input()[0].open("r") as input_file:
            combined_df = pd.read_parquet(
                input_file, engine=self.parquet_engine)

            if self.store_name.lower() == "lidl":
                self.add_receipt_text_from_revenue_file(
                    combined_df, self.receipt_text_column, self.ean_name_column)
            elif self.store_name.lower() == "ah":
                self.couple_ah_receipt_file(
                    combined_df, self.parquet_engine)
            else:
                self.couple_receipt_file(
                    combined_df, self.receipt_text_column, self.parquet_engine)

    def couple_ah_receipt_file(self,
                               combined_df: pd.DataFrame,
                               parquet_engine: str):
        """ This method couples the receipt texts with the combined revenue file for AH.

        Parameters
        ----------
        combined_df : pd.DataFrame
            The combined revenue dataframe.

        parquet_engine : str
            The parquet engine to use to write the data to disk.
        """
        with self.input()[1].open("r") as receipt_texts_file:
            receipt_texts = pd.read_parquet(
                receipt_texts_file, engine=parquet_engine)

            receipt_revenue_df = combined_df.merge(
                receipt_texts, on=["store_id", "esba_number", "isba_number", "ean_number"])

            with self.output().open("w") as output_file:
                receipt_revenue_df.to_parquet(
                    output_file, engine=parquet_engine)

    def add_receipt_text_from_revenue_file(self,
                                           combined_df: pd.DataFrame,
                                           receipt_text_column: str,
                                           ean_name_column: str
                                           ):
        """ Lidl uses the EAN name as the receipt text. The EAN name is already
        present in the combined revenue file. This method adds an extra receipt text column
        to the dataframe derived from the ean name column. After that, it writes the dataframe
        to the output file.

        Parameters
        ----------
        combined_df : pd.DataFrame
            The combined revenue dataframe.

        receipt_text_column : str
            The name of the receipt text column.

        ean_name_column : str
            The name of the EAN name column.
        """
        combined_df[receipt_text_column] = combined_df[ean_name_column]
        with self.output().open("w") as output_file:
            combined_df.to_parquet(
                output_file, engine=self.parquet_engine)


class AddAllReceiptTexts(luigi.WrapperTask):
    """ Adds the receipt texts to all combined revenue files.

    Parameters
    ----------
    input_directory : luigi.PathParameter
        The directory where the combined revenue files are stored.

    output_directory : luigi.PathParameter
        The output directory for the combined revenue files with receipt texts.

    revenue_file_prefix : luigi.Parameter
        The prefix of the combined revenue files.

    receipt_text_column : luigi.Parameter
        The name of the receipt text column.

    ean_name_column : luigi.Parameter
        The name of the EAN name column.

    parquet_engine : luigi.Parameter
        The parquet engine to use.
    """
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    revenue_file_prefix = luigi.Parameter()
    receipt_text_column = luigi.Parameter(default="receipt_text")
    ean_name_column = luigi.Parameter(default="ean_name")
    parquet_engine = luigi.Parameter()

    def requires(self):
        for input_file in get_combined_revenue_files_in_folder(self.input_directory, self.revenue_file_prefix):
            store_name = get_store_name_from_combined_filename(input_file)
            receipt_text_filenames = get_receipt_texts_for_store(
                self.input_directory, store_name)

            receipt_text_filename = receipt_text_filenames[0] if len(
                receipt_text_filenames) > 0 else None

            output_filename = os.path.join(
                self.output_directory, os.path.basename(input_file))
            yield AddReceiptTexts(input_filename=input_file,
                                  output_filename=output_filename,
                                  receipt_texts_filename=receipt_text_filename,
                                  store_name=store_name,
                                  receipt_text_column=self.receipt_text_column,
                                  ean_name_column=self.ean_name_column,
                                  parquet_engine=self.parquet_engine
                                  )
