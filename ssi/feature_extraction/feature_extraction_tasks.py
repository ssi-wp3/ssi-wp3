from .feature_extraction import FeatureExtractorFactory, FeatureExtractorType
from .files import get_combined_revenue_files_in_directory
from ..files import get_features_files_in_directory
from ..preprocessing.combine_unique_values import combine_unique_column_values
from ..parquet_file import ParquetFile
from ..constants import Constants
import luigi
import pandas as pd
import os
import tqdm


class FeatureExtractionTask(luigi.Task):
    """This task extracts features from a text column in a parquet file, normally the receipt_text column
    in a combined revenue file. The features are added to a new column in the dataframe and the new dataframe
    is written to the output_directory as a parquet file. The feature extraction method is specified by the
    feature_extraction_method parameter. The file is processed in batches of batch_size rows at a time.

    Parameters
    ----------
    input_filename : luigi.PathParameter
        The path to the input parquet file. Normally a combined revenue file in the preprocessing directory.

    output_directory : luigi.PathParameter
        The directory where the output parquet file will be written.

    feature_extraction_method : luigi.EnumParameter
        The feature extraction method to use. This is an instance of the FeatureExtractorType enum. Only values
        from that enum are allowed as parameter.

    batch_size : luigi.IntParameter
        The number of rows to process at a time. The default value is 1000.

    source_column : luigi.Parameter
        The name of the column containing the text to extract features from. The default value is Constants.RECEIPT_TEXT_COLUMN.

    destination_column : luigi.Parameter
        The name of the column to write the features to. The default value is "features".

    """
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extraction_method = luigi.EnumParameter(enum=FeatureExtractorType)
    batch_size = luigi.IntParameter(default=1000)

    source_column = luigi.Parameter(default=Constants.RECEIPT_TEXT_COLUMN)
    destination_column = luigi.Parameter(default="features")

    def requires(self):
        return ParquetFile(self.input_filename)

    def output(self):
        input_filename = os.path.splitext(
            os.path.basename(self.input_filename))[0]
        return luigi.LocalTarget(
            os.path.join(self.output_directory, f"{input_filename}_features_{self.feature_extraction_method.value}.parquet"), format=luigi.format.Nop)

    def run(self):
        feature_extractor_factory = FeatureExtractorFactory()

        with self.input().open('r') as input_file:
            dataframe = pd.read_parquet(input_file)
            with tqdm.tqdm(total=len(dataframe), desc=f"Extracting {self.feature_extraction_method.value} features", unit="rows") as progress_bar:
                with self.output().open('w') as output_file:
                    feature_extractor_factory.extract_features_and_save(
                        dataframe,
                        self.source_column,
                        self.destination_column,
                        output_file,
                        feature_extractor_type=self.feature_extraction_method,
                        batch_size=self.batch_size,
                        progress_bar=progress_bar
                    )


class ExtractFeaturesForAllFiles(luigi.WrapperTask):
    """This task extracts features from all combined revenue files in the input_directory using the specified
    feature_extraction_method. It searches for all the files in the input directory that start with the filename_prefix (normally ssi) and contain the string "revenue". The output parquet files are written to the output_directory.
    This task is a wrapper task that runs a FeatureExtractionTask for each combined revenue file in the input_directory.

    Parameters
    ----------
    input_directory : luigi.PathParameter
        The directory containing the combined revenue files.

    output_directory : luigi.PathParameter
        The directory where the output parquet files will be written.

    feature_extraction_method : luigi.EnumParameter
        The feature extraction method to use. This is an instance of the FeatureExtractorType enum. Only values
        from that enum are allowed as parameter.

    batch_size : luigi.IntParameter
        The number of rows to process at a time. The default value is 1000.

    source_column : luigi.Parameter
        The name of the column containing the text to extract features from. The default value is Constants.RECEIPT_TEXT_COLUMN.

    destination_column : luigi.Parameter
        The name of the column to write the features to. The default value is "features".

    """
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extraction_method = luigi.EnumParameter(enum=FeatureExtractorType)
    batch_size = luigi.IntParameter(default=1000)
    source_column = luigi.Parameter(default=Constants.RECEIPT_TEXT_COLUMN)
    destination_column = luigi.Parameter(default="features")
    filename_prefix = luigi.Parameter(default="ssi")

    def requires(self):
        for filename in get_combined_revenue_files_in_directory(self.input_directory, project_prefix=self.filename_prefix):
            yield FeatureExtractionTask(
                input_filename=os.path.join(self.input_directory, filename),
                output_directory=self.output_directory,
                feature_extraction_method=self.feature_extraction_method,
                batch_size=self.batch_size,
                source_column=self.source_column,
                destination_column=self.destination_column
            )


class ExtractAllFeatures(luigi.WrapperTask):
    """This task extracts features from all combined revenue files in the input_directory using all feature extraction
    methods in the FeatureExtractorType enum. It searches for all the files in the input directory that start with the 
    filename_prefix (normally ssi) and contain the string "revenue". For each of these files and each of the feature
    extraction methods it creates a separate ExtractFeatureForAllFiles task. The output parquet files are written to 
    the output_directory.

    Parameters
    ----------

    input_directory : luigi.PathParameter
        The directory containing the combined revenue files.

    output_directory : luigi.PathParameter
        The directory where the output parquet files will be written.

    batch_size : luigi.IntParameter
        The number of rows to process at a time. The default value is 1000.

    source_column : luigi.Parameter
        The name of the column containing the text to extract features from. The default value is Constants.RECEIPT_TEXT_COLUMN.

    destination_column : luigi.Parameter
        The name of the column to write the features to. The default value is "features".

    filename_prefix : luigi.Parameter
        The prefix of the combined revenue files. The default value is "ssi".

    """
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    batch_size = luigi.IntParameter(default=1000)
    source_column = luigi.Parameter(default=Constants.RECEIPT_TEXT_COLUMN)
    destination_column = luigi.Parameter(default="features")
    filename_prefix = luigi.Parameter(default="ssi")

    def requires(self):
        for feature_extraction_method in FeatureExtractorType:
            yield ExtractFeaturesForAllFiles(
                input_directory=self.input_directory,
                output_directory=self.output_directory,
                feature_extraction_method=feature_extraction_method,
                batch_size=self.batch_size,
                source_column=self.source_column,
                destination_column=self.destination_column,
                filename_prefix=self.filename_prefix
            )


class CombineUniqueValues(luigi.Task):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    filename_prefix = luigi.Parameter()
    key_columns = luigi.ListParameter(
        [Constants.RECEIPT_TEXT_COLUMN, Constants.COICOP_LABEL_COLUMN])
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    parquet_engine = luigi.Parameter()

    def requires(self):
        store_filenames = [os.path.join(self.input_directory, filename)
                           for filename in get_features_files_in_directory(
                               self.input_directory, self.filename_prefix)
                           if f"{self.feature_extractor.value}.parquet" in filename]

        return [ParquetFile(store_filename)
                for store_filename in store_filenames]

    @property
    def output_filename(self):
        return os.path.join(self.output_directory, f"{self.filename_prefix}_{self.feature_extractor.value}_unique_values.parquet")

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def run(self):
        input_files = [input_file.open("r") for input_file in self.input()]
        combine_unique_column_values(filenames=input_files,
                                     output_filename=self.output_filename + ".tmp",
                                     key_columns=self.key_columns,
                                     parquet_engine=self.parquet_engine
                                     )

        with self.output().open('w') as output_file:
            with open(self.output_filename + ".tmp", "rb") as tmp_file:
                output_file.write(tmp_file.read())

        os.remove(self.output_filename + ".tmp")
