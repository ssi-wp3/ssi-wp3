from .feature_extraction import FeatureExtractorFactory, FeatureExtractorType
from .files import get_combined_revenue_files_in_directory
import luigi
import pandas as pd
import os


class ParquetFile(luigi.ExternalTask):
    input_filename = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename, format=luigi.format.Nop)


class FeatureExtractionTask(luigi.Task):
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extraction_method = luigi.EnumParameter(enum=FeatureExtractorType)
    batch_size = luigi.IntParameter(default=1000)

    source_column = luigi.Parameter(default="receipt_text")
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
            with self.output().open('w') as output_file:
                feature_extractor_factory.extract_features_and_save(
                    dataframe,
                    self.source_column,
                    self.destination_column,
                    output_file,
                    feature_extractor_type=self.feature_extraction_method,
                    batch_size=self.batch_size
                )


class ExtractFeaturesForAllFiles(luigi.WrapperTask):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extraction_method = luigi.EnumParameter(enum=FeatureExtractorType)
    batch_size = luigi.IntParameter(default=1000)
    source_column = luigi.Parameter(default="receipt_text")
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
