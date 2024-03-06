from .feature_extraction import FeatureExtractorFactory, FeatureExtractorType
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
        return luigi.LocalTarget(
            os.path.join(self.output_directory, f"features_{self.feature_extraction_method}.parquet"), format=luigi.format.Nop)

    def run(self):
        feature_extractor_factory = FeatureExtractorFactory()

        with self.input().open('r') as input_file:
            dataframe = pd.read_parquet(input_file)
            feature_extractor_factory.extract_features_and_save(
                dataframe,
                self.source_column,
                self.destination_column,
                self.output().path,
                batch_size=self.batch_size
            )
