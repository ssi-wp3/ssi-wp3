from .feature_extraction import FeatureExtractorFactory, FeatureExtractorType
import luigi


class ParquetFile(luigi.ExternalTask):
    input_filename = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename)


class FeatureExtractionTask(luigi.Task):
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extraction_method = luigi.EnumParameter(enum=FeatureExtractorType)
    batch_size = luigi.IntParameter(default=1000)

    def requires(self):
        return ParquetFile(self.input_filename)

    def run(self):
        feature_extractor_factory = FeatureExtractorFactory()
        feature_extractor = feature_extractor_factory.create_feature_extractor(
            self.feature_extraction_method)
