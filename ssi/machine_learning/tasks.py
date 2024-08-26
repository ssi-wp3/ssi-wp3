from ..parquet_file import ParquetFile
from .hyper_params.sampler import create_sampler_for_pipeline, FeatureExtractorType
from .bootstrap.bootstrap_models import bootstrap_model
from ..preprocessing.combine_unique_values import drop_empty_receipts
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
import luigi
import pandas as pd


class BootstrapModelTask(luigi.Task):
    input_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()
    number_of_bootstraps = luigi.IntParameter(default=10)
    number_of_samples_per_bootstrap = luigi.IntParameter(default=10)
    feature_extractor_type = luigi.EnumParameter(enum=FeatureExtractorType)
    feature_column = luigi.Parameter(default='features')
    receipt_text_column = luigi.Parameter(default='receipt_text')
    label_column = luigi.Parameter(default='coicop_number')
    random_state = luigi.IntParameter(default=42)
    engine = luigi.Parameter(default='pyarrow')
    delimiter = luigi.Parameter(default=';')

    def requires(self):
        return ParquetFile(self.input_filename)

    def output(self):
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def feature_extractor_for_type(self, feature_extractor_type: FeatureExtractorType):
        if feature_extractor_type == FeatureExtractorType.count_vectorizer:
            return CountVectorizer()
        elif feature_extractor_type == FeatureExtractorType.tfidf_vectorizer:
            return TfidfVectorizer()
        elif feature_extractor_type == FeatureExtractorType.hashing_vectorizer:
            return HashingVectorizer()

    def run(self):
        sklearn_pipeline = Pipeline([
            ('vectorizer', self.feature_extractor_for_type(
                self.feature_extractor_type)),
            ('clf', LogisticRegression())
        ])

        with self.input().open('r') as input_file:
            dataframe = pd.read_parquet(input_file, engine=self.engine)
            dataframe = drop_empty_receipts(
                dataframe, self.receipt_text_column)

            param_sampler = (value for value in create_sampler_for_pipeline(
                self.feature_extractor_type, 'LogisticRegression', self.number_of_bootstraps, self.random_state)
            )

            with self.output().open('w') as output_file:
                results_df = bootstrap_model(sklearn_pipeline,
                                             param_sampler,
                                             dataframe,
                                             self.number_of_bootstraps,
                                             self.number_of_samples_per_bootstrap,
                                             self.feature_column,
                                             self.label_column,
                                             self.random_state)
                results_df.to_csv(output_file, index=False, sep=self.delimiter)
