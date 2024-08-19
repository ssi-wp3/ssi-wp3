from ..parquet_file import ParquetFile
from ..hyper_params.sampler import create_sampler_for_pipeline
from .bootstrap_models import bootstrap_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import luigi
import pandas as pd


class BootstrapModelTask(luigi.Task):
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    number_of_bootstraps = luigi.IntParameter(default=10)
    number_of_samples_per_bootstrap = luigi.IntParameter(default=10)
    feature_column = luigi.Parameter(default='features')
    label_column = luigi.Parameter(default='coicop_number')
    random_state = luigi.IntParameter(default=42)
    engine = luigi.Parameter(default='pyarrow')

    def requires(self):
        return ParquetFile(self.input_filename)

    def output(self):
        return luigi.LocalTarget('bootstrap_model_results.csv')

    def run(self):
        sklearn_pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('clf', LogisticRegression())
        ])

        with self.input().open('r') as input_file:
            dataframe = pd.read_parquet(input_file, engine=self.engine)

            param_sampler = (value for value in create_sampler_for_pipeline(
                'CountVectorizer', 'LogisticRegression', self.number_of_bootstraps, self.random_state)
            )
            bootstrap_df = bootstrap_model(sklearn_pipeline,
                                           param_sampler,
                                           dataframe,
                                           self.number_of_bootstraps,
                                           self.number_of_samples_per_bootstrap,
                                           self.feature_column,
                                           self.label_column,
                                           self.random_state)
            bootstrap_df.to_csv(self.output().path, index=False)
