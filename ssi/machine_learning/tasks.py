from ..parquet_file import ParquetFile
from .hyper_params.sampler import FeatureExtractorType
from .bootstrap.bootstrap_models import bootstrap_model
from ..preprocessing.combine_unique_values import drop_empty_receipts, drop_unknown
from .hyper_params.pipeline import pipeline_and_sampler_for
from sklearn.linear_model import LogisticRegression
import luigi
import pandas as pd


class BootstrapModelTask(luigi.Task):
    """ Task to bootstrap a model. """
    input_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()
    number_of_bootstraps = luigi.IntParameter(default=10)
    number_of_samples_per_bootstrap = luigi.IntParameter(default=10)
    feature_extractor_type = luigi.EnumParameter(enum=FeatureExtractorType)
    feature_column = luigi.Parameter(default='features')
    receipt_text_column = luigi.Parameter(default='receipt_text')
    label_column = luigi.Parameter(default='coicop_number')
    keep_unknown = luigi.BoolParameter(default=False)
    random_state = luigi.IntParameter(default=42)
    engine = luigi.Parameter(default='pyarrow')
    delimiter = luigi.Parameter(default=';')

    def requires(self):
        """ Get the requirements for the task.

        Returns
        -------
        ParquetFile
            The input file
        """
        return ParquetFile(self.input_filename)

    def output(self):
        """ Get the output for the task.

        Returns
        -------
        luigi.LocalTarget
            The output file
        """
        return luigi.LocalTarget(self.output_filename, format=luigi.format.Nop)

    def run(self):
        """ Run the task.

        Raises
        ------
        Exception
            If the task fails
        """
        param_sampler, sklearn_pipeline = pipeline_and_sampler_for(self.feature_extractor_type,
                                                                   LogisticRegression(),
                                                                   self.number_of_bootstraps,
                                                                   self.random_state)

        with self.input().open('r') as input_file:
            dataframe = pd.read_parquet(input_file, engine=self.engine)
            dataframe = drop_empty_receipts(
                dataframe, self.receipt_text_column)

            if not self.keep_unknown:
                dataframe = drop_unknown(dataframe, self.label_column)

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
