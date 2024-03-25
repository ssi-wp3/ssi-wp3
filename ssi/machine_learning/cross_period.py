from typing import List, Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..files import get_features_files_in_directory
from ..parquet_file import ParquetFile
from .train_model import train_model
from .train_model_task import TrainModelTask
import pandas as pd
import luigi
import os


# TODO add an evaluation that trains a model on one supermarket and evaluates it on another.
# Check TFIDF and CountVectorizer for the feature extraction; they use a word dictionary,
# this dictionary may be supermarket specific! i.e. features from one supermarket may not be usable with another.
# TODO Return feature extraction pipeline instead?


class TrainModelOnPeriod(TrainModelTask):
    input_filename = luigi.PathParameter()
    period_column = luigi.Parameter()
    train_period = luigi.Parameter()
    label_column = luigi.Parameter()

    @property
    def train_from_scratch(self) -> List[FeatureExtractorType]:
        """ Return the feature extractors that require training from scratch.
        TFIDF and CountVectorizer require a word dictionary that is specific to the
        receipt texts seen at training time. To evaluate these models correctly we cannot
        use the files with extracted features as they are trained on the full dataset, not
        the specific period that we may want to evaluate.
        """
        return {
            FeatureExtractorType.tfidf_char,
            FeatureExtractorType.tfidf_word,
            FeatureExtractorType.count_char,
            FeatureExtractorType.count_vectorizer
        }

    def requires(self):
        return ParquetFile(self.input_filename)

    @property
    def feature_filename(self) -> str:
        feature_filename, _ = os.path.splitext(
            os.path.basename(self.input_filename))
        return feature_filename

    @property
    def model_filename(self) -> str:
        return os.path.join(self.output_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.joblib")

    @property
    def training_predictions_filename(self) -> str:
        return os.path.join(self.output_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.training_predictions.parquet")

    @property
    def predictions_filename(self) -> str:
        return os.path.join(self.output_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.predictions.parquet")

    @property
    def evaluations_filename(self) -> str:
        return os.path.join(self.output_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.evaluation.json")

    def get_data_for_period(self, input_file):
        dataframe = pd.read_parquet(input_file, engine=self.parquet_engine)
        dataframe = dataframe.drop_duplicates(
            [self.receipt_text_column, self.label_column])
        dataframe["is_train"] = dataframe[self.period_column] == self.train_period
        return dataframe

    def train_period_model(self,
                           train_dataframe: pd.DataFrame,
                           model_type: str,
                           feature_column: str,
                           label_column: str,
                           verbose: bool = False
                           ) -> Tuple[Pipeline, Dict[str, Any]]:
        return train_model(train_dataframe,
                           model_type,
                           feature_column,
                           label_column,
                           verbose=verbose)

    def prepare_data(self) -> pd.DataFrame:
        with self.input().open() as input_file:
            dataframe = self.get_data_for_period(input_file)
            return dataframe

    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ The training split is different for the period evaluation task: we train on one period and evaluate 
        on the others. Therefore we override the split_data method here. Furthermore, we would like to create a graph 
        for the whole range of periods, so we only split the data here for training and evaluate on the whole dataset.  
        The items for the training period have a True value in the is_train column, so we can 
        distinguish them in the test set. We also use this to filter the training items out here.

        Parameters:
        -----------
        dataframe: pd.DataFrame
            The dataframe to split

        test_size: float
            The size of the test set (Unused)

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training and test dataframes
        """
        return dataframe[dataframe["is_train"] == True], dataframe

    def train_model(self, train_dataframe: pd.DataFrame, training_predictions_file):
        self.model_trainer.fit(lambda: train_dataframe,
                               self.train_period_model,
                               training_predictions_file,
                               model_type=self.model_type,
                               feature_column=self.features_column,
                               label_column=self.label_column,
                               verbose=self.verbose)

    def run(self):
        print(
            f"Training model: {self.model_type} on period: {self.train_period}")

        if self.feature_extractor in self.train_from_scratch:
            raise NotImplementedError(
                "Training feature extractor from scratch not implemented")
        super().run()


class TrainModelOnAllPeriods(luigi.WrapperTask):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()

    filename_prefix = luigi.Parameter()
    label_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    batch_size = luigi.IntParameter(default=1000)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)
    period_columns = luigi.ListParameter()

    def requires(self):
        return [TrainModelOnPeriod(input_filename=os.path.join(self.input_directory, feature_filename),
                                   output_directory=self.output_directory,
                                   feature_extractor=self.feature_extractor,
                                   model_type=self.model_type,
                                   label_column=self.label_column,
                                   receipt_text_column=self.receipt_text_column,
                                   features_column=self.features_column,
                                   batch_size=self.batch_size,
                                   parquet_engine=self.parquet_engine,
                                   verbose=self.verbose,
                                   period_column=period_column,
                                   train_period=period)
                for feature_filename in get_features_files_in_directory(self.input_directory, self.filename_prefix, self.feature_extractor.value)
                for period_column in self.period_columns
                for period in pd.read_parquet(os.path.join(self.input_directory, feature_filename), engine=self.parquet_engine)[period_column].unique()
                ]
