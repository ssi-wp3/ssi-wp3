from pandas.core.api import DataFrame as DataFrame
from train_model_task import TrainModelTask
from ..parquet_file import ParquetFile
from ..preprocessing.files import get_store_name_from_combined_filename
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..files import get_features_files_in_directory
from .train_model import train_model
import pandas as pd
import luigi
import os


class TrainWithinStoreModel(TrainModelTask):
    """Train a model for a single store."""
    store_filename = luigi.PathParameter()
    parquet_engine = luigi.Parameter(default="pyarrow")

    @property
    def store(self):
        """Get the store name from the store filename.

        Returns
        -------
        str
            The store name
        """
        return get_store_name_from_combined_filename(self.store_filename)

    @property
    def training_predictions_filename(self) -> str:
        """Get the training predictions filename.

        Returns
        -------
        str
            The training predictions filename
        """
        return os.path.join(
            self.output_directory, f"with_store_{self.store}_{self.feature_extractor.value}_{self.model_type}_training_predictions.parquet")

    @property
    def training_evaluation_filename(self) -> str:
        """Get the training evaluation filename.

        Returns
        -------
        str
            The training evaluation filename
        """
        return os.path.join(
            self.output_directory, f"with_store_{self.store}_{self.feature_extractor.value}_{self.model_type}_training.evaluation.json")

    @property
    def predictions_filename(self) -> str:
        """Get the predictions filename.

        Returns
        -------
        str
            The predictions filename
        """
        return os.path.join(
            self.output_directory, f"with_store_{self.store}_{self.feature_extractor.value}_{self.model_type}_predictions.parquet")

    @property
    def model_filename(self) -> str:
        """Get the model filename.

        Returns
        -------
        str
            The model filename
        """
        return os.path.join(
            self.output_directory, f"with_store_{self.store}_{self.feature_extractor.value}_{self.model_type}_model.joblib")

    @property
    def evaluation_filename(self) -> str:
        """Get the evaluation filename.

        Returns
        -------
        str
            The evaluation filename
        """
        return os.path.join(
            self.output_directory, f"with_store_{self.store}_{self.feature_extractor.value}_{self.model_type}.evaluation.json")

    def requires(self):
        """Get the requirements for the task.

        Returns
        -------
        ParquetFile
            The input file
        """
        return ParquetFile(self.store_filename)

    def prepare_data(self) -> pd.DataFrame:
        """Prepare the data for training.

        Returns
        -------
        pd.DataFrame
            The prepared data
        """
        with self.input().open() as parquet_file:
            df = pd.read_parquet(parquet_file, engine=self.parquet_engine)
            df = df.drop_duplicates(
                [self.receipt_text_column, self.label_column])
            return df

    def train_model(self, train_dataframe: DataFrame, training_predictions_file):
        """Train the model and write the training predictions to the given file.

        Parameters
        ----------
        train_dataframe : DataFrame
            The training data
        training_predictions_file : str
            The file to write the training predictions to
                """
        # TODO other examples use model trainer but this is pytorch specific...
        self.pipeline = train_model(
            train_dataframe,
            model_type=self.model_type,
            feature_column=self.features_column,
            label_column=self.label_column
        )


class TrainAllWithinStoreModels(luigi.WrapperTask):
    """Train a model for all stores."""
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    label_column = luigi.Parameter()
    test_size = luigi.FloatParameter(default=0.2)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    filename_prefix = luigi.Parameter()

    def requires(self):
        """Get the requirements for the task.

        Returns
        -------
        List[TrainWithinStoreModel]
            The requirements
        """
        return [TrainWithinStoreModel(store_filename=feature_filename,
                                      output_directory=self.output_directory,
                                      feature_extractor=self.feature_extractor,
                                      model_type=self.model_type,
                                      receipt_text_column=self.receipt_text_column,
                                      features_column=self.features_column,
                                      label_column=self.label_column,
                                      test_size=self.test_size,
                                      parquet_engine=self.parquet_engine,
                                      verbose=self.verbose)

                for feature_filename in get_features_files_in_directory(self.input_directory, self.filename_prefix, self.feature_extractor.value)
                ]
