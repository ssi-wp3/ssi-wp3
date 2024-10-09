from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from ..preprocessing.files import get_store_name_from_combined_filename
from ..files import get_features_files_in_directory
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..parquet_file import ParquetFile
from ..report import LuigiReportFileManager
from .train_model import train_model, train_and_evaluate_model, drop_labels_with_few_samples
from .train_model_task import TrainModelTask
from .utils import store_combinations, read_parquet_indices
from .model_pipeline import ModelPipeline
from pyarrow import parquet as pq
import pandas as pd
import numpy as np
import luigi
import os


def create_combined_dataframe(store1_dataframe: pd.DataFrame,
                              store2_dataframe: pd.DataFrame,
                              store_id_column: str,
                              receipt_text_column: str,
                              features_column: str
                              ) -> pd.DataFrame:
    """Combine two store dataframes into one dataframe for adversarial validation.
    The two dataframes are combined by concatenating the receipt text columns and the store id columns.
    The machine learning model can then be trained to predict the store id based on the receipt text column.

    Parameters
    ----------
    store1_dataframe : pd.DataFrame
        The first store dataframe

    store2_dataframe : pd.DataFrame
        The second store dataframe

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text
    """
    return pd.concat([
        store1_dataframe[[store_id_column,
                          receipt_text_column, features_column]],
        store2_dataframe[[store_id_column,
                          receipt_text_column, features_column]]
    ])


def filter_unique_combinations(dataframe: pd.DataFrame,
                               store_id_column: str,
                               receipt_text_column: str,
                               features_column: str
                               ) -> pd.DataFrame:
    """Filter the dataframe to only contain unique combinations of store id and receipt text.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to filter

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text

    features_column : str
        The column name containing the features

    Returns
    -------
    pd.DataFrame
        The dataframe containing only unique combinations of store id, receipt text, and features
    """
    return dataframe.drop_duplicates(subset=[store_id_column, receipt_text_column])


def create_combined_and_filtered_dataframe(store1_dataframe: pd.DataFrame,
                                           store2_dataframe: pd.DataFrame,
                                           store_id_column: str,
                                           receipt_text_column: str,
                                           features_column: str):
    """Create a combined and filtered dataframe for adversarial validation.
    It first combines the two store dataframes into one dataframe and then filters the dataframe to only contain unique combinations of store id and receipt text.

    Parameters
    ----------

    store1_dataframe : pd.DataFrame
        The first store dataframe

    store2_dataframe : pd.DataFrame
        The second store dataframe

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text

    features_column : str
        The column name containing the features

    Returns
    -------

    pd.DataFrame
        The combined and filtered dataframe
    """

    print("Creating combined dataframe")
    combined_dataframe = create_combined_dataframe(
        store1_dataframe, store2_dataframe, store_id_column, receipt_text_column, features_column)
    print("Filtering unique combinations")
    unique_dataframe = filter_unique_combinations(
        combined_dataframe, store_id_column, receipt_text_column, features_column)

    return unique_dataframe


def train_adversarial_model(store1_dataframe: pd.DataFrame,
                            store2_dataframe: pd.DataFrame,
                            store_id_column: str,
                            receipt_text_column: str,
                            features_column: str,
                            model_type: str,
                            test_size: float = 0.2,
                            number_of_jobs: int = -1,
                            verbose: bool = False
                            ) -> Pipeline:
    """Train an adversarial model to predict the store id based on the receipt text column.

    Parameters
    ----------
    store1_dataframe : pd.DataFrame
        The first store dataframe

    store2_dataframe : pd.DataFrame
        The second store dataframe

    receipt_text_column : str
        The column name containing the receipt text

    features_column : str
        The column name containing the features

    store_id_column : str
        The column name containing the store id

    model_type : str
        The model to use

    test_size : float, optional
        The test size for the train/test split, by default 0.2

    number_of_jobs : int, optional
        The number of jobs to use, by default -1

    verbose : bool, optional
        Verbose output, by default False

    Returns
    -------
    object
        The trained model
    """
    unique_dataframe = create_combined_and_filtered_dataframe(
        store1_dataframe, store2_dataframe, store_id_column, receipt_text_column, features_column)
    print("Training and evaluating model")

    pipeline, evaluation_dict = train_and_evaluate_model(unique_dataframe,
                                                         features_column,
                                                         store_id_column,
                                                         model_type,
                                                         test_size=test_size,
                                                         evaluation_function=evaluate_adversarial_pipeline,
                                                         verbose=verbose)
    return pipeline, evaluation_dict


def evaluate_adversarial_pipeline(y_true: np.array,
                                  y_pred: np.array,
                                  ) -> Dict[str, Any]:
    """
    Evaluate the adversarial pipeline.

    Parameters
    ----------
    y_true : np.array
        The true labels
    y_pred : np.array
        The predicted labels

    Returns
    -------
    Dict[str, Any]
        The evaluation metrics

    """
    label_encoder = LabelEncoder()

    # Fit the label encoder on the combined labels to prevent errors when the labels are not the same
    combined_labels = np.concatenate([y_true, y_pred])
    label_encoder.fit(combined_labels)

    y_true = label_encoder.transform(y_true)
    y_pred = label_encoder.transform(y_pred)

    return {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


class TrainAdversarialModelTask(TrainModelTask):
    """
    Train an adversarial model to predict the store id based on the receipt text column.
    If we can predict the store id based on the receipt text, then the receipt text between
    stores are very different.

    """
    store1_filename = luigi.PathParameter()
    store2_filename = luigi.PathParameter()

    store_id_column = luigi.Parameter()

    @property
    def training_predictions_filename(self) -> str:
        """ Get the training predictions filename.

        Returns
        -------
        str
            The training predictions filename
        """
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}_training_predictions.parquet")

    @property
    def training_evaluation_filename(self) -> str:
        """ Get the training evaluation filename.

        Returns
        -------
        str
            The training evaluation filename
        """
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}_training.evaluation.json")

    @property
    def predictions_filename(self) -> str:
        """ Get the predictions filename.

        Returns
        -------
        str
            The predictions filename
        """
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}_predictions.parquet")

    @property
    def model_filename(self) -> str:
        """ Get the model filename.

        Returns
        -------
        str
            The model filename
        """
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}_model.joblib")

    @property
    def evaluation_filename(self) -> str:
        """ Get the evaluation filename.

        Returns
        -------
        str
            The evaluation filename
        """
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}.evaluation.json")

    @property
    def store1(self):
        """ Get the store1 name.

        Returns
        -------
        str
            The store1 name
        """
        return get_store_name_from_combined_filename(self.store1_filename)

    @property
    def store2(self):
        """ Get the store2 name.

        Returns
        -------
        str
            The store2 name
        """
        return get_store_name_from_combined_filename(self.store2_filename)

    def requires(self):
        """ Get the required inputs.

        Returns
        -------
        list[ParquetFile]
            The required inputs
        """
        return [ParquetFile(self.store1_filename), ParquetFile(self.store2_filename)]

    def get_adversarial_data(self, store_file, store_name: str) -> pd.DataFrame:
        """ Get the adversarial data for the store.

        Parameters
        ----------
        store_file : file
            The store file
        store_name : str
            The store name
        """
        store_dataframe = pd.read_parquet(
            store_file, engine=self.parquet_engine, columns=[self.receipt_text_column, self.store_id_column])
        store_dataframe = store_dataframe.drop_duplicates(
            [self.receipt_text_column, self.store_id_column])

        store_dataframe = read_parquet_indices(
            store_file, store_dataframe.index, columns=[
                self.receipt_text_column, self.features_column, self.store_id_column])
        store_dataframe[self.store_id_column] = store_name
        return store_dataframe

    def get_all_adversarial_data(self, store1: str, store2: str, store1_file, store2_file) -> pd.DataFrame:
        """ Get the adversarial data for the store.

        Parameters
        ----------
        store1 : str
            The store1 name
        store2 : str
            The store2 name
        store1_file : file
            The store1 file
        store2_file : file
            The store2 file

        Returns
        -------
        pd.DataFrame
            The adversarial data for the store
        """
        store2_dataframe = self.get_adversarial_data(store2_file, store2)
        return create_combined_and_filtered_dataframe(store1_dataframe,
                                                      store2_dataframe,
                                                      self.store_id_column,
                                                      self.receipt_text_column,
                                                      self.features_column)

    def prepare_data(self) -> pd.DataFrame:
        """ Prepare the data for the adversarial model.

        Returns
        -------
        pd.DataFrame
            The prepared data for the adversarial model
        """
        with self.input()[0].open("r") as store1_file, self.input()[1].open("r") as store2_file:
            print("Reading parquet files")
            combined_dataframe = self.get_all_adversarial_data(
                self.store1, self.store2, store1_file, store2_file)

            return drop_labels_with_few_samples(
                combined_dataframe, self.label_column, min_samples=10)

    def train_model(self, train_dataframe: pd.DataFrame, training_predictions_file):
        """ Train the adversarial model.

        Parameters
        ----------
        train_dataframe : pd.DataFrame
            The training data for the adversarial model
        training_predictions_file : str
            The file to save the training predictions to
        """
        self.model_trainer.fit(train_dataframe,
                               self.train_adversarial_model,
                               training_predictions_file,
                               label_mapping=self.train_label_mapping,
                               features_column=self.features_column,
                               store_id_column=self.store_id_column,
                               model_type=self.model_type,
                               verbose=self.verbose)

    def run(self):
        """ Run the adversarial model training task.
        """
        print(
            f"Running adversarial model training task for {self.store1_filename} and {self.store2_filename}")
        print(f"Store1: {self.store1}, Store2: {self.store2}")
        luigi_report_file_manager = LuigiReportFileManager(
            self.input(), self.output())
        model_pipeline = ModelPipeline.pipeline_for(
            model=self.model_type,
            report_file_manager=luigi_report_file_manager) \
            .with_train_dataset(train_dataframe) \
            .with_features_column(self.features_column) \
            .with_label_column(self.store_id_column) \
            .save_training_predictions_to(self.training_predictions_filename) \
            .save_training_evaluation_to(self.training_evaluation_filename) \
            .save_test_predictions_to(self.predictions_filename) \
            .save_test_evaluation_to(self.evaluation_filename) \
            .save_model_to(self.model_filename)
        model_pipeline.train_model()


class TrainAllAdversarialModels(luigi.WrapperTask):
    """
    Train all adversarial models for the given feature extractor and model type.
    """
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()
    store_id_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    test_size = luigi.FloatParameter(default=0.2)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    filename_prefix = luigi.Parameter()

    def requires(self):
        """ Get the required inputs.

        Returns
        -------
        list[TrainAdversarialModelTask]
            The required inputs
        """
        store_filenames = [os.path.join(self.input_directory, filename)
                           for filename in get_features_files_in_directory(
                               self.input_directory, self.filename_prefix)
                           if f"{self.feature_extractor.value}.parquet" in filename]

        return [TrainAdversarialModelTask(store1_filename=store1_filename,
                                          store2_filename=store2_filename,
                                          output_directory=self.output_directory,
                                          feature_extractor=self.feature_extractor,
                                          model_type=self.model_type,
                                          store_id_column=self.store_id_column,
                                          receipt_text_column=self.receipt_text_column,
                                          features_column=self.features_column,
                                          label_column=self.store_id_column,
                                          test_size=self.test_size,
                                          parquet_engine=self.parquet_engine,
                                          verbose=self.verbose)
                for store1_filename, store2_filename in store_combinations(store_filenames)]
