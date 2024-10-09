from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from ..preprocessing.files import get_store_name_from_combined_filename
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..parquet_file import ParquetFile
from ..files import get_features_files_in_directory
from .train_model import train_and_evaluate_model, evaluate_model, evaluate
from .utils import store_combinations, read_parquet_indices, read_unique_rows
import pandas as pd
import luigi
import joblib
import json
import os


class CrossStoreEvaluation(luigi.Task):
    """ Train a model on one store and evaluate it on another. """
    store1_filename = luigi.PathParameter()
    store2_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()

    label_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    test_size = luigi.FloatParameter(default=0.2)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    def requires(self):
        """ Get the required tasks.

        Returns
        -------
        list[ParquetFile]
            The required tasks
        """
        return [(ParquetFile(combinations[0]), ParquetFile(combinations[1]))
                for combinations in [
                    (self.store1_filename, self.store2_filename),
                    (self.store2_filename, self.store1_filename)
        ]
        ]

    def get_model_filename(self, store1: str, store2: str) -> str:
        """ Get the model filename.

        Parameters
        ----------
        store1 : str
            The first store
        store2 : str
            The second store
        """
        return os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_{self.label_column}_model.joblib")

    def get_train_evaluation_filename(self, store1: str, store2: str) -> str:
        """ Get the train evaluation filename.

        Parameters
        ----------
        store1 : str
            The first store
        store2 : str
            The second store
        """
        return os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_{self.label_column}_train_evaluation.json")

    def get_evaluation_filename(self, store1: str, store2: str) -> str:
        """ Get the evaluation filename.

        Parameters
        ----------
        store1 : str
            The first store
        store2 : str
            The second store
        """
        return os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_{self.label_column}_evaluation.json")

    def output(self):
        """ Get the output.

        Returns
        -------
        list[dict]
            The output
        """
        store1 = get_store_name_from_combined_filename(self.store1_filename)
        store2 = get_store_name_from_combined_filename(self.store2_filename)
        return [
            {
                f"model_{combination_store1}_{combination_store2}": luigi.LocalTarget(self.get_model_filename(combination_store1, combination_store2), format=luigi.format.Nop),
                f"train_evaluation_{combination_store1}_{combination_store2}": luigi.LocalTarget(self.get_evaluation_filename(combination_store1, combination_store2)),
                # TODO add evaluation filename here.
                f"evaluation_{combination_store1}_{combination_store2}": luigi.LocalTarget(self.get_evaluation_filename(combination_store1, combination_store2))
            }
            for combination_store1, combination_store2 in [(store1, store2), (store2, store1)
                                                           ]
        ]

    def get_store_data(self, store_file) -> pd.DataFrame:
        """ Get the store data.

        Parameters
        ----------
        store_file : file
            The store file
        """
        return read_unique_rows(store_file,
                                group_columns=[
                                    self.receipt_text_column, self.label_column],
                                value_columns=[
                                    self.receipt_text_column, self.features_column, self.label_column]
                                )

    def get_all_store_data(self, store1_file, store2_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Get the store data for both stores.

        Parameters
        ----------
        store1_file : file
            The first store file
        store2_file : file
            The second store file
        """
        store1_dataframe = self.get_store_data(store1_file)
        store2_dataframe = self.get_store_data(store2_file)
        return store1_dataframe, store2_dataframe

    def run(self):
        """ Run the task. """
        print(
            f"Running cross store evaluation training task for {self.store1_filename} and {self.store2_filename}")
        for index, input_combinations in enumerate(self.input()):
            store1 = get_store_name_from_combined_filename(
                input_combinations[0].path)
            store2 = get_store_name_from_combined_filename(
                input_combinations[1].path)
            print(f"Train on: {store1}, Evaluate on: {store2}")

            with input_combinations[0].open("r") as store1_file, input_combinations[1].open("r") as store2_file:
                print("Reading parquet files")
                store1_dataframe, store2_dataframe = self.get_all_store_data(
                    store1_file, store2_file)

                print("Encoding labels")
                label_encoder = LabelEncoder()
                all_labels = pd.concat(
                    [store1_dataframe[self.label_column], store2_dataframe[self.label_column]])
                label_encoder.fit(all_labels)

                encoded_label_column = f"{self.label_column}_index"
                store1_dataframe[encoded_label_column] = label_encoder.transform(
                    store1_dataframe[self.label_column])
                store2_dataframe[encoded_label_column] = label_encoder.transform(
                    store2_dataframe[self.label_column])

                print(f"Training model on {store1}")
                pipeline, train_evaluation_dict = train_and_evaluate_model(store1_dataframe,
                                                                           self.features_column,
                                                                           encoded_label_column,
                                                                           self.model_type,
                                                                           test_size=self.test_size,
                                                                           verbose=self.verbose)
                print(f"Evaluating model on {store2}")
                evaluation_dict = evaluate_model(pipeline,
                                                 store2_dataframe,
                                                 self.features_column,
                                                 encoded_label_column,
                                                 evaluation_function=evaluate,
                                                 )

                print("Writing model to disk")
                outputs = self.output()[index]
                with outputs[f"model_{store1}_{store2}"].open("w") as model_file:
                    joblib.dump(pipeline, model_file)

                print("Writing training evaluation to disk")
                with outputs[f"train_evaluation_{store1}_{store2}"].open("w") as train_evaluation_file:
                    json.dump(train_evaluation_dict, train_evaluation_file)

                print("Writing evaluation to disk")
                with outputs[f"evaluation_{store1}_{store2}"].open("w") as evaluation_file:
                    json.dump(evaluation_dict, evaluation_file)


class AllCrossStoreEvaluations(luigi.WrapperTask):
    """ Train a model on one store and evaluate it on another for all store combinations. """
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
        """ Get the required tasks.

        Returns
        -------
        list[CrossStoreEvaluation]
            The required tasks
        """
        store_filenames = [os.path.join(self.input_directory, filename)
                           for filename in get_features_files_in_directory(
                               self.input_directory, self.filename_prefix)
                           if self.feature_extractor.value in filename]

        return [CrossStoreEvaluation(store1_filename=store1_filename,
                                     store2_filename=store2_filename,
                                     output_directory=self.output_directory,
                                     feature_extractor=self.feature_extractor,
                                     model_type=self.model_type,
                                     receipt_text_column=self.receipt_text_column,
                                     features_column=self.features_column,
                                     label_column=self.label_column,
                                     test_size=self.test_size,
                                     parquet_engine=self.parquet_engine,
                                     verbose=self.verbose)
                for store1_filename, store2_filename in store_combinations(store_filenames)]
