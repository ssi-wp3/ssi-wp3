from typing import List
from .adversarial import train_adversarial_model
from .train_model import train_and_evaluate_model, evaluate_model, evaluate
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..preprocessing.files import get_store_name_from_combined_filename, get_combined_revenue_files_in_folder
from ..files import get_features_files_in_directory
from .utils import store_combinations
import pandas as pd
import luigi
import joblib
import os
import json


# TODO add an evaluation that trains a model on one supermarket and evaluates it on another.
# Check TFIDF and CountVectorizer for the feature extraction; they use a word dictionary,
# this dictionary may be supermarket specific! i.e. features from one supermarket may not be usable with another.
# TODO Return feature extraction pipeline instead?

# TODO duplicated


class ParquetFile(luigi.ExternalTask):
    filename = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.filename, format=luigi.format.Nop)


class TrainAdversarialModelTask(luigi.Task):
    """
    Train an adversarial model to predict the store id based on the receipt text column.
    If we can predict the store id based on the receipt text, then the receipt text between
    stores are very different.

    """
    store1_filename = luigi.PathParameter()
    store2_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()

    store_id_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    test_size = luigi.FloatParameter(default=0.2)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    def requires(self):
        return [ParquetFile(self.store1_filename), ParquetFile(self.store2_filename)]

    def output(self):
        store1 = get_store_name_from_combined_filename(self.store1_filename)
        store2 = get_store_name_from_combined_filename(self.store2_filename)
        model_filename = os.path.join(
            self.output_directory, f"adversarial_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_model.joblib")
        evaluation_filename = os.path.join(
            self.output_directory, f"adversarial_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}.evaluation.json")
        return {
            "model": luigi.LocalTarget(model_filename, format=luigi.format.Nop),
            "evaluation": luigi.LocalTarget(evaluation_filename)
        }

    def run(self):
        print(
            f"Running adversarial model training task for {self.store1_filename} and {self.store2_filename}")
        store1 = get_store_name_from_combined_filename(self.store1_filename)
        store2 = get_store_name_from_combined_filename(self.store2_filename)
        print(f"Store1: {store1}, Store2: {store2}")
        with self.input()[0].open("r") as store1_file, self.input()[1].open("r") as store2_file:
            print("Reading parquet files")
            store1_dataframe = pd.read_parquet(
                store1_file, engine=self.parquet_engine)
            store1_dataframe[self.store_id_column] = store1
            store2_dataframe = pd.read_parquet(
                store2_file, engine=self.parquet_engine)
            store2_dataframe[self.store_id_column] = store2

            print("Training adversarial model")
            pipeline, evaluation_dict = train_adversarial_model(store1_dataframe,
                                                                store2_dataframe,
                                                                self.store_id_column,
                                                                self.receipt_text_column,
                                                                self.features_column,
                                                                self.model_type,
                                                                self.test_size,
                                                                self.verbose)
            print("Writing adversarial model to disk")
            with self.output()["model"].open("w") as model_file:
                joblib.dump(pipeline, model_file)

            print("Writing evaluation to disk")
            with self.output()["evaluation"].open("w") as evaluation_file:
                json.dump(evaluation_dict, evaluation_file)


class TrainAllAdversarialModels(luigi.WrapperTask):
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
                                          test_size=self.test_size,
                                          parquet_engine=self.parquet_engine,
                                          verbose=self.verbose)
                for store1_filename, store2_filename in store_combinations(store_filenames)]


class CrossStoreEvaluation(luigi.Task):
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
        return [(ParquetFile(combinations[0]), ParquetFile(combinations[1]))
                for combinations in [
                    (self.store1_filename, self.store2_filename),
                    (self.store2_filename, self.store1_filename)
        ]
        ]

    def output(self):
        store1 = get_store_name_from_combined_filename(self.store1_filename)
        store2 = get_store_name_from_combined_filename(self.store2_filename)
        model_filename = os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_model.joblib")
        train_evaluation_filename = os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_train_evaluation.json")
        evaluation_filename = os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_evaluation.json")
        return [
            {
                f"model_{combinations[0]}_{combinations[1]}": luigi.LocalTarget(model_filename, format=luigi.format.Nop),
                f"train_evaluation_{combinations[0]}_{combinations[1]}": luigi.LocalTarget(train_evaluation_filename),
                f"evaluation_{combinations[0]}_{combinations[1]}": luigi.LocalTarget(evaluation_filename)
            }
            for combinations in [(store1, store2), (store2, store1)
                                 ]
        ]

    def run(self):

        print(
            f"Running cross store evaluation training task for {self.store1_filename} and {self.store2_filename}")
        for input_combinations in self.input():
            store1 = get_store_name_from_combined_filename(
                input_combinations[0].path)
            store2 = get_store_name_from_combined_filename(
                input_combinations[1].path)
            print(f"Train on: {store1}, Evaluate on: {store2}")

            with input_combinations[0].open("r") as store1_file, input_combinations[1].open("r") as store2_file:
                print("Reading parquet files")
                store1_dataframe = pd.read_parquet(
                    store1_file, engine=self.parquet_engine)
                store1_dataframe = store1_dataframe.drop_duplicates(
                    [self.receipt_text_column, self.label_column])
                store2_dataframe = pd.read_parquet(
                    store2_file, engine=self.parquet_engine)
                store2_dataframe = store2_dataframe.drop_duplicates(
                    [self.receipt_text_column, self.label_column])

                print(f"Training model on {store1}")
                pipeline, train_evaluation_dict = train_and_evaluate_model(store1_dataframe,
                                                                           self.features_column,
                                                                           self.label_column,
                                                                           self.model_type,
                                                                           test_size=self.test_size,
                                                                           verbose=self.verbose)
                print(f"Evaluating model on {store2}")
                evaluation_dict = evaluate_model(pipeline,
                                                 store2_dataframe,
                                                 self.features_column,
                                                 self.label_column,
                                                 evaluation_function=evaluate,
                                                 )

                print("Writing model to disk")
                with self.output()[f"model_{store1}_{store2}"].open("w") as model_file:
                    joblib.dump(pipeline, model_file)

                print("Writing training evaluation to disk")
                with self.output()[f"train_evaluation_{store1}_{store2}"].open("w") as train_evaluation_file:
                    json.dump(train_evaluation_dict, train_evaluation_file)

                print("Writing evaluation to disk")
                with self.output()[f"evaluation_{store1}_{store2}"].open("w") as evaluation_file:
                    json.dump(evaluation_dict, evaluation_file)


class AllCrossStoreEvaluations(luigi.WrapperTask):
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


class TrainModelOnPeriod(luigi.Task):

    period_column = luigi.Parameter()

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
