from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod, abstractproperty
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from evaluate import ConfusionMatrixEvaluator
from trainer import ModelTrainer
from .adversarial import evaluate_adversarial_pipeline, create_combined_and_filtered_dataframe
from .train_model import train_and_evaluate_model, train_model, evaluate_model, evaluate
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..preprocessing.files import get_store_name_from_combined_filename
from ..files import get_features_files_in_directory
from .utils import store_combinations
import pandas as pd
import numpy as np
import luigi
import joblib
import os
import json
import tqdm

# TODO add an evaluation that trains a model on one supermarket and evaluates it on another.
# Check TFIDF and CountVectorizer for the feature extraction; they use a word dictionary,
# this dictionary may be supermarket specific! i.e. features from one supermarket may not be usable with another.
# TODO Return feature extraction pipeline instead?

# TODO duplicated


class ParquetFile(luigi.ExternalTask):
    filename = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.filename, format=luigi.format.Nop)


class TrainModelTask(luigi.Task, ABC):
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()

    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    test_size = luigi.FloatParameter(default=0.2)
    batch_predict_size = luigi.IntParameter(default=1000)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    @property
    def model_trainer(self) -> ModelTrainer:
        model_evaluator = ConfusionMatrixEvaluator()
        return ModelTrainer(
            model_evaluator=model_evaluator,
            batch_predict_size=self.batch_predict_size,
            parquet_engine=self.parquet_engine
        )

    @property
    def training_predictions_key(self) -> str:
        return "training_predictions"

    @property
    def predictions_key(self) -> str:
        return "predictions"

    @property
    def model_key(self) -> str:
        return f"model"

    @property
    def evaluation_key(self) -> str:
        return "evaluation"

    @abstractproperty
    def training_predictions_filename(self) -> str:
        pass

    @abstractproperty
    def predictions_filename(self) -> str:
        pass

    @abstractproperty
    def model_filename(self) -> str:
        pass

    @abstractproperty
    def evaluation_filename(self) -> str:
        pass

    @abstractmethod
    def prepare_data(self) -> pd.DataFrame:
        pass

    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(dataframe, test_size=test_size)

    @abstractmethod
    def train_model(self, train_dataframe: pd.DataFrame, training_predictions_file):
        pass

    def output(self):
        return {
            self.model_key: luigi.LocalTarget(self.model_filename, format=luigi.format.Nop),
            self.training_predictions_key: luigi.LocalTarget(self.training_predictions_filename, format=luigi.format.Nop),
            self.predictions_key: luigi.LocalTarget(self.predictions_filename, format=luigi.format.Nop),
            self.evaluation_key: luigi.LocalTarget(
                self.evaluation_filename)
        }

    def run(self):
        print("Preparing data")
        dataframe = self.prepare_data()

        print("Splitting data")
        train_dataframe, test_dataframe = self.split_data(
            dataframe, test_size=self.test_size)

        print("Training model & writing training predictions to disk")
        with self.output()[self.training_predictions_key].open("w") as training_predictions_file:
            self.train_model(train_dataframe, training_predictions_file)

        print("Writing test predictions to disk")
        with self.output()[self.predictions_key].open("w") as predictions_file:
            self.model_trainer.predict(lambda: test_dataframe,
                                       predictions_file)

        print("Writing model to disk")
        with self.output()[self.model_key].open("w") as model_file:
            self.model_trainer.write_model(model_file)

        print("Writing evaluation to disk")
        with self.output()[self.evaluation_key].open("w") as evaluation_file:
            self.model_trainer.write_evaluation(evaluation_file)


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
    def model_trainer(self) -> ModelTrainer:
        model_evaluator = ConfusionMatrixEvaluator()
        return ModelTrainer(
            model_evaluator=model_evaluator,
            batch_predict_size=self.batch_predict_size,
            parquet_engine=self.parquet_engine
        )

    @property
    def model_filename(self) -> str:
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}_model.joblib")

    @property
    def training_predictions_filename(self) -> str:
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}_training_predictions.parquet")

    @property
    def predictions_filename(self) -> str:
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}_predictions.parquet")

    @property
    def evaluation_filename(self) -> str:
        return os.path.join(
            self.output_directory, f"adversarial_{self.store1}_{self.store2}_{self.feature_extractor.value}_{self.model_type}.evaluation.json")

    @property
    def store1(self):
        return get_store_name_from_combined_filename(self.store1_filename)

    @property
    def store2(self):
        return get_store_name_from_combined_filename(self.store2_filename)

    def requires(self):
        return [ParquetFile(self.store1_filename), ParquetFile(self.store2_filename)]

    def read_parquet_data(self, store1_file: str) -> pd.DataFrame:
        store1_dataframe = pd.read_parquet(
            store1_file, engine=self.parquet_engine)

        return store1_dataframe

    def get_adversarial_data(self, store1_file, store_name: str):
        store1_dataframe = self.read_parquet_data(store1_file)
        store1_dataframe = store1_dataframe.drop_duplicates(
            [self.receipt_text_column, self.store_id_column])
        store1_dataframe[self.store_id_column] = store_name
        return store1_dataframe

    def get_all_adversarial_data(self, store1: str, store2: str, store1_file, store2_file) -> pd.DataFrame:
        store1_dataframe = self.get_adversarial_data(store1_file, store1)
        store2_dataframe = self.get_adversarial_data(store2_file, store2)
        return create_combined_and_filtered_dataframe(store1_dataframe,
                                                      store2_dataframe,
                                                      self.store_id_column,
                                                      self.receipt_text_column,
                                                      self.features_column)

    def train_adversarial_model(self,
                                adversarial_dataframe: pd.DataFrame,
                                features_column: str,
                                store_id_column: str,
                                model_type: str,
                                test_size: float = 0.1,
                                verbose: bool = False
                                ) -> Tuple[Pipeline, Dict[str, Any]]:
        """This trains the adversarial model and uses an additional validation set to evaluate the model.

        TODO not sure whether we should use validation split.

        Parameters:
        -----------
        adversarial_dataframe: pd.DataFrame
            The dataframe with the adversarial data

        features_column: str
            The column with the features

        store_id_column: str
            The column with the store ids

        model_type: str
            The type of model to train

        test_size: float
            The size of the test set

        verbose: bool
            Whether to print verbose output

        Returns:
        --------
        Tuple[Pipeline, Dict[str, Any]]
            The trained pipeline and the evaluation dictionary
        """
        return train_and_evaluate_model(adversarial_dataframe,
                                        features_column,
                                        store_id_column,
                                        model_type,
                                        test_size=test_size,
                                        evaluation_function=evaluate_adversarial_pipeline,
                                        verbose=verbose)

    def prepare_data(self) -> pd.DataFrame:
        with self.input()[0].open("r") as store1_file, self.input()[1].open("r") as store2_file:
            print("Reading parquet files")
            return self.get_all_adversarial_data(
                self.store1, self.store2, store1_file, store2_file)

    def train_model(self, train_dataframe: pd.DataFrame, training_predictions_file):
        self.model_trainer.fit(lambda: train_dataframe,
                               self.train_adversarial_model,
                               training_predictions_file,
                               features_column=self.features_column,
                               store_id_column=self.store_id_column,
                               model_type=self.model_type,
                               test_size=self.test_size,
                               verbose=self.verbose)

    def run(self):
        print(
            f"Running adversarial model training task for {self.store1_filename} and {self.store2_filename}")
        print(f"Store1: {self.store1}, Store2: {self.store2}")
        super().run()


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

    def get_model_filename(self, store1: str, store2: str) -> str:
        return os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_{self.label_column}_model.joblib")

    def get_train_evaluation_filename(self, store1: str, store2: str) -> str:
        return os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_{self.label_column}_train_evaluation.json")

    def get_evaluation_filename(self, store1: str, store2: str) -> str:
        return os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_{self.label_column}_evaluation.json")

    def get_evaluation_filename(self, store1: str, store2: str) -> str:
        return os.path.join(
            self.output_directory, f"cross_store_{store1}_{store2}_{self.feature_extractor.value}_{self.model_type}_{self.label_column}_evaluation.json")

    def output(self):
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
        store_dataframe = pd.read_parquet(
            store_file, engine=self.parquet_engine)
        store_dataframe = store_dataframe.drop_duplicates(
            [self.receipt_text_column, self.label_column])

        return store_dataframe

    def get_all_store_data(self, store1_file, store2_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
        store1_dataframe = self.get_store_data(store1_file)
        store2_dataframe = self.get_store_data(store2_file)
        return store1_dataframe, store2_dataframe

    def run(self):

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

    def get_model_filename(self, feature_filename: str) -> str:
        return os.path.join(self.output_directory, f"{feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.joblib")

    def get_training_predictions_filename(self, feature_filename: str) -> str:
        return os.path.join(self.output_directory, f"{feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.training_predictions.parquet")

    def get_predictions_filename(self, feature_filename: str) -> str:
        return os.path.join(self.output_directory, f"{feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.predictions.parquet")

    def get_evaluations_filename(self, feature_filename: str) -> str:
        return os.path.join(self.output_directory, f"{feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.evaluation.json")

    def output(self):
        feature_filename, _ = os.path.splitext(
            os.path.basename(self.input_filename))
        return {
            self.model_key: luigi.LocalTarget(self.get_model_filename(feature_filename), format=luigi.format.Nop),
            self.training_predictions_key: luigi.LocalTarget(self.get_predictions_filename(feature_filename), format=luigi.format.Nop),
            self.predictions_key: luigi.LocalTarget(self.get_predictions_filename(feature_filename), format=luigi.format.Nop),
            self.evaluation_key: luigi.LocalTarget(
                self.get_evaluations_filename(feature_filename))
        }

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
            return dataframe[dataframe["is_train"] == True]

    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame]:
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
        Tuple[pd.DataFrame]
            The training and test dataframes
        """
        return dataframe[dataframe["is_train" == True]], dataframe

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
