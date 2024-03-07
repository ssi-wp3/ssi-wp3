from .adversarial import train_adversarial_model
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..preprocessing.files import get_store_name_from_combined_filename, get_combined_revenue_files_in_folder
from .utils import store_combinations
import pandas as pd
import luigi
import joblib
import os


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
            "evaluation": luigi.LocalTarget(evaluation_filename, format=luigi.format.Nop)
        }

    def run(self):
        store1 = get_store_name_from_combined_filename(self.store1_filename)
        store2 = get_store_name_from_combined_filename(self.store2_filename)
        with input()[0].open("r") as store1_file, input()[1].open("r") as store2_file:
            store1_dataframe = pd.read_parquet(
                store1_file, engine=self.parquet_engine)
            store1_dataframe[self.store_id_column] = store1
            store2_dataframe = pd.read_parquet(
                store2_file, engine=self.parquet_engine)
            store2_dataframe[self.store_id_column] = store2

            pipeline, evaluation_dict = train_adversarial_model(store1_dataframe,
                                                                store2_dataframe,
                                                                self.store_id_column,
                                                                self.receipt_text_column,
                                                                self.feature_extractor,
                                                                self.model_type,
                                                                self.test_size,
                                                                self.verbose)
            with self.output()["model"].open("w") as model_file:
                joblib.dump(pipeline, model_file)

            with self.output()["evaluation"].open("w") as evaluation_file:
                evaluation_file.write(evaluation_dict)


class TrainAllAdversarialModels(luigi.Task):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(FeatureExtractorType)
    model_type = luigi.Parameter()
    store_id_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    test_size = luigi.FloatParameter(default=0.2)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)

    filename_prefix = luigi.Parameter()

    def requires(self):
        store_filenames = [filename for filename in get_combined_revenue_files_in_folder(
            self.input_directory, self.filename_prefix)]

        return [TrainAdversarialModelTask(store1_filename=store1_filename,
                                          store2_filename=store2_filename,
                                          output_directory=self.output_directory,
                                          feature_extractor=self.feature_extractor,
                                          model_type=self.model_type,
                                          store_id_column=self.store_id_column,
                                          receipt_text_column=self.receipt_text_column,
                                          test_size=self.test_size,
                                          parquet_engine=self.parquet_engine,
                                          verbose=self.verbose)
                for store1_filename, store2_filename in store_combinations(store_filenames)]
