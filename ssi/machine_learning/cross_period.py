from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from ..feature_extraction.feature_extraction import FeatureExtractorType
from ..files import get_features_files_in_directory
from ..parquet_file import ParquetFile
from .train_model_task import TrainModelTask
from .data_loaders.pytorch_provider import ParquetDataset, TorchLogisticRegression, TorchMLP
from .evaluate import calculate_metrics_per_group
from ..analysis.products import compare_products_per_period
from collections import OrderedDict
from functools import partial
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.profiler import profiler
from torch.utils.data import DataLoader
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from cnvrg import Experiment
import pandas as pd
import pyarrow as pa
import luigi
import os


# TODO add an evaluation that trains a model on one supermarket and evaluates it on another.
# Check TFIDF and CountVectorizer for the feature extraction; they use a word dictionary,
# this dictionary may be supermarket specific! i.e. features from one supermarket may not be usable with another.
# TODO Return feature extraction pipeline instead?


class TrainModelOnPeriod(TrainModelTask):
    """ Train a model on a specific period.
    """
    input_filename = luigi.PathParameter()
    period_column = luigi.Parameter()
    train_period = luigi.Parameter()

    gpu_device = luigi.Parameter(default="cuda:0")
    learning_rate = luigi.FloatParameter(default=0.001)
    batch_size = luigi.IntParameter(default=1000)
    number_of_epochs = luigi.IntParameter(default=10)

    number_of_workers = luigi.IntParameter(default=1)
    prefetch_factor = luigi.IntParameter(default=2)

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
        """ Get the required inputs.

        Returns
        -------
        ParquetFile
            The required input
        """
        return ParquetFile(self.input_filename)

    @property
    def feature_filename(self) -> str:
        """ Get the feature filename.

        Returns
        -------
        str
            The feature filename
        """
        feature_filename, _ = os.path.splitext(
            os.path.basename(self.input_filename))
        return feature_filename

    @property
    def training_predictions_filename(self) -> str:
        """ Get the training predictions filename.

        Returns
        -------
        str
            The training predictions filename
        """
        return os.path.join(self.model_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.training_predictions.parquet")

    @property
    def training_evaluation_filename(self) -> str:
        """ Get the training evaluation filename.

        Returns
        -------
        str
            The training evaluation filename
        """
        return os.path.join(self.model_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.training_evaluation.json")

    @property
    def predictions_filename(self) -> str:
        """ Get the predictions filename.

        Returns
        -------
        str
            The predictions filename
        """
        return os.path.join(self.model_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.predictions.parquet")

    @property
    def model_filename(self) -> str:
        """ Get the model filename.

        Returns
        -------
        str
            The model filename
        """
        return os.path.join(self.model_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.joblib")

    @property
    def evaluation_filename(self) -> str:
        """ Get the evaluation filename.

        Returns
        -------
        str
            The evaluation filename
        """
        return os.path.join(self.model_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.evaluation.json")

    @property
    def per_period_evaluation_key(self) -> str:
        """ Get the per period evaluation key.

        Returns
        -------
        str
            The per period evaluation key
        """
        return f"evaluation_per_period"

    @property
    def per_period_evaluation_filename(self) -> str:
        """ Get the per period evaluation filename.

        Returns
        -------
        str
            The per period evaluation filename
        """
        return os.path.join(self.model_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.per_period_evaluation.parquet")

    @property
    def per_period_coicop_evaluation_key(self) -> str:
        """ Get the per period coicop evaluation key.

        Returns
        -------
        str
            The per period coicop evaluation key
        """
        return f"evaluation_per_coicop_period"

    @property
    def per_period_coicop_evaluation_filename(self) -> str:
        """ Get the per period coicop evaluation filename.

        Returns
        -------
        str
            The per period coicop evaluation filename
        """
        return os.path.join(self.model_directory, f"{self.feature_filename}_{self.model_type}_{self.label_column}_{self.train_period}.per_period_coicop_evaluation.parquet")

    def output(self):
        """ Get the output.

        Returns
        -------
        dict
            The output
        """
        output_dict = super().output()
        output_dict[self.per_period_evaluation_key] = luigi.LocalTarget(
            self.per_period_evaluation_filename, format=luigi.format.Nop)
        output_dict[self.per_period_coicop_evaluation_key] = luigi.LocalTarget(
            self.per_period_coicop_evaluation_filename, format=luigi.format.Nop)
        return output_dict

    def get_data_for_period(self, input_file) -> pd.DataFrame:
        """ Get the data for the period.

        Parameters
        ----------
        input_file : file
            The input file

        Returns
        -------
        pd.DataFrame
            The data for the period
        """
        # dataframe = pd.read_parquet(input_file, engine=self.parquet_engine)

        dataframe = pa.parquet.read_table(
            input_file, columns=[self.period_column, self.receipt_text_column, self.label_column]).to_pandas()

        print("Adding is_train column")
        dataframe["is_train"] = dataframe[self.period_column] == self.train_period
        return dataframe

    def prepare_data(self) -> pd.DataFrame:
        """ Prepare the data.

        Returns
        -------
        pd.DataFrame
            The prepared data
        """
        with self.input().open() as input_file:
            dataframe = self.get_data_for_period(input_file)
            return dataframe

    def split_data(self, dataframe: ParquetDataset, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        training_dataframe = dataframe[dataframe[self.period_column] == self.train_period].drop_duplicates(
            [self.receipt_text_column, self.label_column])
        self.number_of_categories = training_dataframe[self.label_column].nunique(
        )
        print(
            f"Number of categories in training data: {self.number_of_categories}")

        testing_dataframe = dataframe[dataframe[self.period_column] != self.train_period].drop_duplicates(
            [self.receipt_text_column, self.label_column])
        print(
            f"Number of categories in testing data: {testing_dataframe[self.label_column].nunique()}")

        self.retrieve_label_mappings(
            training_dataframe, testing_dataframe, self.label_column)

        parquet_dataset = ParquetDataset(
            self.input().open(), self.features_column, self.label_column, label_mapping=self.train_label_mapping, memory_map=True)
        self.feature_vector_size = parquet_dataset.feature_vector_size
        # self.number_of_categories = parquet_dataset.number_of_classes

        training_dataset = torch.utils.data.Subset(
            parquet_dataset, training_dataframe.index)

        test_dataset = ParquetDataset(self.input().open(
        ), self.features_column, self.label_column, label_mapping=self.test_label_mapping, memory_map=True)

        testing_dataset = torch.utils.data.Subset(
            test_dataset, testing_dataframe.index)

        return training_dataset, testing_dataset

    def fit_model(self,
                  train_dataframe: pd.DataFrame,
                  device: str,
                  learning_rate: float,
                  num_epochs: int,
                  batch_size: int,
                  early_stopping_patience: int) -> Any:
        """ Fit the model.

        Parameters
        ----------
        train_dataframe : pd.DataFrame
            The training dataframe
        device : str
            The device to use
        learning_rate : float
            The learning rate
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        early_stopping_patience : int
            The early stopping patience

        Returns
        -------
        Any
            The model
        """

        # experiment = Experiment()

        print(f"Training model on {device} with learning rate {learning_rate}, num_epochs {num_epochs}, batch_size {batch_size}, number of epochs {num_epochs}, and early stopping patience {early_stopping_patience}")
        print(
            f"Model input dim: {self.feature_vector_size}, output dim: {self.number_of_categories}")
        # model = TorchLogisticRegression(
        model = TorchMLP(
            input_dim=self.feature_vector_size,
            output_dim=self.number_of_categories
        )

        model = model.to(device)
        print(f"Model moved to {device}: {next(model.parameters()).is_cuda}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = F.cross_entropy

        data_size = len(train_dataframe)
        training_size = int(data_size * 0.8)
        val_size = data_size - training_size
        # TODO implement test set!!
        train_set, val_set = torch.utils.data.random_split(
            train_dataframe, [training_size, val_size])

        def collate_fn(batch):
            features, labels, _ = zip(*batch)

            features = torch.stack(features)
            labels = torch.tensor(labels)
            return features, labels

        print("Creating DataLoaders")
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            num_workers=self.number_of_workers,
            collate_fn=collate_fn
            # pin_memory_device=device
        )

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            num_workers=self.number_of_workers,
            collate_fn=collate_fn
            # pin_memory_device=device
        )

        print("Creating trainers and evaluators")

        def loss_with_regularization(y_pred, y, model, lambda_reg, order=2, criterion=F.cross_entropy):
            loss = criterion(y_pred, y)

            # Add L2 regularization
            lx_norm_reg = 0.0
            for param in model.parameters():
                lx_norm_reg += torch.linalg.norm(param, ord=order)
            loss += lambda_reg * lx_norm_reg

            return loss

        loss_function = partial(
            loss_with_regularization, model=model, lambda_reg=0.001, order=2)

        train_engine = create_supervised_trainer(
            model,
            optimizer=optimizer,
            loss_fn=loss_function,
            device=device
        )

        val_metrics = {
            "accuracy": Accuracy(),
            "precision": Precision(average="weighted"),
            "recall": Recall(average="weighted"),
            "loss": Loss(criterion)
        }
        train_evaluator = create_supervised_evaluator(
            model, metrics=val_metrics, device=device)
        val_evaluator = create_supervised_evaluator(
            model, metrics=val_metrics, device=device)

        log_interval = 1000

        @ train_engine.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(engine):
            # experiment.log_metric("loss", engine.state.output)
            print(
                f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

        @ train_engine.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            train_evaluator.run(train_loader)
            metrics = train_evaluator.state.metrics

            # for metric, value in metrics.items():
            #    experiment.log_metric(f"train_{metric}", value)

            metrics = " ".join([f"Avg {metric}: {value}" for metric,
                                value in metrics.items()])
            print(
                f"Training Results - Epoch[{trainer.state.epoch}] {metrics}")

        @ train_engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            val_evaluator.run(val_loader)
            metrics = val_evaluator.state.metrics

            # for metric, value in metrics.items():
            #    experiment.log_metric(f"val_{metric}", value)

            metrics = " ".join([f"Avg {metric}: {value}" for metric,
                                value in metrics.items()])
            print(
                f"Validation Results - Epoch[{trainer.state.epoch}] {metrics}")

        # Score function to return current value of any metric we defined above in val_metrics
        def score_function(engine):
            return engine.state.metrics["accuracy"]

        model_checkpoint = ModelCheckpoint(
            self.model_directory,
            n_saved=2,
            filename_prefix=f"{self.model_type}_{self.label_column}_{self.train_period}_",
            score_function=score_function,
            score_name="accuracy",
            global_step_transform=global_step_from_engine(
                train_engine),  # helps fetch the trainer's state
        )

        # Early stopping
        def early_stopping_score_function(engine):
            val_loss = engine.state.metrics[f'loss']
            return -val_loss

        stopper_handler = EarlyStopping(patience=early_stopping_patience,
                                        score_function=early_stopping_score_function,
                                        trainer=train_engine)
        val_evaluator.add_event_handler(Events.COMPLETED, stopper_handler)

        # Save the model after every epoch of val_evaluator is completed
        val_evaluator.add_event_handler(
            Events.COMPLETED, model_checkpoint, {"model": model})

        print("Training model")
        training_progress_bar = ProgressBar()
        training_progress_bar.attach(
            train_engine, output_transform=lambda x: {"loss": x})

        val_progress_bar = ProgressBar()
        val_progress_bar.attach(
            val_evaluator, output_transform=lambda x: x)

        with profiler.profile(activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA
        ]
        ) as prof:
            train_engine.run(train_loader, max_epochs=num_epochs)

        @ train_engine.on(Events.ITERATION_COMPLETED)
        def log_iteration(engine):
            print(prof.key_averages().table(sort_by="cpu_time_total"))

        return model

    def train_model(self, train_dataframe: torch.utils.data.Subset, training_predictions_file):
        """ Train the model.

        Parameters
        ----------
        train_dataframe : torch.utils.data.Subset
            The training dataframe
        training_predictions_file : str
            The file to save the training predictions to
        label_mapping : Dict[str, int]
            The label mapping
        device : torch.device
            The device to use
        learning_rate : float
            The learning rate
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        early_stopping_patience : int
            The early stopping patience
        """
        device = torch.device(
            self.gpu_device if torch.cuda.is_available() else "cpu")
        self.model_trainer.fit(train_dataframe,
                               self.fit_model,
                               training_predictions_file,
                               label_mapping=self.train_label_mapping,
                               device=device,
                               learning_rate=self.learning_rate,
                               num_epochs=self.number_of_epochs,
                               batch_size=self.batch_size,
                               early_stopping_patience=3
                               )

    def run(self):
        """ Run the task.
        """
        print(
            f"Training model: {self.model_type} on period: {self.train_period}")

        # if self.feature_extractor in self.train_from_scratch:
        #     raise NotImplementedError(
        #         "Training feature extractor from scratch not implemented")
        super().run()

        self.write_per_period_evaluations()

    def write_per_period_evaluations(self):
        """ Write the per period evaluations.
        """
        evaluation_df = pd.read_parquet(
            self.predictions_filename, engine=self.parquet_engine)
        self.write_grouped_results(evaluation_df, group_columns=[
                                   self.period_column], results_key=self.per_period_evaluation_key)
        self.write_grouped_results(evaluation_df, group_columns=[
                                   self.period_column, self.label_column], results_key=self.per_period_coicop_evaluation_key)

    def write_grouped_results(self, evaluation_df, group_columns, results_key):
        """ Write the grouped results.

        Parameters
        ----------
        evaluation_df : pd.DataFrame
            The evaluation dataframe
        group_columns : list[str]
            The columns to group by
        results_key : str
            The key to save the results to
        """
        per_group_evaluations = calculate_metrics_per_group(dataframe=evaluation_df,
                                                            group_columns=group_columns,
                                                            label_column=self.label_column,
                                                            prediction_column="y_pred")
        with self.output()[results_key].open("w") as per_group_evaluation_file:
            per_group_evaluations.to_parquet(
                per_group_evaluation_file, engine=self.parquet_engine)

    def write_product_evaluations(self):
        """ Write the product evaluations.
        """
        evaluation_df = pd.read_parquet(
            self.predictions_filename, engine=self.parquet_engine)
        products_df = compare_products_per_period(
            evaluation_df, self.period_column, product_columns=[self.receipt_text_column])

        evaluation_df = evaluation_df.merge(products_df, on=self.period_column)
        evaluation_df["is_new_product"] = evaluation_df.apply(
            lambda row: row[self.receipt_text_column] in row["{self.receipt_text}_introduced"], axis=1)
        evaluation_df["is_discontinued_product"] = evaluation_df.apply(
            lambda row: row[self.receipt_text_column] in row["{self.receipt_text}_removed"], axis=1)
        evaluation_df["unchanged_product"] = evaluation_df.apply(
            lambda row: row[self.receipt_text_column] in row["{self.receipt_text}_same"], axis=1)

        new_product_evaluations = calculate_metrics_per_group(dataframe=evaluation_df,
                                                              group_columns=[
                                                                  self.period_column, "is_new_product"],
                                                              label_column=self.label_column,
                                                              prediction_column="y_pred")
        discontinued_product_evaluations = calculate_metrics_per_group(dataframe=evaluation_df,
                                                                       group_columns=[
                                                                           self.period_column, "is_discontinued_product"],
                                                                       label_column=self.label_column,
                                                                       prediction_column="y_pred")
        unchanged_product_evaluations = calculate_metrics_per_group(dataframe=evaluation_df,
                                                                    group_columns=[
                                                                        self.period_column, "unchanged_product"],
                                                                    label_column=self.label_column,
                                                                    prediction_column="y_pred")
        total_product_evaluations = calculate_metrics_per_group(dataframe=evaluation_df,
                                                                group_columns=[
                                                                    self.period_column],
                                                                label_column=self.label_column,
                                                                prediction_column="y_pred")


class TrainModelOnAllPeriods(luigi.WrapperTask):
    """ Train a model on all periods.
    """
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    feature_extractor = luigi.EnumParameter(enum=FeatureExtractorType)
    model_type = luigi.Parameter()

    filename_prefix = luigi.Parameter()
    label_column = luigi.Parameter()
    receipt_text_column = luigi.Parameter()
    features_column = luigi.Parameter(default="features")
    batch_size = luigi.IntParameter(default=30000)
    number_of_epochs = luigi.IntParameter(default=10)
    parquet_engine = luigi.Parameter()
    verbose = luigi.BoolParameter(default=False)
    period_columns = luigi.ListParameter()

    def identify_unique_periods(self, input_filename: str, period_column: str) -> pd.Series:
        """ Identify the unique periods.

        Parameters
        ----------
        input_filename : str
            The input filename
        period_column : str
            The period column

        Returns
        -------
        pd.Series
            The unique periods
        """
        print(
            f"Identifying unique periods for column {period_column} in {input_filename}")

        return pa.parquet.read_table(input_filename, columns=[period_column]).to_pandas()[period_column].unique()

    def requires(self):
        """ Get the required tasks.

        Returns
        -------
        list[TrainModelOnPeriod]
            The required tasks
        """
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
                for period in self.identify_unique_periods(os.path.join(self.input_directory, feature_filename), period_column)
                ]
