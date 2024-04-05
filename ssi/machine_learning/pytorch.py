from typing import List, Tuple
import torch.nn as nn
import torch
import pyarrow.parquet as pq
from skorch import NeuralNetClassifier
from sklearn.preprocessing import LabelEncoder
from .model import Model
from torch.nn import functional as F
from torch.multiprocessing import Queue
import pandas as pd
import numpy as np


class ParquetDataset(torch.utils.data.Dataset):
    """ This class is a PyTorch Dataset specifically designed to read Parquet files.
    The class reads the Parquet file in batches and returns the data in the form of a PyTorch tensor.
    """

    def __init__(self, filename: str,
                 feature_column: str,
                 target_column: str,
                 batch_size: int,
                 filters: List[Tuple[str]],
                 memory_map: bool = False):
        super().__init__()
        # TODO revert back to ParquetFile
        # self.__parquet_file = pq.ParquetFile(filename, memory_map=memory_map)
        self.__parquet_file = pq.read_table(
            filename,
            columns=[feature_column, target_column],
            memory_map=memory_map,
            filters=filters)
        self.__feature_column = feature_column
        self.__target_column = target_column
        self.__label_encoder = self._fit_label_encoder(self.parquet_file)

    @property
    def parquet_file(self):
        return self.__parquet_file

    @property
    def number_of_row_groups(self) -> int:
        return self.parquet_file.num_row_groups

    @property
    def feature_column(self) -> str:
        return self.__feature_column

    @property
    def target_column(self) -> str:
        return self.__target_column

    @property
    def feature_vector_size(self) -> int:
        # TODO read from file. This is hardcoded for now
        # feature_df = self.parquet_file.take(
        #    0).to_pandas()

        # return len(feature_df[self.feature_column].iloc[0])

        # TODO hardcoded this for now
        return 768

    @property
    def number_of_classes(self) -> int:
        return len(self.label_encoder.classes_)

    @property
    def label_encoder(self) -> LabelEncoder:
        return self.__label_encoder

    def _fit_label_encoder(self, parquet_file) -> LabelEncoder:
        label_df = parquet_file.select(
            columns=[self.target_column]).to_pandas()
        label_encoder = LabelEncoder()
        label_encoder.fit(label_df[self.target_column])
        return label_encoder

    def number_of_rows_in_row_group(self, row_group_index: int) -> int:
        return self.parquet_file.metadata.row_group(row_group_index).num_rows

    def get_row_group_for_index(self, index: int) -> int:
        previous_index = 0
        for row_group_index in range(self.number_of_row_groups):
            number_of_rows = self.number_of_rows_in_row_group(row_group_index)
            previous_index = index
            index -= number_of_rows
            if index < 0:
                return row_group_index, previous_index
        raise ValueError("Index out of bounds")

    def get_data_for_row_group(self, row_group_index: int) -> pd.DataFrame:
        row_group = self.parquet_file.read_row_group(row_group_index)
        return row_group.to_pandas()

    def __len__(self):
        return self.parquet_file.metadata.num_rows

    def __getitem__(self, index):
        # Cache the row group
        row_group_index, index_in_row_group = self.get_row_group_for_index(
            index)
        dataframe = self.get_data_for_row_group(row_group_index)
        sample = dataframe.iloc[index_in_row_group]
        return self.process_sample(sample)

    def __getitems__(self, idx):
        # Get the sort order of the idx array
        # In order retrieval of the data will be more efficient because we can use
        # the cache of the row group
        sorting_indices = np.argsort(idx)
        # Create a dictionary to map the original index to the sorted index
        sort_dict = {original_index: sorted_index
                     for original_index, sorted_index in zip(idx, sorting_indices)}

        # Order the idx array
        sorted_idx = idx[sorting_indices]
        # Get the data for the sorted index
        items = [self.__getitem__(index) for index in sorted_idx]

        # Sort the items back to the original order
        return [items[sort_dict[original_index]] for original_index in idx]

    def process_sample(self, sample: pd.DataFrame):
        feature_tensor = torch.tensor(
            sample[self.feature_column], dtype=torch.float32)

        label_tensor = torch.tensor(
            self.label_encoder.transform([sample[self.target_column]]), dtype=torch.long)
        one_hot_label = F.one_hot(label_tensor, num_classes=len(
            self.label_encoder.classes_)).float()

        return feature_tensor, one_hot_label


class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        prediction = self.linear(x)
        print("Prediction: ", prediction)
        return prediction


class PytorchModel(Model):
    def __init__(self,
                 model):
        super().__init__(model)
        self.__classifier = None

    @property
    def classifier(self):
        return self.__classifier

    @classifier.setter
    def classifier(self, value):
        self.__classifier = value

    # TODO: this should use fit(dataset) instead of fit(X, y)
    # See: https://skorch.readthedocs.io/en/stable/user/FAQ.html#faq-how-do-i-use-a-pytorch-dataset-with-skorch
    def fit(self, X, y,
            max_epochs: int,
            batch_size: int,
            lr: float,
            test_size: float,
            iterator_train__shuffle: bool = True,
            **kwargs):
        self.classifier = NeuralNetClassifier(
            self.model,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            train_split=test_size,
            iterator_train__shuffle=iterator_train__shuffle,
            **kwargs
        )
        self.classifier.fit(X, y)
        return self

    def fit(self,
            dataset: ParquetDataset,
            max_epochs: int,
            batch_size: int,
            lr: float,
            test_size: float,
            iterator_train__shuffle: bool = True,
            **kwargs):
        self.classifier = NeuralNetClassifier(
            self.model,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            train_split=test_size,
            iterator_train__shuffle=iterator_train__shuffle,
            **kwargs
        )
        self.classifier.fit(dataset)
        return self

    def predict(self, X):
        self._check_classifier_trained()
        return self.classifier.predict(X)

    def predict_proba(self, X):
        self._check_classifier_trained()
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        self._check_classifier_trained()
        return self.classifier.score(X, y)

    def _check_classifier_trained(self):
        if not self.classifier:
            raise ValueError(
                "Cannot predict without fitting the model first. Call the fit method first.")

    def load_data(self, filename: str, **kwargs) -> ParquetDataset:
        return ParquetDataset(filename, **kwargs)
