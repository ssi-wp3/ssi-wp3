import torch.nn as nn
import torch
import pyarrow.parquet as pq
from skorch import NeuralNetClassifier
from .model import Model
import pandas as pd
import numpy as np


class ParquetDataset(torch.utils.data.Dataset):
    """ This class is a PyTorch Dataset specifically designed to read Parquet files.
    The class reads the Parquet file in batches and returns the data in the form of a PyTorch tensor.
    """

    def __init__(self, filename: str, feature_column: str, target_column: str, memory_map: bool = False):
        self.__parquet_file = pq.ParquetFile(filename, memory_map=memory_map)
        self.__feature_column = feature_column
        self.__target_column = target_column

    @property
    def parquet_file(self):
        return self.__parquet_file

    @property
    def feature_column(self) -> str:
        return self.__feature_column

    @property
    def target_column(self) -> str:
        return self.__target_column

    @property
    def number_of_row_groups(self) -> int:
        return self.parquet_file.num_row_groups

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
        row_group_index, index_in_row_group = self.get_row_group_for_index(
            index)
        dataframe = self.get_data_for_row_group(row_group_index)
        sample = dataframe.iloc[index_in_row_group]
        feature_tensor = torch.tensor(
            sample[self.feature_column], dtype=torch.float32)
        label_tensor = torch.tensor(
            int(sample[self.target_column]), dtype=torch.long)

        print(
            f"Feature tensor: {feature_tensor.shape}, Label tensor: {label_tensor}")
        return feature_tensor, label_tensor


class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


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
