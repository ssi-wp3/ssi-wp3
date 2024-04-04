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


class ParquetDataset(torch.utils.data.IterableDataset):
    """ This class is a PyTorch Dataset specifically designed to read Parquet files.
    The class reads the Parquet file in batches and returns the data in the form of a PyTorch tensor.
    """

    def __init__(self, filename: str, feature_column: str, target_column: str, batch_size: int,  memory_map: bool = False):
        self.__parquet_file = pq.ParquetFile(filename, memory_map=memory_map)
        self.__feature_column = feature_column
        self.__target_column = target_column
        self.__label_encoder = self._fit_label_encoder(self.parquet_file)

        self.batches = Queue()
        [self.batches.put(batch)
         for batch in self.parquet_file.iter_batches(batch_size=batch_size)]

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
    def feature_vector_size(self) -> int:
        print("Column names:", self.parquet_file.schema.names)
        feature_column_index = self.parquet_file.schema.names.index(
            "element")

        return self.parquet_file.schema.column(feature_column_index).length

    @property
    def label_encoder(self) -> LabelEncoder:
        return self.__label_encoder

    def _fit_label_encoder(self, parquet_file) -> LabelEncoder:
        label_df = parquet_file.read(
            columns=[self.target_column]).to_pandas()
        label_encoder = LabelEncoder()
        label_encoder.fit(label_df[self.target_column])
        return label_encoder

    def __len__(self):
        return self.parquet_file.count_rows()

    def process_batch(self, batch: pd.DataFrame):
        feature_tensor = torch.tensor(
            batch[self.feature_column], dtype=torch.float32)

        label_tensor = torch.tensor(
            self.label_encoder.transform([batch[self.target_column]]), dtype=torch.long)
        one_hot_label = F.one_hot(label_tensor, num_classes=len(
            self.label_encoder.classes_)).float()

        return feature_tensor, one_hot_label

    def __iter__(self):
        while True:
            if self.batches.empty() == True:
                self.batches.close()
                break

            batch = self.batches.get().to_table().to_pandas()
            batch.update(self.process_batch(batch))
            yield batch


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
