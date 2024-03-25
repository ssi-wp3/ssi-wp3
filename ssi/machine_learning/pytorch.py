import torch as nn
import torch
import pyarrow.parquet as pq
from skorch import NeuralNetClassifier
from .model import Model
import pandas as pd


class ParquetDataset(nn.utils.data.Dataset):
    """ This class is a PyTorch Dataset specifically designed to read Parquet files.
    The class reads the Parquet file in batches and returns the data in the form of a PyTorch tensor.
    """

    def __init__(self, dataframe: pd.DataFrame, feature_column: str, target_column: str):
        self.__dataframe = dataframe
        self.__feature_column = feature_column
        self.__target_column = target_column

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__dataframe

    @property
    def feature_column(self) -> str:
        return self.__feature_column

    @property
    def target_column(self) -> str:
        return self.__target_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        sample = self.dataframe.iloc[index]
        return torch.tensor(sample[self.feature_column].values, dtype=torch.float32), torch.tensor(sample[self.target_column], dtype=torch.float32)

    @staticmethod
    def from_filename(filename: str, feature_column: str, target_column: str, engine: str = "pyarrow"):
        dataframe = pd.read_parquet(filename, engine=engine)
        return ParquetDataset(dataframe, feature_column, target_column)

    @staticmethod
    def from_dataframe(dataframe, feature_column: str, target_column: str):
        return ParquetDataset(dataframe, feature_column, target_column)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class PytorchModel(Model):
    def __init__(self,
                 model,
                 ):
        super().__init__(model)
        self.__classifier = None

    @property
    def classifier(self):
        return self.__classifier

    @classifier.setter
    def classifier(self, value):
        self.__classifier = value

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
