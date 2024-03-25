import torch as nn
import torch
import pyarrow.parquet as pq
from skorch import NeuralNetClassifier
from .model import Model


class ParquetDataset(nn.utils.data.Dataset):
    """ This class is a PyTorch Dataset specifically designed to read Parquet files.
    The class reads the Parquet file in batches and returns the data in the form of a PyTorch tensor.
    """

    def __init__(self, filename, feature_column: str, target_column: str):
        self.parquet = pq.ParquetFile(filename)
        self.columns = self.parquet.schema.names if not feature_column and not target_column else [
            feature_column, target_column]
        self.length = len(self.parquet)
        self.row_group_size = self.parquet.metadata.row_group(0).num_rows
        self.data = None
        self.current_row_group = -1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row_group = index // self.row_group_size
        row_within_group = index % self.row_group_size
        if row_group != self.current_row_group:
            self.data = self.parquet.read_row_group(
                row_group, columns=self.columns).to_pandas()
            self.current_row_group = row_group
        sample = self.data.iloc[row_within_group]
        return torch.tensor(sample.values, dtype=torch.float32)


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
            **kwargs):
        self.classifier = NeuralNetClassifier(
            self.model,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            train_split=test_size,
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
