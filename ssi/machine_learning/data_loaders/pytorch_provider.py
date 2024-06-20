from typing import List, Tuple, Dict, Optional
from bisect import bisect_right
import torch.nn as nn
import torch
import pyarrow.parquet as pq
from ..model import Model
from .label_encoder import DataLabelEncoder
from .data_provider import DataProvider
from skorch import NeuralNetClassifier
from torch.nn import functional as F
import pandas as pd
import numpy as np


class ParquetDataset(torch.utils.data.Dataset, DataProvider):
    """ This class is a PyTorch Dataset specifically designed to read Parquet files.
    The class reads the Parquet file in batches and returns the data in the form of a PyTorch tensor.
    """
    # TODO class if not thread safe yet

    def __init__(self, filename: str,
                 features_column: str,
                 label_column: str,
                 label_mapping: Dict[str, int],
                 memory_map: bool = False):
        super().__init__(features_column, label_column, DataLabelEncoder(label_mapping))
        self.__parquet_file = pq.ParquetFile(filename, memory_map=memory_map)
        self.__label_mapping = label_mapping
        self.__current_row_group_index = 0
        self.__current_row_group = None
        self.__cumulative_row_index = None

    @property
    def cumulative_row_index(self) -> List[int]:
        if not self.__cumulative_row_index:
            self.__cumulative_row_index = [0]
            for row_group_index in range(self.number_of_row_groups):
                number_of_rows = self.number_of_rows_in_row_group(
                    row_group_index)
                self.__cumulative_row_index.append(
                    self.__cumulative_row_index[-1] + number_of_rows)
        return self.__cumulative_row_index

    @property
    def parquet_file(self):
        return self.__parquet_file

    @property
    def number_of_row_groups(self) -> int:
        return self.parquet_file.num_row_groups

    @property
    def current_row_group_index(self) -> int:
        return self.__current_row_group_index

    @current_row_group_index.setter
    def current_row_group_index(self, value: int):
        self.__current_row_group_index = value

    @property
    def current_row_group(self) -> pd.DataFrame:
        return self.__current_row_group

    @current_row_group.setter
    def current_row_group(self, value: pd.DataFrame):
        self.__current_row_group = value

    @property
    def label_mapping(self) -> Dict[str, int]:
        """ Returns a label mapping used to encode the target column.
        The labels are string values and need to be converted into integer
         labels that can be used by the machine learning algorithms. 

        Returns
        -------
        Dict[str, int]
            The label mapping used to encode the target column.
        """
        return self.__label_mapping

    @property
    def feature_vector_size(self) -> int:
        """ Returns the size of the feature vector. The size of the feature vector
        is determined by the length of the first feature vector in the feature
        vector column in the Parquet file. We assume that all feature vectors in
        the ParquetFile have the same length.

        Returns
        -------
        int
            The size of the feature vector.
        """
        feature_df = self.parquet_file.read_row_group(0,
                                                      columns=[self.features_column]).to_pandas()
        return len(feature_df[self.features_column].iloc[0])

    def number_of_rows_in_row_group(self, row_group_index: int) -> int:
        """ Get the number of rows in a row group. The number of rows
        in a row group is determined from the metadata of the Parquet file.

        Parameters
        ----------
        row_group_index : int
            The index of the row group to get the number of rows for.

        Returns
        -------
        int
            The number of rows in the row group.
        """
        return self.parquet_file.metadata.row_group(row_group_index).num_rows

    def get_row_group_for_index(self, index: int) -> Tuple[int, int]:
        """ Get the row group index and the index within the row group for a given index.
        To get the row group index, we first create a cumulative row index list in the
        cumulative_row_index property. This list is created only once and then cached.
        After that, we use a binary search to find the row group index for a given index.
        The binary search is performed by using the bisect_right function from the bisect module.
        The bisect_right function returns the row group index for the given index. The index within
        the row group is calculated by subtracting the cumulative number of rows in the
        previous row groups from the index passed as parameter.

        Parameters
        ----------
        index : int
            The index for which we want to find the row group index and index within the row group for.

        Returns
        -------

        Tuple[int, int]
            A tuple containing the row group index and the index within the row group.
        """
        row_group_index = bisect_right(self.cumulative_row_index, index) - 1
        if row_group_index < 0 or row_group_index >= self.number_of_row_groups:
            raise ValueError("Index out of bounds")
        index_within_row_group = index - \
            self.cumulative_row_index[row_group_index]
        return row_group_index, index_within_row_group

    def get_data_for_row_group(self, row_group_index: int) -> pd.DataFrame:
        """ Get the data for a row group. If the data is already cached, return the cached data.
        Otherwise, read the data from the Parquet file and cache it.

        Parameters
        ----------
        row_group_index : int
            The index of the row group to read the data from.

        Returns
        -------
        pd.DataFrame
            The data for the row group.
        """
        if self.__current_row_group is None or row_group_index != self.current_row_group_index:
            self.current_row_group_index = row_group_index
            row_group = self.parquet_file.read_row_group(
                self.current_row_group_index)
            self.current_row_group = row_group.to_pandas()

        return self.current_row_group

    def __len__(self):
        return self.parquet_file.metadata.num_rows

    def process_sample(self, sample: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Process a sample of rows from the Parquet file. The process_sample method
        takes the feature_column and the label_column from the sample and converts them
        into PyTorch tensors. The feature_column is converted into a PyTorch tensor of
        type float32. The label_column is mapped into an integer index using the set 
        label_mapping.

        Parameters
        ----------
        sample : pd.DataFrame
            The sample to process.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]
            A tuple containing the feature tensor, the label tensor, and the additional columns.
        """
        feature_tensor = torch.tensor(
            sample[self.features_column], dtype=torch.float32)

        label_vector = sample[self.label_column]
        mapped_label = self.label_encoder.transform(label_vector)

        label_tensor = torch.tensor(
            mapped_label, dtype=torch.long)

        additional_columns = sample.drop(
            labels=[self.features_column, self.label_column])

        return feature_tensor, label_tensor, additional_columns

    def get_column(self, column_name: str) -> pd.Series:
        return self.parquet_file.read(columns=[column_name]).to_pandas()

    def __getitem__(self, index):
        row_group_index, index_in_row_group = self.get_row_group_for_index(
            index)
        dataframe = self.get_data_for_row_group(row_group_index)
        sample = dataframe.iloc[index_in_row_group]
        return self.process_sample(sample)

    def __getitems__(self, indices):
        # TODO training is really slow, see if PyTorch has some profiling
        # TODO see if selection of period makes training already faster
        # Return indices smaller than max index
        indices = [index for index in indices if index < len(self)]

        # Get the sort order of the idx array
        # In order retrieval of the data will be more efficient because we can use
        # the cache of the row group
        indices = np.array(indices)
        sorting_indices = np.argsort(indices)
        # Create a dictionary to map the original index to the sorted index
        sort_dict = {original_index: sorted_index
                     for original_index, sorted_index in zip(indices, sorting_indices)}

        # Order the idx array
        sorted_idx = indices[sorting_indices]
        # Get the data for the sorted index
        # TODO see if we can optimize this a bit by retrieving the full batches.
        items = [self.__getitem__(index) for index in sorted_idx]

        # Sort the items back to the original order
        return [items[sort_dict[original_index]] for original_index in indices]

    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None):
        # TODO implement this
        # self.fit_or_refit_labels(original_label_encoder)
        raise NotImplementedError("Method not implemented")


class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        prediction = self.linear(x)
        return prediction

    def predict(self, x):
        y_proba = self.predict_proba(x)
        return y_proba.argmax(axis=1)  # dim=1, keepdim=True)

    def predict_proba(self, x):
        return F.softmax(self.forward(torch.from_numpy(x).to("cuda:0")), dim=1).cpu().detach().numpy()


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(
            in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def predict(self, x):
        y_proba = self.predict_proba(x)
        return y_proba.argmax(axis=1)

    def predict_proba(self, x):
        return F.softmax(self.forward(torch.from_numpy(x).to("cuda:0")), dim=1).cpu().detach().numpy()


class PytorchModel(Model):
    def __init__(self,
                 model):
        super().__init__(model)
        self.__classifier = None

    @ property
    def classifier(self):
        return self.__classifier

    @ classifier.setter
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
