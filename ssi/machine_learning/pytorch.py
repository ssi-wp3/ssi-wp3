from typing import List, Tuple
from bisect import bisect_right
import torch.nn as nn
import torch
import pyarrow.parquet as pq
from skorch import NeuralNetClassifier
from sklearn.preprocessing import LabelEncoder
from .model import Model
from torch.nn import functional as F
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
        self.__parquet_file = pq.ParquetFile(filename, memory_map=memory_map)
        # self.__parquet_file = pq.read_table(
        #     filename,
        #     columns=[feature_column, target_column],
        #     memory_map=memory_map,
        #     filters=filters)
        self.__feature_column = feature_column
        self.__target_column = target_column
        self.__label_encoder = self._fit_label_encoder(self.parquet_file)
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
    def feature_column(self) -> str:
        """ Returns the name of the feature column in the Parquet file.

        Returns
        -------
        str
            The name of the feature column in the Parquet file.
        """
        return self.__feature_column

    @property
    def target_column(self) -> str:
        """ Returns the name of the target column in the Parquet file.

        Returns
        -------
        str
            The name of the target column in the Parquet file.
        """
        return self.__target_column

    @property
    def label_encoder(self) -> LabelEncoder:
        """ Returns the label encoder used to encode the target column.
        The labels are string values and need to be converted into integer
         labels that can be used by the machine learning algorithms. We use 
        the LabelEncoder class from scikit-learn to perform this encoding.

        Labels are encoded when the ParquetDataset is created.

        Returns
        -------
        LabelEncoder
            The label encoder used to encode the target column.
        """
        return self.__label_encoder

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
        # TODO read from file. This is hardcoded for now
        # feature_df = self.parquet_file.take(
        #    0).to_pandas()

        # return len(feature_df[self.feature_column].iloc[0])

        # TODO hardcoded this for now
        return 768

    @property
    def number_of_classes(self) -> int:
        """ Returns the number of classes in the target column.
        The number of classes is determined by the number of 
        unique classes determined by the LabelEncoder.

        Returns
        -------
        int
            The number of classes in the target column.
        """
        return len(self.label_encoder.classes_)

    def _fit_label_encoder(self, parquet_file: pq.ParquetFile) -> LabelEncoder:
        """ Fit a label encoder to the target column in the Parquet file.
        The label encoder is used to encode the target column into integer labels.
        Only the label column is read by this function.

        Parameters
        ----------
        parquet_file : pq.ParquetFile
            The Parquet file to read the target column from.

        Returns
        -------
        LabelEncoder
            The label encoder fitted to the target column.
        """
        label_df = parquet_file.read(
            columns=[self.target_column]).to_pandas()
        label_encoder = LabelEncoder()
        label_encoder.fit(label_df[self.target_column])
        return label_encoder

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
        if not self.__current_row_group or row_group_index != self.current_row_group_index:
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
        type float32. The label_column is one-hot encoded. 

        Parameters
        ----------
        sample : pd.DataFrame
            The sample to process.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the feature tensor and the one-hot encoded label tensor.
        """
        feature_tensor = torch.tensor(
            sample[self.feature_column], dtype=torch.float32)

        label_tensor = torch.tensor(
            self.label_encoder.transform([sample[self.target_column]]), dtype=torch.long)
        one_hot_label = F.one_hot(label_tensor, num_classes=len(
            self.label_encoder.classes_)).float()

        return feature_tensor, one_hot_label

    def __getitem__(self, index):
        row_group_index, index_in_row_group = self.get_row_group_for_index(
            index)
        dataframe = self.get_data_for_row_group(row_group_index)
        sample = dataframe.iloc[index_in_row_group]
        return self.process_sample(sample)

    def __getitems__(self, indices):
        # Get the sort order of the idx array
        # In order retrieval of the data will be more efficient because we can use
        # the cache of the row group
        indices = np.array(indices)
        sorting_indices = np.argsort(indices)

        print("Indices: ", indices)
        print("Sorting indices: ", sorting_indices)

        # Create a dictionary to map the original index to the sorted index
        sort_dict = {original_index: sorted_index
                     for original_index, sorted_index in zip(indices, sorting_indices)}

        # Order the idx array
        sorted_idx = indices[sorting_indices]
        # Get the data for the sorted index
        items = [self.__getitem__(index) for index in sorted_idx]

        # Sort the items back to the original order
        return [items[sort_dict[original_index]] for original_index in indices]


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
