from typing import List, Tuple, Dict, Optional
from bisect import bisect_right
import torch.nn as nn
import torch
import pyarrow.parquet as pq
from ..model import Model
from .label_encoder import DataLabelEncoder
from .data_provider import DataProvider
from torch.nn import functional as F
import pandas as pd
import numpy as np


class ParquetDataset(torch.utils.data.Dataset, DataProvider):
    """ This class is a PyTorch Dataset specifically designed to read Parquet files.
    The class reads the Parquet file in batches and returns the data in the form of a PyTorch tensor.
    """
    # TODO class if not thread safe yet

    def __init__(self,
                 filename: str,
                 features_column: str,
                 label_column: str,
                 label_mapping: Dict[str, int],
                 memory_map: bool = False):
        """ Initializes the ParquetDataset.

        Parameters
        ----------
        filename : str
            The filename of the Parquet file.
        features_column : str
            The name of the column containing the feature data.
        label_column : str
            The name of the column containing the label data.
        label_mapping : Dict[str, int]
            The mapping of the labels to their encoded values.
        memory_map : bool, optional
            Whether to memory map the Parquet file, by default False.
        """
        super().__init__(filename, features_column,
                         label_column, DataLabelEncoder(label_mapping))
        self.__parquet_file = pq.ParquetFile(filename, memory_map=memory_map)
        self.__label_mapping = label_mapping
        self.__current_row_group_index = 0
        self.__current_row_group = None
        self.__cumulative_row_index = None

    @property
    def cumulative_row_index(self) -> List[int]:
        """ The cumulative row index of the Parquet file.

        Returns
        -------
        List[int]
            The cumulative row index of the Parquet file.
        """
        if not self.__cumulative_row_index:
            self.__cumulative_row_index = [0]
            for row_group_index in range(self.number_of_row_groups):
                number_of_rows = self.number_of_rows_in_row_group(
                    row_group_index)
                self.__cumulative_row_index.append(
                    self.__cumulative_row_index[-1] + number_of_rows)
        return self.__cumulative_row_index

    @property
    def parquet_file(self) -> pq.ParquetFile:
        """ The Parquet file.

        Returns
        -------
        pq.ParquetFile
            The Parquet file.
        """
        return self.__parquet_file

    @property
    def number_of_row_groups(self) -> int:
        """ The number of row groups in the Parquet file.

        Returns
        -------
        int
            The number of row groups in the Parquet file.
        """
        return self.parquet_file.num_row_groups

    @property
    def current_row_group_index(self) -> int:
        """ The index of the current row group.

        Returns
        -------
        int
            The index of the current row group.
        """
        return self.__current_row_group_index

    @current_row_group_index.setter
    def current_row_group_index(self, value: int):
        self.__current_row_group_index = value

    @property
    def current_row_group(self) -> pd.DataFrame:
        """ The current row group.

        Returns
        -------
        pd.DataFrame
            The current row group.
        """
        return self.__current_row_group

    @current_row_group.setter
    def current_row_group(self, value: pd.DataFrame):
        """ Sets the current row group.

        Parameters
        ----------
        value : pd.DataFrame
            The current row group.
        """
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

    def __len__(self) -> int:
        """ The number of rows in the Parquet file.

        Returns
        -------
        int
            The number of rows in the Parquet file.
        """
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
        """ Get a column from the Parquet file.

        Parameters
        ----------
        column_name : str
            The name of the column to get.

        Returns
        -------
        pd.Series
            The column with the specified name.
        """
        return self.parquet_file.read(columns=[column_name]).to_pandas()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """ Get a sample from the Parquet file.

        Parameters
        ----------
        index : int
            The index of the sample to get.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]
            A tuple containing the feature tensor, the label tensor, and the additional columns.
        """
        row_group_index, index_in_row_group = self.get_row_group_for_index(
            index)
        dataframe = self.get_data_for_row_group(row_group_index)
        sample = dataframe.iloc[index_in_row_group]
        return self.process_sample(sample)

    def __getitems__(self, indices: pd.Series) -> List[Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]]:
        """ Get multiple samples from the Parquet file.

        Parameters
        ----------
        indices : pd.Series
            The indices of the samples to get.

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]]
            A list of tuples containing the feature tensor, the label tensor, and the additional columns.
        """
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

    def load(self):
        """ Load the data from the Parquet file. Does nothing because the data is already loaded in the constructor.
        """
        pass

    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None) -> DataProvider:
        """ Get a subset of the ParquetDataset.

        Parameters
        ----------
        indices : pd.Series
            The indices of the samples to get.
        original_label_encoder : DataLabelEncoder, optional
            The original label encoder to use, by default None.

        Returns
        -------
        DataProvider
            A new DataProvider object with the specified subset of the data.
        """
        subset_dataset = torch.utils.data.Subset(self, indices)
        # TODO figure out how to create a new ParquetDataset from the subset!!
        raise NotImplementedError("Method not implemented")


class TorchLogisticRegression(nn.Module):
    """ A logistic regression model implemented in PyTorch.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """ Initializes the logistic regression model.

        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        output_dim : int
            The dimension of the output data.
        """
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        """ Forward pass of the logistic regression model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output of the logistic regression model.
        """
        prediction = self.linear(x)
        return prediction

    def predict(self, x):
        """ Predict the class of the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The predicted class of the input data.
        """
        y_proba = self.predict_proba(x)
        return y_proba.argmax(axis=1)  # dim=1, keepdim=True)

    def predict_proba(self, x):
        """ Predict the probability of the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The probability of the input data.
        """
        return F.softmax(self.forward(torch.from_numpy(x).to("cuda:0")), dim=1).cpu().detach().numpy()


class TorchMLP(nn.Module):
    """ A multi-layer perceptron model implemented in PyTorch.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100):
        """ Initializes the multi-layer perceptron model.

        Parameters
        ----------
        input_dim : int
            The dimension of the input data.
        output_dim : int
            The dimension of the output data.
        hidden_dim : int, optional
            The dimension of the hidden layer, by default 100.
        """
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(
            in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        """ Forward pass of the multi-layer perceptron model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output of the multi-layer perceptron model.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def predict(self, x):
        """ Predict the class of the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The predicted class of the input data.
        """
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x):
        """ Predict the probability of the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The probability of the input data.
        """
        return F.softmax(self.forward(torch.from_numpy(x).to("cuda:0")), dim=1).cpu().detach().numpy()
