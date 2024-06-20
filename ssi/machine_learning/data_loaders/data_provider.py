from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from .label_encoder import DataLabelEncoder
import pandas as pd
import enum


class DataProvider:
    def __init__(self,
                 features_column: str,
                 label_column: str,
                 label_encoder: DataLabelEncoder = DataLabelEncoder()
                 ):
        self.__features_column = features_column
        self.__label_column = label_column
        self.__label_encoder = label_encoder

    @property
    def features_column(self) -> str:
        """ Returns the name of the feature column for this DataProvider.

        Returns
        -------
        str
            The name of the feature column for this DataProvider.
        """
        return self.__features_column

    @property
    def label_column(self) -> str:
        """ Returns the name of the label column for this DataProvider.

        Returns
        -------
        str
            The name of the label column for this DataProvider.
        """
        return self.__label_column

    @property
    def encoded_label_column(self) -> str:
        """ Returns the name of the encoded label column for this DataProvider.
        The encoded label column is the label column with the labels encoded as integers.

        Returns
        -------
        str
            The name of the encoded label column for this DataProvider.
            The name is derived from the label column name by appending "_index".
        """
        return f"{self.label_column}_index"

    @property
    @abstractmethod
    def feature_vector_size(self) -> int:
        """ Returns the size of the feature vector.
        This can be useful to determine the input size of a neural network.

        Returns
        -------
        int
            The size of the feature vector.
        """
        pass

    @property
    @abstractmethod
    def labels(self) -> pd.Series:
        """ Returns all labels from the label column.

        Returns
        -------
        pd.Series
            A pandas Series object with all labels from the label column.
        """
        return self[self.label_column]

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
        return len(self.label_encoder.label_mapping)

    @property
    def label_encoder(self) -> DataLabelEncoder:
        """ Returns the label encoder for this DataProvider.

        Returns
        -------
        DataLabelEncoder
            The label encoder for this DataProvider.
        """
        return self.__label_encoder

    @label_encoder.setter
    def label_encoder(self, value: DataLabelEncoder):
        """ Sets the label encoder for this DataProvider.

        Parameters

        value : DataLabelEncoder
            The label encoder to set for this DataProvider.
        """
        self.__label_encoder = value

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, key: Union[int, str]) -> pd.Series:
        if isinstance(key, int):
            return self.get_column(key)
        else:
            return self.get_item(key)

    @abstractmethod
    def __getitems__(self, indices):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass

    @abstractmethod
    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None) -> 'DataProvider':
        pass

    @abstractmethod
    def get_column(self, column_name: str) -> pd.Series:
        pass

    @abstractmethod
    def get_item(self, index: int):
        pass

    def fit_or_refit_labels(self, original_label_encoder: Optional[DataLabelEncoder]):
        if original_label_encoder:
            self.label_encoder = original_label_encoder.refit(self.labels)
        else:
            self.label_encoder = self.label_encoder.fit(self.labels)

    def train_test_split(self, test_size: float) -> Tuple['DataProvider', 'DataProvider']:
        train_indices, test_indices = train_test_split(
            len(self), test_size=test_size, stratify=self.labels)

        train_data = self.get_subset(train_indices)
        test_data = self.get_subset(test_indices, train_data.label_encoder)

        return train_data, test_data
