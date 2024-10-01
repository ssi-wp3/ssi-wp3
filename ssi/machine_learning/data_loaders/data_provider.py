from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from .label_encoder import DataLabelEncoder
import pandas as pd
import enum


class DataProvider:
    """ A DataProvider is responsible for providing data to a machine learning model.
    It can load data from a file, split it into a training and test set, and provide
    batches of data for training.
    """

    def __init__(self,
                 filename: str,
                 features_column: str,
                 label_column: str,
                 label_encoder: DataLabelEncoder = DataLabelEncoder()
                 ):
        """ Initialize the DataProvider.

        Parameters
        ----------
        filename : str
            The filename of the data source.
        features_column : str
            The name of the feature column.
        label_column : str
            The name of the label column.
        label_encoder : DataLabelEncoder, optional
            The label encoder to use, by default a new DataLabelEncoder is created.
        """
        self.__filename = filename
        self.__features_column = features_column
        self.__label_column = label_column
        self.__label_encoder = label_encoder

    @property
    def filename(self) -> str:
        """ Returns the filename of the data source for this DataProvider.

        Returns
        -------
        str
            The filename of the data source for this DataProvider.
        """
        return self.__filename

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
    def labels(self) -> pd.Series:
        """ Returns all labels from the label column.

        Returns
        -------
        pd.Series
            A pandas Series object with all labels from the label column.
        """
        return self[self.label_column]

    @property
    def features(self) -> pd.Series:
        """ Returns all features from the feature column.

        Returns
        -------
        pd.Series
            A pandas Series object with all features from the feature column.
        """
        return self[self.features_column]

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
        """ Returns the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        pass

    def __getitem__(self, key: Union[int, str]) -> pd.Series:
        """ Returns a column or an item from the dataset.

        Parameters
        ----------
        key : Union[int, str]
            The index or the name of the column to return.

        Returns
        -------
        pd.Series
            The column or item at the given index or name.
        """
        if isinstance(key, int):
            return self.get_column(key)
        else:
            return self.get_item(key)

    @abstractmethod
    def __getitems__(self, indices):
        """ Returns a subset of the dataset.

        Parameters
        ----------
        indices : pd.Series
            The indices of the samples to return.

        Returns
        -------
        DataProvider
            A new DataProvider object with the specified samples.
        """
        pass

    @abstractmethod
    def load(self):
        """ Loads the data from the file 
        """
        pass

    @abstractmethod
    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None) -> 'DataProvider':
        """ Returns a subset of the dataset.

        Parameters
        ----------
        indices : pd.Series
            The indices of the samples to return.
        original_label_encoder : DataLabelEncoder, optional
            The original label encoder to use, by default None.

        Returns
        -------
        DataProvider
            A new DataProvider object with the specified samples.
        """
        pass

    @abstractmethod
    def get_column(self, column_name: str) -> pd.Series:
        """ Returns a column from the dataset.

        Parameters
        ----------
        column_name : str
            The name of the column to return.

        Returns
        -------
        pd.Series
        """
        pass

    @abstractmethod
    def get_item(self, index: int):
        """ Returns an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item to return.

        Returns
        -------
        """
        pass

    @abstractmethod
    def drop_duplicates(self, subset=None, keep: str = 'first') -> 'DataProvider':
        """ Drops duplicate rows from the dataset.

        Parameters
        ----------
        subset : str or list of str, optional
            The column or columns to use for identifying duplicates, by default None.
        keep : {'first', 'last', 'False'}, optional
            The method to use for identifying duplicates, by default 'first'.

        Returns
        -------
        DataProvider
            A new DataProvider object with the duplicate rows removed.
        """
        pass

    def fit_or_refit_labels(self, original_label_encoder: Optional[DataLabelEncoder]):
        """ Fits or refits the label encoder to the labels in the dataset.
        If the original label encoder is provided, the label encoder is refitted to the labels in the dataset.
        Otherwise, the label encoder is fitted to the labels in the dataset.

        Parameters
        ----------
        original_label_encoder : DataLabelEncoder, optional
            The original label encoder to use, by default None.
        """
        if original_label_encoder:
            self.label_encoder = original_label_encoder.refit(self.labels)
        else:
            self.label_encoder = self.label_encoder.fit(self.labels)

    def train_test_split(self, test_size: float = 0.2) -> Tuple['DataProvider', 'DataProvider']:
        """ Splits the data into a training and a test set, the test size is determined by the
        test_size parameter. The data is split using the train_test_split method from scikit-learn.

        Parameters
        ----------
        test_size : float, optional
            The size of the test set, by default 0.2

        Returns
        -------
        Tuple[DataProvider, DataProvider]
            A tuple containing the training and test DataProviders.
        """
        train_indices, test_indices = train_test_split(
            len(self), test_size=test_size, stratify=self.labels)

        train_data = self.get_subset(train_indices)
        test_data = self.get_subset(test_indices, train_data.label_encoder)

        return train_data, test_data

    def train_validation_test_split(self, validation_size: float = 0.1, test_size: float = 0.2) -> Tuple['DataProvider', 'DataProvider', 'DataProvider']:
        """ Splits the data into training, validation, and test sets.
        The data is first split into a training validation set and a test set, using the test_size parameter.
        After that, the training validation set is split into a training set and a validation set, using the validation_size parameter. To split the data, the train_test_split method is used.

        Parameters
        ----------
        validation_size : float, optional
            The size of the validation set, by default 0.1

        test_size : float, optional
            The size of the test set, by default 0.2

        Returns
        -------

        Tuple[DataProvider, DataProvider, DataProvider]
            A tuple containing the training, validation, and test DataProviders.
        """
        train_validation_provider, test_provider = self.train_test_split(
            test_size)
        train_provider, validation_provider = train_validation_provider.train_test_split(
            validation_size)
        return train_provider, validation_provider, test_provider
