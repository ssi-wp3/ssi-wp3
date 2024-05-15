from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import pandas as pd


class DataLabelEncoder:
    def init(self, label_mapping: OrderedDict = None):
        self.__label_mapping = label_mapping

    @property
    def label_mapping(self) -> OrderedDict:
        return self.__label_mapping

    @label_mapping.setter
    def label_mapping(self, value: OrderedDict):
        self.__label_mapping = value

    @property
    def inverse_label_mapping(self) -> OrderedDict:
        return OrderedDict([(v, k) for k, v in self.label_mapping.items()])

    def fit(self, labels: pd.Series):
        """ Fit the label encoder to the labels.

        Parameters
        ----------

        labels : pd.Series
            The labels to encode.
        """
        self.train_label_mapping = OrderedDict([(original_label, index)
                                                for index, original_label in enumerate(labels.unique())])

    def refit(self, extra_labels: pd.Series) -> 'DataLabelEncoder':
        """ Refit the label encoder to the labels while preserving the original label->index mapping.
        Sometimes the test dataset can have more categories than the training dataset, add them add the end of the mapping.

        Parameters
        ----------
        extra_labels : pd.Series
            The extra labels to encode.

        Returns
        -------
        DataLabelEncoder
            The refitted label encoder.        
        """
        # Test dataset can have more categories than the training dataset, add them add the end of the mapping
        # In this way, we preserve the original label->index mapping for the training dataset
        self.new_label_mapping = self.label_mapping
        for label in extra_labels.unique():
            if label not in self.new_label_mapping:
                self.new_label_mapping[label] = len(self.new_label_mapping)
        return DataLabelEncoder(label_mapping=self.new_label_mapping)

    def transform(self, labels: pd.Series) -> pd.Series:
        """ Transform the labels to their encoded values.

        Parameters
        ----------
        labels : pd.Series
            The labels to encode.

        Returns
        -------
        pd.Series
            The encoded labels.
        """
        return labels.map(self.label_mapping)

    def fit_transform(self, labels: pd.Series) -> pd.Series:
        """Fit and transform the labels to their encoded values.

        Parameters
        ----------
        labels : pd.Series
            The labels to encode.

        Returns
        -------
        pd.Series
            The encoded labels.
        """
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, encoded_labels: pd.Series) -> pd.Series:
        """Inverse transform the labels back to their original values.

        Parameters
        ----------
        encoded_labels : pd.Series
            The encoded labels.

        Returns
        -------
        pd.Series
            The original labels.
        """
        return encoded_labels.map(self.inverse_label_mapping)


class DataProvider:
    def __init__(self,
                 features_column: str,
                 label_column: str,
                 label_encoder: DataLabelEncoder
                 ):
        self.__features_column = features_column
        self.__label_column = label_column
        self.__train_label_encoder = label_encoder
        self.__test_label_encoder = None

    @property
    def features_column(self) -> str:
        return self.__features_column

    @property
    def label_column(self) -> str:
        return self.__label_column

    @property
    def train_label_encoder(self) -> DataLabelEncoder:
        return self.__train_label_encoder

    @train_label_encoder.setter
    def train_label_encoder(self, value: DataLabelEncoder):
        self.__train_label_encoder = value

    @property
    def test_label_encoder(self) -> DataLabelEncoder:
        return self.__test_label_encoder

    @test_label_encoder.setter
    def test_label_encoder(self, value: DataLabelEncoder):
        self.__test_label_encoder = value

    @abstractmethod
    def load(self, filename: str):
        pass

    @abstractmethod
    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class DataframeDataProvider(DataProvider):
    def init(self,
             features_column: str,
             label_column: str,
             label_encoder: DataLabelEncoder,
             parquet_engine: str = "pyarrow"
             ):
        super().init(features_column, label_column, label_encoder)
        self.__parquet_engine = parquet_engine

    @property
    def parquet_engine(self) -> str:
        return self.__parquet_engine

    def load(self, filename: str) -> pd.DataFrame:
        return pd.read_parquet(filename, engine=self.parquet_engine)

    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            dataframe, test_size=test_size, stratify=dataframe[self.label_column])

        self.train_label_encoder.fit(train_df[self.label_column])

        self.test_label_encoder = self.train_label_encoder.refit(
            test_df[self.label_column])

        train_df[f"{self.label_column}_index"] = train_df[self.label_column].map(
            self.train_label_encoder.transform)
        test_df[f"{self.label_column}_index"] = test_df[self.label_column].map(
            self.test_label_encoder.transform)
        return train_df, test_df


class PyTorchDataProvider(DataProvider):
    pass
