from typing import Tuple, Union
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from .label_encoder import DataLabelEncoder
import pandas as pd
import enum


class DataProvider:
    def __init__(self,
                 features_column: str,
                 label_column: str,
                 label_encoder: DataLabelEncoder
                 ):
        self.__features_column = features_column
        self.__label_column = label_column
        self.__label_encoder = label_encoder

    @property
    def features_column(self) -> str:
        return self.__features_column

    @property
    def label_column(self) -> str:
        return self.__label_column

    @property
    def encoded_label_column(self) -> str:
        return f"{self.label_column}_index"

    @property
    @abstractmethod
    def feature_vector_length(self) -> int:
        pass

    @property
    @abstractmethod
    def labels(self) -> pd.Series:
        return self[self.label_column]

    @property
    def number_of_classes(self) -> int:
        return len(self.label_encoder.label_mapping)

    @property
    def label_encoder(self) -> DataLabelEncoder:
        return self.__label_encoder

    @label_encoder.setter
    def label_encoder(self, value: DataLabelEncoder):
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
    def get_subset(self, indices: pd.Series) -> 'DataProvider':
        pass

    @abstractmethod
    def get_column(self, column_name: str) -> pd.Series:
        pass

    @abstractmethod
    def get_item(self, index: int):
        pass

    def train_test_split(self, test_size: float) -> Tuple['DataProvider', 'DataProvider']:
        train_indices, test_indices = train_test_split(
            len(self), test_size=test_size, stratify=self.labels)

        train_data = self.get_subset(train_indices)
        test_data = self.get_subset(test_indices)

        # self.label_encoder.fit(self.labels)

        # self.test_label_encoder = self.label_encoder.refit(
        #     test_df[self.label_column])

        # train_df[f"{self.label_column}_index"] = train_df[self.label_column].map(
        #     self.label_encoder.transform)
        # test_df[f"{self.label_column}_index"] = test_df[self.label_column].map(
        #     self.test_label_encoder.transform)

        return train_data, test_data
