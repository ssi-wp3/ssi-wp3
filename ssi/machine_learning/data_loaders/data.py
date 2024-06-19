from typing import Tuple
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
    @abstractmethod
    def feature_vector_length(self) -> int:
        pass

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

    @abstractmethod
    def load(self, filename: str):
        pass

    @abstractmethod
    def get_subset(self, indices: pd.Series) -> pd.DataFrame:
        pass


class DataProviderType(enum.Enum):
    DataFrame = "dataframe"
    HuggingFace = "huggingface"
    PyTorch = "pytorch"


class DataProviderFactory:
    @staticmethod
    def create_data_provider(data_provider_type: DataProviderType,
                             features_column: str,
                             label_column: str,
                             **kwargs) -> 'DataProvider':
        if data_provider_type == DataProviderType.DataFrame:
            return DataframeDataProvider(features_column, label_column, **kwargs)
        else:
            raise ValueError(
                f"Data provider type {data_provider_type} is not supported.")


class DataframeDataProvider(DataProvider):
    def init(self,
             features_column: str,
             label_column: str,
             label_encoder: DataLabelEncoder = DataLabelEncoder(),
             parquet_engine: str = "pyarrow"
             ):
        super().init(features_column, label_column, label_encoder)
        self.__parquet_engine = parquet_engine
        self.__dataframe = None

    @property
    def parquet_engine(self) -> str:
        return self.__parquet_engine

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__dataframe

    def __len__(self) -> int:
        return len(self.dataframe)

    def load(self, filename: str):
        self.__dataframe = pd.read_parquet(
            filename, engine=self.parquet_engine)

    def get_subset(self, indices: pd.Series) -> pd.DataFrame:
        return self.dataframe.loc[indices]

    def split_data(self, dataframe: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # TODO split off to DataSplitter.

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
