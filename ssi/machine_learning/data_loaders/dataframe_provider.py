from typing import Optional
from .data_provider import DataProvider
from .label_encoder import DataLabelEncoder
import pandas as pd


class DataframeDataProvider(DataProvider):
    def init(self,
             filename: str,
             features_column: str,
             label_column: str,
             label_encoder: DataLabelEncoder = DataLabelEncoder(),
             parquet_engine: str = "pyarrow"
             ):
        super().init(filename, features_column, label_column, label_encoder)
        self.__parquet_engine = parquet_engine
        self.__dataframe = None

    @property
    def parquet_engine(self) -> str:
        return self.__parquet_engine

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__dataframe

    @dataframe.setter
    def dataframe(self, value: pd.DataFrame):
        self.__dataframe = value

    @property
    def feature_vector_size(self) -> int:
        return len(self.dataframe[self.features_column].iloc[0])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitems__(self, indices):
        return self.dataframe[self.dataframe.index.isin(indices)]

    def get_column(self, column_name: str) -> pd.Series:
        return self.dataframe[column_name]

    def get_item(self, index: int):
        return self.dataframe.iloc[index]

    def drop_duplicates(self, subset=None, keep='first') -> DataProvider:
        dataframe_without_duplicates = self.dataframe.drop_duplicates(
            subset=subset, keep=keep, ignore_index=False)
        return self.get_subset(dataframe_without_duplicates.index)

    def load(self):
        self.dataframe = pd.read_parquet(
            self.filename, engine=self.parquet_engine)

    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None) -> DataProvider:
        self.fit_or_refit_labels(original_label_encoder)
        subset_df = self.__getitems__(indices)
        subset_df[self.encoded_label_column] = self.label_encoder.transform(
            subset_df[self.label_column])
        return self.__data_provider(subset_df,
                                    self.features_column,
                                    self.label_column,
                                    self.label_encoder)

    def __data_provider(self,
                        dataframe: pd.DataFrame,
                        features_column: str,
                        label_column: str,
                        label_encoder: DataLabelEncoder):
        provider = DataframeDataProvider(
            features_column, label_column, label_encoder)
        provider.dataframe = dataframe
        return provider
