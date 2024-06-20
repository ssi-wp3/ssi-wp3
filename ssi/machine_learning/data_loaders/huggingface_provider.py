from typing import Optional
from datasets import Dataset
from .data_provider import DataProvider
from .label_encoder import DataLabelEncoder
import pandas as pd


class HuggingFaceDataProvider(DataProvider):
    def __init__(self, filename: str, features_column: str, label_column: str):
        super().__init__(filename, features_column, label_column)
        self.__dataset = None

    @property
    def dataset(self) -> Optional[Dataset]:
        return self.__dataset

    @dataset.setter
    def dataset(self, value: Dataset):
        self.__dataset = value

    def load(self):
        self.dataset = Dataset.from_pandas(
            pd.read_parquet(self.filename, engine="pyarrow"))
        self.label_encoder.fit(self.dataset[self.label_column])
        self.dataset = self.dataset.class_encode_column(self.label_column)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitems__(self, indices):
        return self.dataset[indices]

    def get_column(self, column_name: str) -> pd.Series:
        return self.dataset[column_name]

    def get_item(self, index: int) -> pd.Series:
        return self.dataset[index]

    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None) -> DataProvider:
        self.fit_or_refit_labels(original_label_encoder)
        subset_dataset = self.__getitems__(indices)
        subset_provider = self.__data_provider_for(subset_dataset)
        return subset_provider

    def __data_provider_for(self, subset_dataset: Dataset):
        subset_provider = HuggingFaceDataProvider(
            self.filename, self.features_column, self.label_column)
        subset_provider.dataset = subset_dataset
        return subset_provider
