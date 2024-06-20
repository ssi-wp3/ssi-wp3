from typing import Optional
from .data_provider import DataProvider
from .label_encoder import DataLabelEncoder
import pandas as pd


class HuggingFaceDataProvider(DataProvider):
    def __init__(self, filename: str, features_column: str, label_column: str):
        super().__init__(filename, features_column, label_column)

    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None) -> DataProvider:

        # subset_dataset =
        pass
