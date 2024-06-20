from .data_provider import DataProvider
from .dataframe_provider import DataframeDataProvider
from .pytorch_provider import ParquetDataset
from .huggingface_provider import HuggingFaceDataProvider
import enum


class DataProviderType(enum.Enum):
    DataFrame = "dataframe"
    HuggingFace = "huggingface"
    PyTorch = "pytorch"


class DataProviderFactory:
    @staticmethod
    def create_data_provider(data_provider_type: DataProviderType,
                             filename: str,
                             features_column: str,
                             label_column: str,
                             **kwargs) -> 'DataProvider':
        if data_provider_type == DataProviderType.DataFrame:
            return DataframeDataProvider(filename, features_column, label_column, **kwargs)
        elif data_provider_type == DataProviderType.HuggingFace:
            return HuggingFaceDataProvider(filename, features_column, label_column, **kwargs)
        elif data_provider_type == DataProviderType.PyTorch:
            return ParquetDataset(filename,
                                  features_column,
                                  label_column,
                                  **kwargs)
        else:
            raise ValueError(
                f"Data provider type {data_provider_type} is not supported.")
