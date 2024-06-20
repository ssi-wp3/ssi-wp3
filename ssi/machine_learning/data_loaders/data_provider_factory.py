from .dataframe_provider import DataframeDataProvider
import enum


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
