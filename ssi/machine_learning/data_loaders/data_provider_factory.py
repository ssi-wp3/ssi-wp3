from .dataframe_provider import DataframeDataProvider
from .pytorch_provider import ParquetDataset
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
        elif data_provider_type == DataProviderType.HuggingFace:
            raise NotImplementedError(
                "HuggingFace data provider is not implemented.")
        elif data_provider_type == DataProviderType.PyTorch:
            # return ParquetDataset(filename: str,
            #                      features_column,
            #                      label_column,
            #                      **kwargs):
            raise NotImplementedError(
                "PyTorch data provider is not implemented.")
        else:
            raise ValueError(
                f"Data provider type {data_provider_type} is not supported.")
