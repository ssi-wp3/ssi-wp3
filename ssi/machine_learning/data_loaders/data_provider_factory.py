from .data_provider import DataProvider
from .dataframe_provider import DataframeDataProvider
from .pytorch_provider import ParquetDataset
from .huggingface_provider import HuggingFaceDataProvider
import enum


class DataProviderType(enum.Enum):
    """ The type of data provider to create.

    Attributes
    ----------
    DataFrame : str
        The data provider is a pandas DataFrame.
    HuggingFace : str
        The data provider is a HuggingFace dataset.
    PyTorch : str
        The data provider is a PyTorch dataset.
    """
    DataFrame = "dataframe"
    HuggingFace = "huggingface"
    PyTorch = "pytorch"


class DataProviderFactory:
    """ A factory for creating data providers.
    """

    @staticmethod
    def create_data_provider(data_provider_type: DataProviderType,
                             filename: str,
                             features_column: str,
                             label_column: str,
                             **kwargs) -> 'DataProvider':
        """ Creates a data provider of the specified type.

        Parameters
        ----------
        data_provider_type : DataProviderType
            The type of data provider to create.
        filename : str
            The filename of the data provider.
        features_column : str
            The column containing the feature data.
        label_column : str
            The column containing the label data.
        **kwargs : dict
            Additional keyword arguments to pass to the data provider.

        Returns
        -------
        DataProvider
            The data provider of the specified type.
        """
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
