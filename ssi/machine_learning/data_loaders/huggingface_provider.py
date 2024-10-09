from typing import Optional
from datasets import Dataset
from .data_provider import DataProvider
from .label_encoder import DataLabelEncoder
import pandas as pd


class HuggingFaceDataProvider(DataProvider):
    def __init__(self, filename: str, features_column: str, label_column: str):
        """ Initializes the data provider.

        Parameters
        ----------
        filename : str
            The filename of the data provider.
        features_column : str
            The column containing the feature data.
        label_column : str
            The column containing the label data.
        """
        super().__init__(filename, features_column, label_column)
        self.__dataset = None

    @property
    def dataset(self) -> Optional[Dataset]:
        """ The HuggingFace Dataset containing the data.

        Returns
        -------
        Dataset
            The HuggingFace Dataset containing the data.
        """
        return self.__dataset

    @dataset.setter
    def dataset(self, value: Dataset):
        """ Sets the HuggingFace Dataset containing the data.

        Parameters
        ----------
        value : Dataset
            The HuggingFace Dataset containing the data.
        """
        self.__dataset = value

    def load(self):
        """ Loads the data from the filename.
        """
        self.dataset = Dataset.from_pandas(
            pd.read_parquet(self.filename, engine="pyarrow"))
        self.label_encoder.fit(self.dataset[self.label_column])
        self.dataset = self.dataset.class_encode_column(self.label_column)

    def __len__(self) -> int:
        """ Returns the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.dataset)

    def __getitems__(self, indices):
        """ Returns the items at the specified indices.

        Parameters
        ----------
        indices : pd.Series
            The indices of the items to return.

        Returns
        -------
        Dataset
            The items at the specified indices.
        """
        return self.dataset[indices]

    def get_column(self, column_name: str) -> pd.Series:
        """ Returns the column with the specified name.

        Parameters
        ----------
        column_name : str
            The name of the column to return.

        Returns
        -------
        pd.Series
            The column with the specified name.
        """
        return self.dataset[column_name]

    def get_item(self, index: int) -> pd.Series:
        """ Returns the item at the specified index.

        Parameters
        ----------
        index : int
            The index of the item to return.

        Returns
        -------
        pd.Series
            The item at the specified index.
        """
        return self.dataset[index]

    def get_subset(self,
                   indices: pd.Series,
                   original_label_encoder: Optional[DataLabelEncoder] = None) -> DataProvider:
        """ Returns a subset of the data.

        Parameters
        ----------
        indices : pd.Series
            The indices of the items to return.
        original_label_encoder : DataLabelEncoder, optional
            The original label encoder to use, by default None.

        Returns
        -------
        DataProvider
            A new DataProvider object with the specified subset of the data.
        """
        self.fit_or_refit_labels(original_label_encoder)
        subset_dataset = self.__getitems__(indices)
        subset_provider = self.__data_provider_for(subset_dataset)
        return subset_provider

    def __data_provider_for(self, subset_dataset: Dataset):
        """ Creates a new DataProvider object with the specified subset of the data.

        Parameters
        ----------
        subset_dataset : Dataset
            The HuggingFace Dataset containing the subset of the data.

        Returns
        -------
        DataProvider
            A new DataProvider object with the specified subset of the data.
        """
        subset_provider = HuggingFaceDataProvider(
            self.filename, self.features_column, self.label_column)
        subset_provider.dataset = subset_dataset
        return subset_provider
