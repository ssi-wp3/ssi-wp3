from typing import Optional
from .data_provider import DataProvider
from .label_encoder import DataLabelEncoder
import pandas as pd


class DataframeDataProvider(DataProvider):
    """ A data provider that loads data from a pandas DataFrame.
    """

    def init(self,
             filename: str,
             features_column: str,
             label_column: str,
             label_encoder: DataLabelEncoder = DataLabelEncoder(),
             parquet_engine: str = "pyarrow"
             ):
        """ Initializes the data provider.

        Parameters
        ----------
        filename : str
            The filename of the data provider.
        features_column : str
            The column containing the feature data.
        """
        super().init(filename, features_column, label_column, label_encoder)
        self.__parquet_engine = parquet_engine
        self.__dataframe = None

    @property
    def parquet_engine(self) -> str:
        """ The parquet engine to use for reading the data.
        """
        return self.__parquet_engine

    @property
    def dataframe(self) -> pd.DataFrame:
        """ The pandas DataFrame containing the data.
        """
        return self.__dataframe

    @dataframe.setter
    def dataframe(self, value: pd.DataFrame):
        """ Sets the pandas DataFrame containing the data.
        """
        self.__dataframe = value

    @property
    def feature_vector_size(self) -> int:
        """ The size of the feature vector.
        """
        return len(self.dataframe[self.features_column].iloc[0])

    def __len__(self) -> int:
        """ The number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.dataframe)

    def __getitems__(self, indices):
        """ Returns the items at the specified indices.

        Parameters
        ----------
        indices : pd.Series
            The indices of the items to return.

        Returns
        -------
        pd.DataFrame
            The items at the specified indices.
        """
        return self.dataframe[self.dataframe.index.isin(indices)]

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
        return self.dataframe[column_name]

    def get_item(self, index: int):
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
        return self.dataframe.iloc[index]

    def drop_duplicates(self, subset=None, keep='first') -> DataProvider:
        """ Drops duplicate rows from the dataset.

        Parameters
        ----------
        subset : str or list of str, optional
            The column or columns to use for identifying duplicates, by default None.
        keep : {'first', 'last', 'False'}, optional
            The method to use for identifying duplicates, by default 'first'.

        Returns
        -------
        DataProvider
            A new DataProvider object with the duplicate rows removed.
        """
        dataframe_without_duplicates = self.dataframe.drop_duplicates(
            subset=subset, keep=keep, ignore_index=False)
        return self.get_subset(dataframe_without_duplicates.index)

    def load(self):
        """ Loads the data from the filename.
        """
        self.dataframe = pd.read_parquet(
            self.filename, engine=self.parquet_engine)

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
        """ Creates a new DataProvider object with the specified subset of the data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The pandas DataFrame containing the data.
        features_column : str
            The column containing the feature data.
        label_column : str
            The column containing the label data.
        label_encoder : DataLabelEncoder
            The label encoder to use.

        Returns
        -------
        DataProvider
            A new DataProvider object with the specified subset of the data.
        """
        provider = DataframeDataProvider(
            features_column, label_column, label_encoder)
        provider.dataframe = dataframe
        return provider
