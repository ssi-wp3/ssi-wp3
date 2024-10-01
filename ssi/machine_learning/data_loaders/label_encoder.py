from collections import OrderedDict
import pandas as pd


class DataLabelEncoder:
    """ Encodes labels to their encoded values. It also provides the inverse mapping.

    """

    def __init__(self, label_mapping: OrderedDict = None):
        """ Initializes the label encoder.

        Parameters
        ----------
        label_mapping : OrderedDict, optional
            The mapping of the labels to their encoded values, by default None.
        """
        self.__label_mapping = label_mapping

    @property
    def label_mapping(self) -> OrderedDict:
        """ The mapping of the labels to their encoded values.

        Returns
        -------
        OrderedDict
            The mapping of the labels to their encoded values.
        """
        return self.__label_mapping

    @label_mapping.setter
    def label_mapping(self, value: OrderedDict):
        """ Sets the mapping of the labels to their encoded values.

        Parameters
        ----------
        value : OrderedDict
            The mapping of the labels to their encoded values.
        """
        self.__label_mapping = value

    @property
    def inverse_label_mapping(self) -> OrderedDict:
        """ The mapping of the encoded labels to their original values.

        Returns
        -------
        OrderedDict
            The mapping of the encoded labels to their original values.
        """
        return OrderedDict([(v, k) for k, v in self.label_mapping.items()])

    def fit(self, labels: pd.Series):
        """ Fit the label encoder to the labels.

        Parameters
        ----------
        labels : pd.Series
            The labels to encode.
        """
        self.train_label_mapping = OrderedDict([(original_label, index)
                                                for index, original_label in enumerate(labels.unique())])

    def refit(self, extra_labels: pd.Series) -> 'DataLabelEncoder':
        """ Refit the label encoder to the labels while preserving the original label->index mapping.
        Sometimes the test dataset can have more categories than the training dataset, add them add the end of the mapping.

        Parameters
        ----------
        extra_labels : pd.Series
            The extra labels to encode.

        Returns
        -------
        DataLabelEncoder
            The refitted label encoder.        
        """
        # Test dataset can have more categories than the training dataset, add them add the end of the mapping
        # In this way, we preserve the original label->index mapping for the training dataset
        self.new_label_mapping = self.label_mapping
        for label in extra_labels.unique():
            if label not in self.new_label_mapping:
                self.new_label_mapping[label] = len(self.new_label_mapping)
        return DataLabelEncoder(label_mapping=self.new_label_mapping)

    def transform(self, labels: pd.Series) -> pd.Series:
        """ Transform the labels to their encoded values.

        Parameters
        ----------
        labels : pd.Series
            The labels to encode.

        Returns
        -------
        pd.Series
            The encoded labels.
        """
        return labels.map(self.label_mapping)

    def fit_transform(self, labels: pd.Series) -> pd.Series:
        """Fit and transform the labels to their encoded values.

        Parameters
        ----------
        labels : pd.Series
            The labels to encode.

        Returns
        -------
        pd.Series
            The encoded labels.
        """
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, encoded_labels: pd.Series) -> pd.Series:
        """Inverse transform the labels back to their original values.

        Parameters
        ----------
        encoded_labels : pd.Series
            The encoded labels.

        Returns
        -------
        pd.Series
            The original labels.
        """
        return encoded_labels.map(self.inverse_label_mapping)
