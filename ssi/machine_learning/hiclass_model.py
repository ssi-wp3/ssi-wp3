from hiclass import LocalClassifierPerParentNode
from .model import Model
import pandas as pd


class HiClassModel(Model):
    """ Class to wrap the HiClass model. """

    def __init__(self,
                 model
                 ):
        super().__init__(model)
        self.__classifier = LocalClassifierPerParentNode(
            local_classifier=self.model)

    @property
    def classifier(self):
        """ Get the classifier.

        Returns
        -------
        LocalClassifierPerParentNode
            The classifier
        """
        return self.__classifier

    def fit(self, X, y):
        """ Fit the model to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series
            The labels
        """
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        """ Predict the labels for the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data

        Returns
        -------
        np.array
            The predicted labels
        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """ Predict the probabilities for the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data

        Returns
        -------
        np.array
            The predicted probabilities
        """
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        """ Score the model on the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series
            The labels

        Returns
        -------
        float
            The score
        """
        return self.classifier.score(X, y)

    def load_data(self, filename: str, **kwargs) -> pd.DataFrame:
        """ Load data from a parquet file using pandas

        Parameters
        ----------

        filename : str
            The filename of the parquet file to load

        **kwargs
            Additional keyword arguments to pass to pd.read_parquet, in this way you can, for example, specify the engine to use.

        Returns
        -------
        pd.DataFrame
            The data from the parquet file
        """
        return pd.read_parquet(filename, **kwargs)
