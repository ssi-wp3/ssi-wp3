from sklearn.base import ClassifierMixin
from .model import Model
import pandas as pd


class SklearnModel(Model):
    """ The SklearnModel class is a wrapper object around scikit-learn classifiers. It provides
    a load_data method that uses pandas to read a parquet file from disk.

    Parameters
    ----------
    model_type : str
        The model type, this is the name of the classifier

    model : ClassifierMixin
        The scikit-learn classifier
    """

    def __init__(self, model: ClassifierMixin):
        super().__init__()
        self.__model = model

    @property
    def model(self):
        return self.__model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

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
