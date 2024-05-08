from hiclass import LocalClassifierPerParentNode
from .model import Model
import pandas as pd


class HiClassModel(Model):
    def __init__(self,
                 model
                 ):
        super().__init__(model)
        self.__classifier = LocalClassifierPerParentNode(
            local_classifier=self.model)

    @property
    def classifier(self):
        return self.__classifier

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
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
