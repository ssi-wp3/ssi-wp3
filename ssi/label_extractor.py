from typing import List
import pandas as pd

class LabelExtractor:
    def get_labels(self, dataframe: pd.DataFrame):
        raise NotImplementedError()    

class SingleColumnLabelExtractor(LabelExtractor):
    def __init__(self, label_column: str):
        self.__label_column = label_column

    @property
    def label_column(self):
        return self.__label_column

    def get_labels(self, dataframe: pd.DataFrame):
        return dataframe[self.label_column]


class MultiColumnLabelExtractor(LabelExtractor):
    def __init__(self, label_columns: List[str]):
        self.__label_columns = label_columns

    @property
    def label_columns(self):
        return self.__label_columns

    def get_labels(self, dataframe: pd.DataFrame):
        return dataframe[self.label_columns].values.tolist()