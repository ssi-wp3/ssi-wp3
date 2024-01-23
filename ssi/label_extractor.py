from typing import List, Optional
from .constants import Constants
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

class LabelExtractorFactory:
    def get_label_extractor_for_model(self, model_name: str, coicop_column: Optional[str] = None):
        if model_name == "hiclass":
            return MultiColumnLabelExtractor(Constants.COICOP_LEVELS_COLUMNS[::-1])   
        elif coicop_column is not None: 
            return SingleColumnLabelExtractor(coicop_column)    
        return SingleColumnLabelExtractor(Constants.COICOP_LEVELS_COLUMNS[-1])