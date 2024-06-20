from .data_provider import DataProvider


class HuggingFaceDataProvider(DataProvider):
    def __init__(self, features_column: str, label_column: str):
        super().__init__(features_column, label_column)
