from typing import List
import os


def get_combined_revenue_filename(supermarket_name: str, project_prefix: str = "ssi") -> str:
    return f"{project_prefix}_{supermarket_name.lower()}_revenue.parquet"


def get_combined_revenue_files_in_directory(directory: str, extension: str = ".parquet", project_prefix: str = "ssi") -> List[str]:
    return [os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.startswith(project_prefix) and filename.endswith(extension) and "revenue" in filename]


class AnalysisDirectories:
    def __init__(self, base_directory: str):
        self.__base_directory = base_directory

    @property
    def base_directory(self) -> str:
        return self.__base_directory

    @property
    def directories(self) -> List[str]:
        return [
            os.path.join(self.base_directory, "plots"),
            os.path.join(self.base_directory, "wordclouds"),
            os.path.join(self.base_directory, "data")
        ]
