from typing import List
import os

# TODO duplicate code


def get_combined_revenue_files_in_directory(directory: str, extension: str = ".parquet", project_prefix: str = "ssi") -> List[str]:
    return [os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.startswith(project_prefix) and filename.endswith(extension) and "revenue" in filename]


class FeatureDirectories:
    def __init__(self, base_directory: str):
        self.__base_directory = base_directory

    @property
    def base_directory(self) -> str:
        return self.__base_directory

    @property
    def directories(self) -> List[str]:
        return [
        ]
