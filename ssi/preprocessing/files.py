from typing import List, Optional
import os


def get_revenue_files_in_folder(data_directory: str, supermarket_name: str, filename_prefix: str = "Omzet", file_type: Optional[str] = None) -> List[str]:
    return [os.path.join(data_directory, filename)
            for filename in os.listdir(data_directory)
            if filename.startswith(filename_prefix) and supermarket_name in filename and (not file_type or filename.endswith(file_type))]


class PreprocessingDirectories:
    def __init__(self, base_directory: str):
        self.__base_directory = base_directory

    @property
    def base_directory(self) -> str:
        return self.__base_directory

    @property
    def raw_directory(self) -> str:
        return os.path.join(self.base_directory, "raw")

    @property
    def preprocessed_directory(self) -> str:
        return os.path.join(self.base_directory, "preprocessed")

    @property
    def combined_directory(self) -> str:
        return os.path.join(self.base_directory, "combined")

    @property
    def final_directory(self) -> str:
        return os.path.join(self.base_directory, "final")

    @property
    def directories(self) -> List[str]:
        return [
            self.raw_directory,
            self.preprocessed_directory,
            self.combined_directory,
            self.final_directory
        ]
