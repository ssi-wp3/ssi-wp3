from typing import List, Optional
import os


def get_store_name(filename: str, fileprefix: str = "omzeteanscoicops") -> str:
    return os.path.basename(filename).split("_")[0].lower().replace(fileprefix, "")


def get_revenue_files_in_folder(data_directory: str, supermarket_name: str, filename_prefix: str = "Omzet", file_type: Optional[str] = None) -> List[str]:
    return [os.path.join(data_directory, filename)
            for filename in os.listdir(data_directory)
            if filename.lower().startswith(filename_prefix.lower()) and supermarket_name in filename.lower() and (not file_type or filename.endswith(file_type))]


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
    def cleaned_directory(self) -> str:
        return os.path.join(self.base_directory, "cleaned")

    @property
    def parquet_directory(self) -> str:
        return os.path.join(self.base_directory, "parquet")

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
            self.cleaned_directory,
            self.parquet_directory,
            self.combined_directory,
            self.final_directory
        ]
