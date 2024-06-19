from pathlib import Path
from typing import Dict


class FileIndex:
    def __init__(self, root_directory: str, file_extension: str):
        self.__root_directory = Path(root_directory)
        self.__file_extension = file_extension

    @property
    def root_directory(self) -> Path:
        return self.__root_directory

    @property
    def file_extension(self) -> str:
        return self.__file_extension

    @property
    def files(self) -> Dict[str, Path]:
        return {file.name: file for file in self.root_directory.glob(f"*{self.file_extension}")}
