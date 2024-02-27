from .preprocessing.files import PreprocessingDirectories
import os


class DirectoryStructure:
    def __init__(self, base_directory: str):
        self.__base_directory = base_directory

    @property
    def base_directory(self):
        return self.__base_directory

    @property
    def directories(self):
        return PreprocessingDirectories(self.base_directory).directories

    def create_directories(self):
        for directory in self.directories:
            os.makedirs(directory, exist_ok=True)
