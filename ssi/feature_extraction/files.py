from typing import List
import os


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
