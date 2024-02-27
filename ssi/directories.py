from .preprocessing.files import PreprocessingDirectories
from .analysis.files import AnalysisDirectories
from .feature_extraction.files import FeatureDirectories
from .machine_learning.files import MLDirectories
from typing import List
import os


class DirectoryStructure:
    def __init__(self, base_directory: str):
        self.__base_directory = base_directory

    @property
    def base_directory(self):
        return self.__base_directory

    @property
    def preprocessing_directories(self) -> PreprocessingDirectories:
        return PreprocessingDirectories(os.path.join(self.base_directory, "preprocessing"))

    @property
    def analysis_directories(self) -> AnalysisDirectories:
        return AnalysisDirectories(os.path.join(self.base_directory, "analysis"))

    @property
    def feature_directories(self) -> FeatureDirectories:
        return FeatureDirectories(os.path.join(self.base_directory, "feature_extraction"))

    @property
    def machine_learning_directories(self) -> MLDirectories:
        return MLDirectories(os.path.join(self.base_directory, "machine_learning"))

    @property
    def stages(self):
        return [
            self.preprocessing_directories,
            self.analysis_directories,
            self.feature_directories,
            self.machine_learning_directories
        ]

    @property
    def directories(self) -> List[str]:
        return [directory
                for stage in self.stages
                for directory in stage.directories
                ]

    def create_directories(self):
        for directory in self.directories:
            os.makedirs(directory, exist_ok=True)
