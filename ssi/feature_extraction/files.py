from typing import List
import os

# TODO duplicate code


def get_combined_revenue_files_in_directory(directory: str, extension: str = ".parquet", project_prefix: str = "ssi") -> List[str]:
    """ This method returns the list of revenue files in the given directory.

    Parameters
    ----------
    directory : str
        The directory to search for revenue files.

    extension : str
        The extension of the files to search for.

    project_prefix : str
        The prefix of the project.

    Returns
    -------
    List[str]
        The list of revenue files in the given directory.
    """
    return [os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.startswith(project_prefix) and filename.endswith(extension) and "revenue" in filename]


class FeatureDirectories:
    """ This class is used to store the directories for the feature extraction tasks.

    Parameters
    ----------
    base_directory : str
        The base directory for the feature extraction tasks.
    """

    def __init__(self, base_directory: str):
        self.__base_directory = base_directory

    @property
    def base_directory(self) -> str:
        """ This property returns the base directory for the feature extraction tasks.

        Returns
        -------
        str
            The base directory for the feature extraction tasks.
        """
        return self.__base_directory

    @property
    def directories(self) -> List[str]:
        """ This property returns the list of directories for the feature extraction tasks.

        Returns
        -------
        List[str]
            The list of directories for the feature extraction tasks.
        """
        return [
            os.path.join(self.base_directory, "features"),
            os.path.join(self.base_directory, "unique_values")
        ]
