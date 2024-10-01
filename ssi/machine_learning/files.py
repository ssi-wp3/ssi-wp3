from typing import List
import os


class MLDirectories:
    """ Class to manage the directories for the machine learning pipeline. """

    def __init__(self, base_directory: str):
        """ Initialize the directories.

        Parameters
        ----------
        base_directory : str
            The base directory
        """
        self.__base_directory = base_directory

    @property
    def base_directory(self) -> str:
        """ Get the base directory.

        Returns
        -------
        str
            The base directory
        """
        return self.__base_directory

    @property
    def directories(self) -> List[str]:
        """ Get the directories.

        Returns
        -------
        List[str]
            The directories
        """
        return [
        ]
