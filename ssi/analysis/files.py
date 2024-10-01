from typing import List
import os


def get_combined_revenue_filename(supermarket_name: str, project_prefix: str = "ssi") -> str:
    """Returns the filename for the combined revenue file for a given supermarket.

    Parameters
    ----------

    supermarket_name : str
        The name of the supermarket.

    """
    return f"{project_prefix}_{supermarket_name.lower()}_revenue.parquet"


def get_combined_revenue_files_in_directory(directory: str, extension: str = ".parquet", project_prefix: str = "ssi") -> List[str]:
    """Returns the filenames for the combined revenue files in a given directory.

    Parameters
    ----------

    directory : str
        The directory to search for the files.

    """
    return [os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.startswith(project_prefix) and filename.endswith(extension) and "revenue" in filename]


class AnalysisDirectories:
    def __init__(self, base_directory: str):
        """Initializes the AnalysisDirectories object.

        Parameters
        ----------

        base_directory : str
            The base directory for the analysis.

        """
        self.__base_directory = base_directory

    @property
    def base_directory(self) -> str:
        """Returns the base directory for the analysis.

        Returns
        -------

        str
            The base directory for the analysis.

        """
        return self.__base_directory

    @property
    def directories(self) -> List[str]:
        """Returns the directories for the analysis.

        Returns
        -------

        List[str]
            The directories for the analysis.

        """
        return [
            os.path.join(self.base_directory, "plots"),
            os.path.join(self.base_directory, "wordclouds"),
            os.path.join(self.base_directory, "data")
        ]
