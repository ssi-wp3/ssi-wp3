from .files import get_combined_revenue_files_in_directory
import pandas as pd
import pathlib
import luigi
import os


class StoreFile(luigi.ExternalTask):
    input_filename = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename)


class StoreProductAnalysis(luigi.Task):
    """ This task analyzes the store product inventory and dynamics.

    """
    input_filename = luigi.PathParameter()
    output_directory = luigi.PathParameter()

    def requires(self):
        return StoreFile(self.input_filename)

    def run(self):
        pass

    # def output(self):
    #    return luigi.LocalTarget(os.path.join(self.output_directory, "product_analysis_results.csv"), format=luigi.format.Nop)


class AllStoresAnalysis(luigi.WrapperTask):
    """ This task analyses the product inventory and dynamics for all stores in a certain
    directory.

    Parameters
    ----------

    input_directory : luigi.PathParameter
        The directory containing the store product inventory files.

    output_directory : luigi.PathParameter
        The directory to store the analysis results.

    project_prefix : luigi.Parameter
        The project prefix used in the revenue file names.
    """
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()
    project_prefix = luigi.Parameter(default="ssi")

    def requires(self):
        return [StoreProductAnalysis(filename, self.output_directory)
                for filename in get_combined_revenue_files_in_directory(self.input_directory, project_prefix=self.project_prefix)]
