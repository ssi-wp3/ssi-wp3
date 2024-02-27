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
    output_directory = luigi.PathParameter(default="product_analysis_results")
    store_name = luigi.Parameter()

    def requires(self):
        return

    def run(self):
        pass

    # def output(self):
    #    return luigi.LocalTarget(os.path.join(self.output_directory, "product_analysis_results.csv"), format=luigi.format.Nop)
