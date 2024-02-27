from .directories import DirectoryStructure
import luigi
import os


class CsvFile(luigi.ExternalTask):
    input_filename = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename, format=luigi.format.Nop)


class CreateProject(luigi.Task):
    """ This task creates a new project directory.

    Parameters
    ----------
    project_name : luigi.Parameter
        The name of the new project.

    """
    input_directory = luigi.PathParameter()
    project_directory = luigi.PathParameter()
    csv_extension = luigi.Parameter(default=".csv")

    @property
    def csv_files(self):
        return [os.path.join(self.input_directory, input_filename)
                for input_filename in os.listdir(self.input_directory)
                if input_filename.endswith(self.csv_extension)
                ]

    @property
    def project(self):
        return DirectoryStructure(self.project_directory)

    def requires(self):
        return [CsvFile(input_filename=input_filename)
                for input_filename in self.csv_files
                ]

    def run(self):
        self.project.create_directories()

        for input in self.input():
            with input.open("r") as input_file:
                with self.output.open("w") as output_file:
                    output_file.write(input_file.read())

    def output(self):
        return [luigi.LocalTarget(os.path.join(
            self.project.preprocessing_directories.raw_directory, os.path.basename(input_filename)))
            for input_filename in self.csv_files]
