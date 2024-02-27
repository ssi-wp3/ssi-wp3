from .directories import DirectoryStructure
import luigi
import os


class CsvFile(luigi.ExternalTask):
    input_filename = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename, format=luigi.format.Nop)


class CreateProject(luigi.WrapperTask):
    """ This task creates a new project directory.

    Parameters
    ----------
    project_name : luigi.Parameter
        The name of the new project.

    """
    input_directory = luigi.PathParameter()
    project_directory = luigi.PathParameter()
    csv_extension = luigi.Parameter(default=".csv")

    def requires(self):
        return [CsvFile(input_filename=os.path.join(self.input_directory, input_filename))
                for input_filename in os.listdir(self.input_directory)
                if input_filename.endswith(self.csv_extension)
                ]

    def run(self):
        project_directory = DirectoryStructure(self.project_directory)
        project_directory.create_directories()

        for input in self.input():
            with input.open("r") as input_file:
                input_filename = os.path.basename(input.path)
                output_filename = os.path.join(
                    project_directory.preprocessing_directories.raw_directory, input_filename)
                with open(output_filename, "w") as output_file:
                    output_file.write(input_file.read())
