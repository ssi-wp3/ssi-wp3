from luigi.contrib.external_program import ExternalProgramTask
import luigi


class ConvertCPIFiles(ExternalProgramTask):
    input_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()

    def program_args(self):
        return ['bash', './convert_cpi_files.sh', self.input_directory, self.output_directory]
