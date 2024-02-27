from luigi.contrib.external_program import ExternalProgramTask
import luigi
import pathlib
import os


class CleanCPIFile(ExternalProgramTask):
    """ This task converts CBS CPI CSV files to CSV file format that can be read by python.

    The CBS CPI files use an unusual encoding. The shell script takes an input 
    file, reads the input_filename, skips the first 3 bytes (BOM), and converts 
    them from cp1252 to utf-8. Last, it filters out the carriage returns '\r' 
    and writes the result to a file with output_filename.

    Parameters
    ----------
    input_filename : luigi.PathParameter
        The input filename of the original CBS CPI file.

    output_filename : luigi.PathParameter
        The output filename for the cleaned CBS CPI file.

    """
    input_filename = luigi.PathParameter()
    output_filename = luigi.PathParameter()

    def program_args(self):
        current_file_path = pathlib.Path(__file__).parent.resolve()
        return ['bash', os.path.join(current_file_path,  'clean_cpi_file.sh'), self.input_filename, self.output_filename]

    def output(self):
        """

        TODO: check if this doesn't cause the atomic writes problem mentioned here: 
        https://luigi.readthedocs.io/en/latest/luigi_patterns.html#atomic-writes-problem
        """
        return luigi.LocalTarget(self.output_filename)
