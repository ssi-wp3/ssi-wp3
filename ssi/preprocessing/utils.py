
import luigi


class ParquetFile(luigi.ExternalTask):
    input_filename = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.input_filename, format=luigi.format.Nop)
