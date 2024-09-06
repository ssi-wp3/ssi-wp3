from typing import Dict, Any
from .report import ReportEngine, LuigiReportFileManager
from .parquet_file import ParquetFile
from .settings import Settings
from .file_index import FileIndex
import pandas as pd
import os
import luigi
import zipfile


class ReportTask(luigi.Task):
    data_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()

    data_file_extension = luigi.Parameter(default=".parquet")
    settings_filename = luigi.PathParameter()
    settings_section_name = luigi.Parameter(default="report_settings")
    render_as_template = luigi.BoolParameter(default=True)
    parquet_engine = luigi.Parameter(default="pyarrow")
    __report_engine = None

    @property
    def report_engine(self) -> ReportEngine:
        if not self.__report_engine:
            self.__report_engine = ReportEngine(self.settings_filename)
        return self.__report_engine

    def input_file_for(self, filename: str) -> ParquetFile:
        return ParquetFile(os.path.join(self.data_directory, filename))

    def output_target_for(self, filename: str, binary_file: bool = True) -> luigi.LocalTarget:
        if binary_file:
            return luigi.LocalTarget(filename, format=luigi.format.Nop)
        return luigi.LocalTarget(filename)

    def requires(self):
        file_index = FileIndex(self.data_directory, self.data_file_extension)
        return {file_key: self.input_file_for(file_path)
                for file_key, file_path in file_index.files.items()
                if file_key in self.report_engine.reports.keys()}

    def output(self):
        return {report.output_filename: self.output_target_for(os.path.join(self.output_directory, report.output_filename), binary_file=False
                                                               # TODO hack, should be fixed in the future
                                                               if report.output_filename.endswith(".html") else report.needs_binary_file)
                for report in self.report_engine.flattened_reports}

    def zip_output_files(self, filename: str):
        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for report in self.output().values():
                zipf.write(report.path)

    def run(self):
        luigi_report_file_manager = LuigiReportFileManager(
            self.input(), self.output())
        self.report_engine.reports_for_file_index(
            self.input(),
            report_file_manager=luigi_report_file_manager,
            parquet_engine=self.parquet_engine
        )
        self.zip_output_files(os.path.join(
            self.output_directory, "reports.zip"))
