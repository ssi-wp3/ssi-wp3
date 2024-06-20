from typing import Dict, Any
from .report import ReportEngine, LuigiReportFileManager
from .parquet_file import ParquetFile
from .settings import Settings
import pandas as pd
import os
import luigi


class ReportTask(luigi.Task):
    data_directory = luigi.PathParameter()
    output_directory = luigi.PathParameter()

    data_file_extension = luigi.Parameter(default=".parquet")
    settings_filename = luigi.PathParameter()
    settings_section_name = luigi.Parameter(default="report_settings")
    render_as_template = luigi.BoolParameter(default=True)
    parquet_engine = luigi.Parameter(default="pyarrow")

    @property
    def report_engine(self) -> ReportEngine:
        return ReportEngine(self.settings_filename)

    def input_file_for(self, filename: str) -> ParquetFile:
        return ParquetFile(os.path.join(self.data_directory, filename))

    def output_target_for(self, filename: str, binary_file: bool = True) -> luigi.LocalTarget:
        if binary_file:
            return luigi.LocalTarget(filename, format=luigi.format.Nop)
        return luigi.LocalTarget(filename)

    def requires(self):
        return {report.input_filename: self.input_file_for(report.input_filename)
                for report in self.report_engine.flattened_reports}

    def output(self):
        return {report.output_filename: self.output_target_for(os.path.join(self.output_directory, report.output_filename), binary_file=report.needs_binary_file)
                for report in self.report_engine.flattened_reports}

    def run(self):
        luigi_report_file_manager = LuigiReportFileManager(
            self.input(), self.output())
        self.report_engine.reports_for_path(
            self.data_directory,
            file_extension=self.data_file_extension,
            report_file_manager=luigi_report_file_manager,
            parquet_engine=self.parquet_engine
        )
