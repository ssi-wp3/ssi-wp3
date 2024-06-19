from typing import Dict, Any
from .report import ReportEngine, LuigiReportFileManager
from .settings import Settings
import pandas as pd
import os
import luigi


class ReportTask(luigi.Task):
    data_directory = luigi.PathParameter()

    data_file_extension = luigi.Parameter(default=".parquet")
    settings_filename = luigi.PathParameter()
    settings_section_name = luigi.Parameter(default="report_settings")
    render_as_template = luigi.BoolParameter(default=True)
    template_key_values = luigi.DictParameter(default={})
    parquet_engine = luigi.Parameter(default="pyarrow")

    @property
    def report_settings(self) -> Dict[str, Any]:
        constants_settings = Settings.load(self.settings_filename,
                                           "constants",
                                           False)

        kwargs = dict()
        kwargs.update(constants_settings)
        kwargs.update(self.template_key_values)

        settings = Settings.load(self.settings_filename,
                                 self.settings_section_name,
                                 self.render_as_template,
                                 **kwargs)
        #  store_name=self.store_name,
        #  period_column=self.period_column,
        #  receipt_text_column=self.receipt_text_column,
        #  product_id_column=self.product_id_column,
        #  amount_column=self.amount_column,
        #  revenue_column=self.revenue_column,
        #  coicop_column=self.coicop_column,
        #  coicop_columns=list(self.coicop_columns)
        #  )

        return settings

    @property
    def report_engine(self) -> ReportEngine:
        return ReportEngine(self.report_settings)

    def target_for(self, filename: str, binary_file: bool = True) -> luigi.LocalTarget:
        if binary_file:
            return luigi.LocalTarget(filename, format=luigi.format.Nop)
        return luigi.LocalTarget(filename)

    def requires(self):
        return []

    def output(self):
        return {report.output_filename: self.target_for(os.path.join(self.output_directory, report.output_filename), binary_file=report.needs_binary_file)
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
