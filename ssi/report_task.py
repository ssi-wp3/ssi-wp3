from typing import Dict, Any
from .report import ReportEngine
from .settings import Settings
import pandas as pd
import os
import luigi


class ReportTask(luigi.Task):
    data_directory = luigi.PathParameter()

    settings_filename = luigi.PathParameter()
    settings_section_name = luigi.Parameter(default="report_settings")
    render_as_template = luigi.BoolParameter(default=True)
    template_key_values = luigi.DictParameter(default={})

    @property
    def report_settings(self) -> Dict[str, Any]:
        settings = Settings.load(self.settings_filename,
                                 self.settings_section_name,
                                 self.render_as_template,
                                 **self.template_key_values)
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

    def output(self):
        return {report.output_filename: self.target_for(os.path.join(self.output_directory, report.output_filename), binary_file=report.needs_binary_file)
                for report in self.report_engine.flattened_reports}

    def run(self):
        for task in self.input():
            for function_name, input in task.items():
                if function_name not in self.report_settings:
                    continue

                with input.open("r") as input_file:
                    dataframe = pd.read_parquet(
                        input_file, engine=self.parquet_engine)

                    reports = self.report_engine.reports[function_name]
                    for report in reports:
                        with self.output()[report.output_filename].open("w") as output_file:
                            report.write_to_file(dataframe, output_file)
