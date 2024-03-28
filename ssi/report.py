from typing import Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
from .analysis.utils import unpivot
from .plots import PlotEngine
from .settings import Settings
import pandas as pd


class ReportType(Enum):
    plot = "plot"
    table = "table"


class Report(ABC):
    def __init__(self, settings: Dict[str, Any], needs_binary_file: bool = True):
        self.__settings = settings
        self.__needs_binary_file = needs_binary_file

    @property
    def settings(self) -> Dict[str, Any]:
        return self.__settings

    @property
    def type(self) -> ReportType:
        return ReportType[self.settings["type"].lower()]

    @property
    def type_settings(self) -> Dict[str, Any]:
        return self.settings["settings"]

    @property
    def needs_binary_file(self) -> bool:
        return self.__needs_binary_file

    @needs_binary_file.setter
    def needs_binary_file(self, value: bool):
        self.__needs_binary_file = value

    @property
    def output_filename(self) -> str:
        return self.settings["output_filename"]

    @abstractmethod
    def write_to_file(self, dataframe: pd.DataFrame, filename: str):
        pass


class PlotReport(Report):
    def __init__(self, settings: Dict[str, Any], plot_engine: PlotEngine = PlotEngine()):
        super().__init__(settings, True)
        self.__plot_engine = plot_engine

    @property
    def plot_engine(self) -> PlotEngine:
        return self.__plot_engine

    def write_to_file(self, dataframe: pd.DataFrame, filename: str):
        self.plot_with_settings(dataframe, self.settings, filename)

    def plot_with_settings(self,
                           dataframe: pd.DataFrame,
                           plot_settings: Dict[str, Any],
                           output_file,
                           value_columns: List[str] = None):

        if "pivot" in plot_settings and plot_settings["pivot"]:
            value_columns = plot_settings["value_columns"]
            dataframe = unpivot(dataframe, value_columns)

        figure = self.plot_engine.plot_from_settings(
            dataframe, self.type_settings)
        figure.save(output_file)


class TableReport(Report):
    class TableType(Enum):
        csv = "csv"
        markdown = "markdown"
        latex = "latex"

    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings, False)

    def write_to_file(self, dataframe: pd.DataFrame, filename: str):
        table_type = TableReport.TableType[self.type_settings.get(
            "table_type", "csv")]

        table_settings = self.type_settings.copy()
        table_settings.pop("table_type", None)

        self.to_table(dataframe,
                      table_type=table_type,
                      output_file=filename,
                      **table_settings)

    def to_table(self, dataframe: pd.DataFrame, table_type: 'TableReport.TableType', output_file: str, **kwargs):
        if table_type == TableReport.TableType.csv:
            dataframe.to_csv(output_file, **kwargs)
        elif table_type == TableReport.TableType.markdown:
            dataframe.to_markdown(output_file, **kwargs)
        elif table_type == TableReport.TableType.latex:
            dataframe.to_latex(output_file, **kwargs)
        else:
            raise ValueError(f"Unknown table type: {table_type}")


class ReportEngine:
    def __init__(self, settings: Settings):
        self.__settings = settings

    @property
    def settings(self) -> Settings:
        return self.__settings

    @property
    def flattened_reports(self) -> List[Report]:
        return [report
                for _, report_list in self.reports.items()
                for report in report_list]

    @property
    def reports(self) -> Dict[str, List['Report']]:
        reports = defaultdict(list)
        for report_key, report_settings in self.settings.items():
            if isinstance(report_settings, list):
                for settings in report_settings:
                    reports[report_key].append(self.report_for(settings))
            else:
                reports[report_key].append(self.report_for(report_settings))
        return reports

    def report_for(self, result_settings: Dict[str, Any]) -> 'Report':
        if result_settings["type"].lower() == ReportType.plot.value:
            return PlotReport(result_settings)
        elif result_settings["type"].lower() == ReportType.table.value:
            return TableReport(result_settings)
        else:
            raise ValueError(f"Unknown report type: {result_settings['type']}")
