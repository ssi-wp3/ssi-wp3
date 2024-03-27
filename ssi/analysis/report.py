from typing import Dict, Any, List
from enum import Enum
from .utils import unpivot
from ..plots import PlotEngine
from ..settings import Settings
import pandas as pd


class ReportEngine:
    def __init__(self, settings: Settings):
        self.__settings = settings

    @property
    def settings(self) -> Settings:
        return self.__settings

    @property
    def output_filenames(self) -> Dict[str, str]:
        return {result_key: result_settings["filename"]
                for result_key, result_settings in self.settings.items()}

    @property
    def reports(self) -> List['Report']:
        reports = []
        for report_settings in self.settings.items():
            if isinstance(report_settings, list):
                for settings in report_settings:
                    reports.append(self.create_report(settings))
            else:
                reports.append(self.create_report(report_settings))
        return reports

    def create_report(self, result_settings: Dict[str, Any]) -> 'Report':
        if result_settings["type"] == "plot":
            return PlotReport(result_settings)
        elif result_settings["type"] == "table":
            return TableReport(result_settings)
        else:
            raise ValueError(f"Unknown report type: {result_settings['type']}")


class ReportType(Enum):
    plot = "plot"
    table = "table"


class Report:
    def __init__(self, settings: Dict[str, Any]):
        self.__settings = settings

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
    def output_filename(self) -> str:
        return self.settings["output_filename"]


class PlotReport(Report):
    def __init__(self, settings: Dict[str, Any], plot_engine: PlotEngine = PlotEngine()):
        super().__init__(settings)
        self.__plot_engine = plot_engine

    @property
    def plot_engine(self) -> PlotEngine:
        return self.__plot_engine

    def plot_with_settings(self,
                           dataframe: pd.DataFrame,
                           plot_settings: Dict[str, Any],
                           output_file,
                           value_columns: List[str] = None):
        if "pivot" in plot_settings and plot_settings["pivot"]:
            value_columns = plot_settings["value_columns"]
            dataframe = unpivot(dataframe, value_columns)

        figure = self.plot_engine.plot_from_settings(
            dataframe, plot_settings["plot_settings"])
        figure.save(output_file)


class TableReport(Report):
    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings)

    def to_table(self, dataframe: pd.DataFrame, output_file: str):
        dataframe.to_csv(output_file, index=False)
