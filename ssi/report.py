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
    """Abstract class for a report that can be written to a file.

    Parameters
    ----------
    settings : Dict[str, Any]
        The settings for the report. These settings are often read from a YAML configuration file.

    needs_binary_file : bool
        Whether the report needs a binary file to be written to.
    """

    def __init__(self, settings: Dict[str, Any], needs_binary_file: bool = True):
        self.__settings = settings
        self.__needs_binary_file = needs_binary_file

    @property
    def settings(self) -> Dict[str, Any]:
        """The settings for the report.

        Returns
        -------
        Dict[str, Any]
            The settings for the report.
        """
        return self.__settings

    @property
    def type(self) -> ReportType:
        """The type of the report. The type is one of the values of the ReportType enum.
        At this moment only two types are supported: plot and table.

        Returns
        -------
        ReportType
            The type of the report.
        """
        return ReportType[self.settings["type"].lower()]

    @property
    def type_settings(self) -> Dict[str, Any]:
        """The settings for the specific report type.
        For plots or tables these settings look different.
        These type settings are stored in the settings dictionary under the key "settings"
        and are often passed as kwargs to the plotting or table functions.

        Returns
        -------
        Dict[str, Any]
            The settings for the specific report type.        
        """
        return self.settings["settings"]

    @property
    def needs_binary_file(self) -> bool:
        """Whether the report needs a binary file to be written to.

        Returns
        -------
        bool
            Whether the report needs a binary file to be written to.
        """
        return self.__needs_binary_file

    @needs_binary_file.setter
    def needs_binary_file(self, value: bool):
        """Set whether the report needs a binary file to be written to.

        Parameters
        ----------
        value : bool
            Whether the report needs a binary file to be written to.
        """
        self.__needs_binary_file = value

    @property
    def output_filename(self) -> str:
        """The filename of the output file the report needs to be written to.

        Returns
        -------
        str
            The filename of the output file.
        """
        return self.settings["output_filename"]

    @abstractmethod
    def write_to_file(self, dataframe: pd.DataFrame, filename: str):
        """ This method writes the report to a file. On the report class this
        is an abstractmethod. Each subclass implements its own behavior in this method.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data to generate the report about.

        filename : str
            The filename of the file to write the data to.
        """
        pass


class PlotReport(Report):
    def __init__(self, settings: Dict[str, Any], plot_engine: PlotEngine = PlotEngine()):
        """ A report class for plots. This class is a subclass of the Report class.

        Parameters
        ----------
        settings : Dict[str, Any]
            The settings for the report. These settings are often read from a YAML configuration file.

        plot_engine : PlotEngine
            The plot engine to use for plotting. The default is the PlotEngine class.
        """
        super().__init__(settings, True)
        self.__plot_engine = plot_engine
        # Bit of a hack
        self.__set_plot_format()

    def __set_plot_format(self):
        plot_format = self.type_settings.get("format", "png")
        if plot_format.lower() == "html":
            self.needs_binary_file = False

    @property
    def plot_engine(self) -> PlotEngine:
        """Returns the plot engine to use for plotting.

        Returns
        -------
        PlotEngine
            The plot engine to use for plotting.
        """
        return self.__plot_engine

    def write_to_file(self, dataframe: pd.DataFrame, filename: str):
        """Writes the plot to a file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data to generate the plot about.

        filename : str
            The filename of the file to write the data to.
        """
        self.plot_with_settings(dataframe, self.settings, filename)

    def plot_with_settings(self,
                           dataframe: pd.DataFrame,
                           plot_settings: Dict[str, Any],
                           output_file
                           ):
        """Plots the data with the given settings.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The data to plot.

        plot_settings : Dict[str, Any]
            The settings for the plot. These settings are often read from a YAML configuration file.

        output_file : str
            The filename of the file to write the plot to.
        """
        if "pivot" in plot_settings and plot_settings["pivot"]:
            value_columns = plot_settings["value_columns"]
            dataframe = unpivot(dataframe, value_columns)
        if "sort_values" in plot_settings:
            sort_settings = plot_settings["sort_values"]
            columns = sort_settings.get("columns", [])
            ascending = sort_settings.get("ascending", True)

            dataframe = dataframe.sort_values(
                by=columns, ascending=ascending)

        plot_format = self.type_settings.get("format", "png")

        figure = self.plot_engine.plot_from_settings(
            dataframe, self.type_settings)
        figure.save(output_file, format=plot_format)


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
