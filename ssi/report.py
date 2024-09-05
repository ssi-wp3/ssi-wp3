from typing import Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
from .analysis.utils import unpivot
from .plots import PlotEngine
from .settings import Settings
from .file_index import FileIndex
import luigi
import pandas as pd
import tqdm
import itertools
import os


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
    def preprocessing_settings(self) -> Dict[str, Any]:
        """The settings for the preprocessing of the data before generating the report.

        Returns
        -------
        Dict[str, Any]
            The settings for the preprocessing of the data before generating the report.
        """
        return self.settings.get("preprocessing", dict())

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


class CustomLatex:
    @staticmethod
    def encode_column_name(column_name: Any, rename_columns: Dict[str, str]) -> str:
        column_name = f"{column_name}"
        if column_name in rename_columns:
            column_name = rename_columns[column_name]

        return column_name.replace("_", "\\_").lower()

    @staticmethod
    def encode_column(column_data: Any, float_format: str) -> str:
        if isinstance(column_data, str):
            return f"{column_data}"
        if isinstance(column_data, float):
            return f"{column_data:{float_format}}"
        return f"{column_data}"

    @staticmethod
    def index_to_latex(index: pd.Index, add_index: bool) -> str:
        return f"{index} " if add_index else ""

    @staticmethod
    def index_name(index: pd.Index, add_index: bool) -> str:
        return f"{index.name} " if add_index and index.name else ""

    @staticmethod
    def to_latex(dataframe: pd.DataFrame,
                 output_file: str,
                 title: str,
                 label: str,
                 float_format: str,
                 add_resize_box: bool = False,
                 rename_columns: Dict[str, str] = dict(),
                 add_index: bool = False,
                 sort_index: bool = False,
                 **kwargs):
        if sort_index:
            dataframe = dataframe.sort_index()

        column_names = CustomLatex.index_name(dataframe.index, add_index) + " & ".join([CustomLatex.encode_column_name(column_name, rename_columns)
                                                                                        for column_name in dataframe.columns.values])
        column_alignments = "".join(["l" if column_index == 0 else "r" for column_index in range(
            len(dataframe.columns))])
        table_data = "\\\\\n".join([CustomLatex.index_to_latex(index, add_index) + " & ".join([CustomLatex.encode_column(column, float_format)
                                                                                               for column in row.values])
                                    for index, row in dataframe.iterrows()])

        begin_resize_box = ""
        end_resize_box = ""
        if add_resize_box:
            begin_resize_box = "\\resizebox{\\textwidth}{!}{"
            end_resize_box = "}"

        latex_string = f"""
            \\begin{{table}}
                \\centering
                {begin_resize_box}
                \\begin{{tabular}}{{ {column_alignments} }}
                    \\toprule
                    {column_names}
                    \\midrule
                    {table_data}
                    \\bottomrule
                    \\caption{{ {title} }}
                    \\label{{table:{label} }}
                \\end{{tabular}}
                {end_resize_box}
            \\end{{table}}
            """
        output_file.write(latex_string)


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
            CustomLatex.to_latex(dataframe, output_file, **kwargs)
        else:
            raise ValueError(f"Unknown table type: {table_type}")


class ReportFileManager(ABC):
    """Abstract class for a report file manager.
    This class is used to make it possible to use the ReportEngine with luigi.
    """
    @abstractmethod
    def open_input_file(self, filename: str):
        """ Opens a file for reading.

        Parameters
        ----------
        filename : str
            The filename of the file to open.

        Returns
        -------
        A context manager used for reading the opened file.

        """
        pass

    @abstractmethod
    def open_output_file(self, filename: str):
        """ Opens a file for writing.

        Parameters
        ----------
        filename : str
            The filename of the file to open.

        Returns
        -------
        A context manager used for writing to the opened file.

        """
        pass


class DefaultReportFileManager(ReportFileManager):
    """A default implementation of the ReportFileManager class. This implementation uses the built-in open function to open files.
    """

    def open_input_file(self, filename: str):
        return open(filename, "r")

    def open_output_file(self, filename: str):
        return open(filename, "w")


class LuigiReportFileManager(ReportFileManager):
    def __init__(self, luigi_input_targets: Dict[str, luigi.Target], luigi_output_targets: Dict[str, luigi.Target]):
        self.__luigi_input_targets = luigi_input_targets
        self.__luigi_output_targets = luigi_output_targets

    @property
    def luigi_input_targets(self) -> Dict[str, luigi.Target]:
        return self.__luigi_input_targets

    @property
    def luigi_output_targets(self) -> Dict[str, luigi.Target]:
        return self.__luigi_output_targets

    def open_input_file(self, filename: str):
        return self.luigi_input_targets[filename].open("r")

    def open_output_file(self, filename: str):
        return self.luigi_output_targets[filename].open("w")


class ReportEngine:
    def __init__(self, settings_filename: str):
        self.__settings_filename = settings_filename
        self.__reports = None
        self.__all_report_permutations = dict()

    @property
    def settings_filename(self) -> str:
        return self.__settings_filename

    @property
    def report_settings(self) -> Settings:
        return Settings.load(self.settings_filename,
                             "report_settings",
                             False)

    @property
    def reports_config(self) -> Settings:
        return Settings.load(self.settings_filename,
                             "reports",
                             True,
                             **self.report_settings)

    @property
    def flattened_reports(self) -> List[Report]:
        return [report
                for _, report_list in self.reports.items()
                for report in report_list]

    @property
    def all_report_permutations(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary with all possible permutations of the reports.
        The permutations are based on the reports_config settings and the report_templates settings.

        """
        if not self.__all_report_permutations:
            self.__all_report_permutations = defaultdict(list)
            for report_id, report_id_settings in self.reports_config.items():
                keys, values = zip(*report_id_settings.items())
                all_report_settings = [dict(zip(keys, combination))
                                       for combination in itertools.product(*values)]
                for all_report_dict in all_report_settings:
                    # Retrieve the generic report settings section
                    template_settings = self.report_settings.copy()
                    template_settings.update(
                        all_report_dict)

                    all_report_templates = Settings.load(
                        self.settings_filename, "report_templates", True, **template_settings)
                    print(all_report_templates)
                    if report_id not in all_report_templates:
                        print(f"No report template found for {report_id}")
                        continue

                    report_template = all_report_templates[report_id]
                    input_filename = report_template["input_filename"]
                    self.__all_report_permutations[input_filename].append(
                        report_template)

        # Combine the permutations with the report template settings
        return self.__all_report_permutations

    @property
    def reports(self) -> Dict[str, List['Report']]:
        if not self.__reports:
            self.__reports = defaultdict(list)
            for report_key, report_settings_list in self.all_report_permutations.items():
                for report_settings in report_settings_list:
                    report_template = report_settings["reports"]
                    if isinstance(report_template, list):
                        for settings in report_template:
                            self.__reports[report_key].append(
                                self.report_for(settings))
                    else:
                        self.__reports[report_key].append(
                            self.report_for(report_template))
        return self.__reports

    def report_for(self, result_settings: Dict[str, Any]) -> 'Report':
        if result_settings["type"].lower() == ReportType.plot.value:
            return PlotReport(result_settings)
        elif result_settings["type"].lower() == ReportType.table.value:
            return TableReport(result_settings)
        else:
            raise ValueError(f"Unknown report type: {result_settings['type']}")

    def reports_for_path(self,
                         data_path: str,
                         file_extension: str = ".parquet",
                         parquet_engine: str = "pyarrow",
                         report_file_manager: ReportFileManager = DefaultReportFileManager()):
        file_index = FileIndex(data_path, file_extension)
        files_for_reports = {file_key: file_path
                             for file_key, file_path in file_index.files.items()
                             if file_key in self.reports.keys()}

        self.reports_for_file_index(files_for_reports,
                                    parquet_engine=parquet_engine,
                                    report_file_manager=report_file_manager)

    def reports_for_file_index(self,
                               files_for_reports: Dict[str, str],
                               parquet_engine: str = "pyarrow",
                               report_file_manager: ReportFileManager = DefaultReportFileManager()):
        print("Number of files: ", len(files_for_reports))
        print("Number of reports: ", len(self.reports))

        print("Missing files: ", set(self.reports.keys()) -
              set(files_for_reports.keys()))
        with tqdm.tqdm(total=len(files_for_reports)) as progress_bar:
            for file_key, _ in files_for_reports.items():
                if file_key not in self.reports:
                    progress_bar.update(1)
                    continue

                with report_file_manager.open_input_file(file_key) as input_file:
                    dataframe = pd.read_parquet(
                        input_file, engine=parquet_engine)

                reports = self.reports[file_key]
                for report in reports:
                    progress_bar.set_description(
                        f"Writing {report.output_filename}")

                    with report_file_manager.open_output_file(report.output_filename) as output_file:
                        self.create_directory(report.output_filename)
                        preprocessed_dataframe = self.preprocess_data(
                            dataframe, report.preprocessing_settings)
                        report.write_to_file(
                            preprocessed_dataframe, output_file)
                progress_bar.update(1)

    def preprocess_data(self, dataframe: pd.DataFrame, preprocessing_settings: Dict[str, Any]) -> pd.DataFrame:
        # if len(set(["pivot", "select_columns", "sort_values"]).intersection(set(preprocessing_settings.keys()))) > 0:
        copied_dataframe = dataframe.copy()

        if "pivot" in preprocessing_settings:
            value_columns = preprocessing_settings["value_columns"]
            copied_dataframe = unpivot(copied_dataframe, value_columns)

        if "select_columns" in preprocessing_settings:
            select_columns = preprocessing_settings["select_columns"]
            copied_dataframe = copied_dataframe[select_columns]

        if "sort_values" in preprocessing_settings:
            sort_settings = preprocessing_settings["sort_values"]
            columns = sort_settings.get("columns", [])
            ascending = sort_settings.get("ascending", True)

            copied_dataframe = copied_dataframe.sort_values(
                by=columns, ascending=ascending)

        if "bool_to_int" in preprocessing_settings:
            if preprocessing_settings["bool_to_int"]:
                copied_dataframe.replace({False: 0, True: 1}, inplace=True)
        if "to_categorical" in preprocessing_settings:
            columns = sort_settings.get("columns", [])
            for column in columns:
                copied_dataframe[column] = copied_dataframe[column].astype(
                    "category")

        return copied_dataframe

    def create_directory(self, filename: str):
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
