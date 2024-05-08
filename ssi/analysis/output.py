from typing import Any, Dict
from ..plots import PlotEngine
from ..settings import Settings
import pandas as pd


class ResultsWriter:
    def __init__(self,
                 result_settings_filename: str,
                 result_settings_section: str,  **kwargs):
        self.__result_settings_filename = result_settings_filename
        self.__result_settings_section = result_settings_section
        self.__dict__.update(kwargs)

    @property
    def result_settings_filename(self) -> str:
        return self.__result_settings_filename

    @property
    def result_settings_section(self) -> str:
        return self.__result_settings_section

    @property
    def plot_engine(self) -> PlotEngine:
        return PlotEngine()

    @property
    def plot_settings(self) -> Dict[str, Any]:
        # TODO split up in plots that need to be run once for the whole dataset, plots that needt to be run for each
        # COICOP level, and plots that need to be run for each period.
        # TODO Add sunburst with number of products (EAN/Receipt texts) per coicop
        # TODO Add sunburst with total products sold/revenue?
        settings = Settings.load(self.result_settings_filename,
                                 self.result_settings_section,
                                 True,

                                 )

        return settings

    def write_results(self,
                      dataframe: pd.DataFrame,
                      filename: str,
                      **kwargs):
        pass
