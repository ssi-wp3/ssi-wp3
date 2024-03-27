from typing import Dict, Any, List
from .utils import unpivot
from ..plots import PlotEngine
from ..settings import Settings
import pandas as pd


class Report:
    def __init__(self, plot_engine: PlotEngine):
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
