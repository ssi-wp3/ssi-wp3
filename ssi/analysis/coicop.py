from typing import List
from ..plots import PlotEngine, PlotBackend
import pandas as pd


def coicop_sunburst(dataframe: pd.DataFrame, coicop_columns: List[str], value_column: str) -> PlotBackend.FigureWrapper:
    """Returns a dictionary with the sunburst data for a coicop column in a dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe to process

    coicop_columns : List[str]
        The columns to traverse to make the sunburst plot, these columns should be in the order of
        the hierarchy to traverse. Each column gives a new level in the hierarchy, and a different
        grouping of the data.

    value_column : str
        The value column containing the values to plot in the sunburst plot.

    Returns
    -------

    PlotBackend.FigureWrapper
        The FigureWrapper object for the sunburst plot.
    """
    group_column = coicop_columns[-1]
    dataframe_with_counts = dataframe.groupby(
        group_column)["value_column"].nunique().reset_index(name="count")
    return PlotEngine().sunburst_chart(dataframe_with_counts, coicop_columns, "count")
