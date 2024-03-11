from typing import List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def line_chart(dataframe: pd.DataFrame,
               x_column: str,
               y_column: str,
               group_column: str = None,
               title: str = None
               ) -> go.Figure:
    return px.line(dataframe, x=x_column, y=y_column, color=group_column, title=title)


def bar_chart(dataframe: pd.DataFrame,
              x_column: str,
              y_column: str,
              group_column: str,
              title: str = None
              ) -> go.Figure:
    """ Create a bar chart with the given dataframe and x and y columns.
    The bar chart can be grouped by a column, indicated by the color of the bar.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data to plot.

    x_column : str
        The column to use as the x axis.

    y_column : str
        The column to use as the y axis.

    group_column : str
        The column to group the data into separate groups, indicated by the the color of the bars.

    title : str, optional
        The title of the plot.

    Returns
    -------
    go.Figure
        The bar chart figure.    
    """
    return px.bar(dataframe, x=x_column, y=y_column, color=group_column, title=title)


def scatter_plot(dataframe: pd.DataFrame,
                 x_column: str,
                 y_column: str,
                 group_column: str = None,
                 size_column: str = None,
                 title: str = None
                 ) -> go.Figure:
    """ Create a scatter plot with the given dataframe and x and y columns.
    The scatter plot can be grouped by a column, indicated by the color of the point, 
    and the size of the points can be determined by another column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data to plot.

    x_column : str
        The column to use as the x axis.

    y_column : str
        The column to use as the y axis.

    group_column : str, optional
        The column to group the data into separate groups, indicated by the the color of the points.

    size_column : str, optional
        The column to use as the size of the points.

    title : str, optional
        The title of the plot.

    Returns
    -------
    go.Figure
        The scatter plot figure.
    """
    return px.scatter(dataframe, x=x_column, y=y_column, color=group_column, size=size_column, title=title)


def sunburst_chart(dataframe: pd.DataFrame,
                   sunburst_columns: List[str],
                   amount_column: str
                   ) -> go.Figure:
    return px.sunburst(dataframe, path=sunburst_columns, values=amount_column)
