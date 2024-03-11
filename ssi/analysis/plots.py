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
    """ Create a line chart with the given dataframe and x and y columns.
    Using the group column, data can be grouped into separate lines in the plot.
    Each separate line/group has a different line color.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data to plot.

    x_column : str
        The column to use as the x axis.

    y_column : str
        The column to use as the y axis.

    group_column : str, optional
        The column to group the data into separate lines.

    title : str, optional
        The title of the plot.

    Returns
    -------
    go.Figure
        The line chart figure.
    """
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
                   amount_column: str,
                   title: str = None
                   ) -> go.Figure:
    """ Create a sunburst chart with the given dataframe and sunburst columns.
    The sunburst chart can be used to visualize hierarchical data, the sunburst columns
    list are the column names that indicated the levels of the hierarchy, and the amount
    column is the column that contains the size of the sunburst section to be visualized.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data to plot.

    sunburst_columns : List[str]
        The list of columns that indicate the levels of the hierarchy.

    amount_column : str
        The column that contains the size of the sunburst section to be visualized.

    title : str, optional
        The title of the plot.

    Returns
    -------
    go.Figure
        The sunburst chart figure.
    """
    return px.sunburst(dataframe, path=sunburst_columns, values=amount_column, title=title)


def heatmap(dataframe: pd.DataFrame,
            show_text: bool = False,
            title: str = None
            ) -> go.Figure:
    """ Create a heatmap with the given dataframe and x, y and z columns.
    The heatmap can be used to visualize the relationship between two variables, and the
    intensity of the relationship is indicated by the color of the cells.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data to plot. The dataframe should be in 2D matrix form.

    show_text : bool, optional
        Whether to show the cell values in the heatmap.

    title : str, optional
        The title of the plot.

    Returns
    -------
    go.Figure
        The heatmap chart for the dataframe.
    """
    return px.imshow(dataframe, text_auto=show_text, title=title)


def parallel_coordinates_plot(dataframe: pd.DataFrame,
                              dimensions: List[str],
                              color_column: str,
                              dimension_titles: List[str] = None,
                              title: str = None,
                              color_continuous_scale: str = px.colors.diverging.Tealrose,
                              color_continuous_midpoint: float = 2
                              ) -> go.Figure:
    """ Create a parallel coordinates plot with the given dataframe and dimensions.
    The parallel coordinates plot can be used to visualize the relationship between multiple variables,
    and the color column can be used to group the data into separate lines.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data to plot.

    dimensions : List[str]
        The list of columns to use as the dimensions of the parallel coordinates plot.

    color_column : str
        The value column to plot in the parallel coordinate plot. This column often contains
        an evaluate metric (i.e accuracy, F1, etc.) that is plotted against a number of dimensions
        that may influence the metric (i.e. hyperparameters). The color of the line will indicate
        the height of the metric.

    dimension_titles : List[str], optional
        The titles to use for the names of the dimensions (instead of the column names).
        If not provided, the column names will be used.

    color_continuous_scale : str, optional
        The color scale to use for the color column.

    color_continuous_midpoint : float, optional
        The midpoint of the color scale.

    title : str, optional
        The title of the plot.

    Returns
    -------
    go.Figure
        The parallel coordinates plot figure.
    """
    if not dimension_titles:
        return px.parallel_coordinates(dataframe,
                                       dimensions=dimensions,
                                       color=color_column,
                                       title=title,
                                       color_continuous_scale=color_continuous_scale,
                                       color_continuous_midpoint=color_continuous_midpoint)
    labels = {dim: title for dim, title in zip(dimensions, dimension_titles)}
    return px.parallel_coordinates(dataframe,
                                   color=color_column,
                                   labels=labels,
                                   title=title,
                                   color_continuous_scale=color_continuous_scale,
                                   color_continuous_midpoint=color_continuous_midpoint,
                                   )
