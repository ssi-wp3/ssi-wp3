from typing import List, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from abc import ABC, abstractmethod


class PlotBackend(ABC):
    """Abstract class for a plot backend that can be used to create different types of plots.
    The plot backend is used to create different types of plots, such as line charts, bar charts, etc.
    It hides the implementation details of the plot library used to create the plots, and provides
    a consistent interface to create different types of plots.
    """
    class FigureWrapper(ABC):
        """Abstract class for a figure wrapper that wraps a plot figure.
        The FigureWrapper provides a consistent interface to show and save the plot figure.
        It also provides access to the original plot figure via the figure property.
        """

        def __init__(self, figure):
            self.figure = figure

        @property
        def figure(self):
            return self._figure

        @abstractmethod
        def show(self):
            pass

        @abstractmethod
        def save(self, filename: str):
            pass

    @abstractmethod
    def line_chart(self, dataframe: pd.DataFrame,
                   x_column: str,
                   y_column: str,
                   group_column: str = None,
                   title: str = None
                   ) -> FigureWrapper:
        pass

    @abstractmethod
    def bar_chart(dataframe: pd.DataFrame,
                  x_column: str,
                  y_column: str,
                  group_column: str,
                  title: str = None
                  ) -> FigureWrapper:
        pass

    @abstractmethod
    def scatter_plot(dataframe: pd.DataFrame,
                     x_column: str,
                     y_column: str,
                     group_column: str = None,
                     size_column: str = None,
                     title: str = None
                     ) -> FigureWrapper:

        pass

    @abstractmethod
    def sunburst_chart(dataframe: pd.DataFrame,
                       sunburst_columns: List[str],
                       amount_column: str,
                       title: str = None
                       ) -> FigureWrapper:
        pass

    @abstractmethod
    def heatmap(self,
                dataframe: pd.DataFrame,
                show_text: bool = False,
                title: str = None
                ) -> FigureWrapper:
        pass

    @abstractmethod
    def parallel_coordinates_plot(self,
                                  dataframe: pd.DataFrame,
                                  dimensions: List[str],
                                  color_column: str,
                                  dimension_titles: List[str] = None,
                                  title: str = None,
                                  color_continuous_scale: str = px.colors.diverging.Tealrose,
                                  color_continuous_midpoint: float = 2
                                  ) -> FigureWrapper:
        pass

    @abstractmethod
    def histogram(self,
                  dataframe: pd.DataFrame,
                  x_column: str,
                  number_of_bins: Optional[int] = None,
                  title: Optional[str] = None,
                  category_orders: Optional[str] = None
                  ) -> FigureWrapper:
        pass

    @abstractmethod
    def box_plot(self,
                 dataframe: pd.DataFrame,
                 y_column: str,
                 x_column: Optional[str] = None,
                 title: Optional[str] = None
                 ) -> FigureWrapper:
        pass


class PlotlyBackend(PlotBackend):
    """Plot backend that uses Plotly to create different types of plots.
    The PlotlyBackend provides an implementation for creating different types of plots,
    such as line charts, bar charts, etc. using Plotly.
    """
    class PlotlyFigureWrapper(PlotBackend.FigureWrapper):
        """Figure wrapper that wraps a Plotly figure.
        """

        def show(self):
            self.figure.show()

        def save(self, filename: str):
            if filename.endswith(".html"):
                self.figure.write_html(filename)
            else:
                self.figure.write_image(filename)

    def line_chart(self,
                   dataframe: pd.DataFrame,
                   x_column: str,
                   y_column: str,
                   group_column: str = None,
                   title: str = None
                   ) -> PlotBackend.FigureWrapper:
        return self.__figure_wrapper_for(
            px.line(dataframe, x=x_column, y=y_column,
                    color=group_column, title=title)
        )

    def bar_chart(self,
                  dataframe: pd.DataFrame,
                  x_column: str,
                  y_column: str,
                  group_column: str,
                  title: str = None
                  ) -> PlotBackend.FigureWrapper:
        return self.__figure_wrapper_for(
            px.bar(dataframe, x=x_column, y=y_column,
                   color=group_column, title=title)
        )

    def scatter_plot(self, dataframe: pd.DataFrame,
                     x_column: str,
                     y_column: str,
                     group_column: str = None,
                     size_column: str = None,
                     title: str = None
                     ) -> PlotBackend.FigureWrapper:
        return self.__figure_wrapper_for(
            px.scatter(dataframe, x=x_column, y=y_column,
                       color=group_column, size=size_column, title=title)
        )

    def sunburst_chart(self,
                       dataframe: pd.DataFrame,
                       sunburst_columns: List[str],
                       amount_column: str,
                       title: str = None
                       ) -> PlotBackend.FigureWrapper:
        return self.__figure_wrapper_for(
            px.sunburst(dataframe, path=sunburst_columns,
                        values=amount_column, title=title)
        )

    def heatmap(self,
                dataframe: pd.DataFrame,
                show_text: bool = False,
                title: str = None
                ) -> go.Figure:
        return self.__figure_wrapper_for(
            px.imshow(dataframe, text_auto=show_text, title=title)
        )

    def parallel_coordinates_plot(self,
                                  dataframe: pd.DataFrame,
                                  dimensions: List[str],
                                  color_column: str,
                                  dimension_titles: List[str] = None,
                                  title: str = None,
                                  color_continuous_scale: str = px.colors.diverging.Tealrose,
                                  color_continuous_midpoint: float = 2
                                  ) -> go.Figure:
        if not dimension_titles:
            figure = px.parallel_coordinates(dataframe,
                                             dimensions=dimensions,
                                             color=color_column,
                                             title=title,
                                             color_continuous_scale=color_continuous_scale,
                                             color_continuous_midpoint=color_continuous_midpoint)
        else:
            labels = {dim: title for dim, title in zip(
                dimensions, dimension_titles)}
            figure = px.parallel_coordinates(dataframe,
                                             color=color_column,
                                             labels=labels,
                                             title=title,
                                             color_continuous_scale=color_continuous_scale,
                                             color_continuous_midpoint=color_continuous_midpoint,
                                             )
        return self.__figure_wrapper_for(figure)

    def histogram(self,
                  dataframe: pd.DataFrame,
                  x_column: str,
                  number_of_bins: Optional[int] = None,
                  title: Optional[str] = None,
                  category_orders: Optional[str] = None
                  ) -> PlotBackend.FigureWrapper:
        return self.__figure_wrapper_for(
            px.histogram(dataframe, x=x_column, nbins=number_of_bins,
                         title=title, category_orders=category_orders)
        )

    def box_plot(self,
                 dataframe: pd.DataFrame,
                 y_column: str,
                 x_column: Optional[str] = None,
                 title: Optional[str] = None
                 ) -> PlotBackend.FigureWrapper:
        return self.__figure_wrapper_for(
            px.box(dataframe, x=x_column, y=y_column, title=title)
        )

    def __figure_wrapper_for(self, figure):
        return PlotlyBackend.PlotlyFigureWrapper(figure)


class PlotEngine(PlotBackend):
    """Facade class for creating different types of plots using a plot backend.
    The user of the PlotEngine can create plots without worrying about the implementation details
    of the plot library used to create the plots. The PlotEngine provides a consistent interface
    to create different types of plots, such as line charts, bar charts, etc.

    For now the PlotEngine only supports Plotly as the plot backend, but it can be extended to
    support other plot backends, such as Matplotlib, Seaborn, etc.
    """

    def __init__(self, plot_backend: PlotBackend = PlotlyBackend()):
        """ Constructor

        Parameters
        ----------
        plot_backend : PlotBackend
            The plot backend to use to create the plots. By default, it uses the PlotlyBackend.
        """
        self._plot_backend = plot_backend

    @property
    def plot_backend(self):
        return self._plot_backend

    def line_chart(self,
                   dataframe: pd.DataFrame,
                   x_column: str,
                   y_column: str,
                   group_column: str = None,
                   title: str = None
                   ) -> PlotBackend.FigureWrapper:
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
        PlotBackend.FigureWrapper
            The line chart figure.
        """
        return self.plot_backend.line_chart(dataframe, x_column, y_column,
                                            group_column, title)

    @abstractmethod
    def bar_chart(self,
                  dataframe: pd.DataFrame,
                  x_column: str,
                  y_column: str,
                  group_column: str,
                  title: str = None
                  ) -> PlotBackend.FigureWrapper:
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
        PlotBackend.FigureWrapper
            The bar chart figure.    
        """
        return self.plot_backend.bar_chart(dataframe, x_column, y_column,
                                           group_column, title)

    @abstractmethod
    def scatter_plot(self,
                     dataframe: pd.DataFrame,
                     x_column: str,
                     y_column: str,
                     group_column: str = None,
                     size_column: str = None,
                     title: str = None
                     ) -> PlotBackend.FigureWrapper:
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
        PlotBackend.FigureWrapper
            The scatter plot figure.
        """
        return self.plot_backend.scatter_plot(dataframe, x_column, y_column,
                                              group_column, size_column, title)

    def histogram(self,
                  dataframe: pd.DataFrame,
                  x_column: str,
                  number_of_bins: Optional[int] = None,
                  title: Optional[str] = None,
                  category_orders: Optional[str] = None
                  ) -> PlotBackend.FigureWrapper:
        """ Create a histogram with the given dataframe and x column.
        The histogram can be used to visualize the distribution of a single variable.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the data to plot.

        x_column : str
            The column to use as the x axis.

        number_of_bins : int, optional
            The number of bins to use in the histogram.

        title : str, optional
            The title of the plot.

        category_orders : str, optional
            The order of the categories in the histogram.

        Returns
        -------
        PlotBackend.FigureWrapper
            The histogram figure.
        """
        return self.plot_backend.histogram(dataframe,
                                           x=x_column,
                                           number_of_bins=number_of_bins,
                                           title=title,
                                           category_orders=category_orders)

    def box_plot(self,
                 dataframe: pd.DataFrame,
                 y_column: str,
                 x_column: Optional[str] = None,
                 title: Optional[str] = None
                 ) -> PlotBackend.FigureWrapper:
        """ Create a box plot with the given dataframe and y column.
        The box plot can be used to visualize the distribution of a variable.
        Optionally, an x column can be provided to group the data into separate boxes.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the data to plot.

        y_column : str
            The column to use as the y axis, i.e. the column containing the values.

        x_column : str
            The column to use as the x axis.

        title : str, optional
            The title of the plot.

        Returns
        -------
        PlotBackend.FigureWrapper
            The box plot figure.
        """
        return self.plot_backend.box_plot(dataframe, y_column, x_column, title)

    @abstractmethod
    def sunburst_chart(self,
                       dataframe: pd.DataFrame,
                       sunburst_columns: List[str],
                       amount_column: str,
                       title: str = None
                       ) -> PlotBackend.FigureWrapper:
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
        PlotBackend.FigureWrapper
            The sunburst chart figure.
        """
        return self.plot_backend.sunburst_chart(dataframe, sunburst_columns, amount_column, title)

    def heatmap(self,
                dataframe: pd.DataFrame,
                show_text: bool = False,
                title: str = None
                ) -> PlotBackend.FigureWrapper:
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
        PlotBackend.FigureWrapper
            The heatmap chart for the dataframe.
        """
        return self.plot_backend.heatmap(dataframe, show_text, title)

    def parallel_coordinates_plot(self,
                                  dataframe: pd.DataFrame,
                                  dimensions: List[str],
                                  color_column: str,
                                  dimension_titles: List[str] = None,
                                  title: str = None,
                                  color_continuous_scale: str = px.colors.diverging.Tealrose,
                                  color_continuous_midpoint: float = 2
                                  ) -> PlotBackend.FigureWrapper:
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
        PlotBackend.FigureWrapper
            The parallel coordinates plot figure.
        """
        return self.plot_backend.parallel_coordinates_plot(dataframe, dimensions, color_column,
                                                           dimension_titles, title, color_continuous_scale,
                                                           color_continuous_midpoint)
