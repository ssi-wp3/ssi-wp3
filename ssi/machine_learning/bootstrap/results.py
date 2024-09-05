import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List


def combine_results(stores: List[str], dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """ Combine results from different stores into one DataFrame.

    Parameters
    ----------

    stores : List[str]
        List of store names.

    dataframes : List[pd.DataFrame]
        List of DataFrames containing the results for each store.

    Returns
    -------
        DataFrame containing the combined results.
    """
    for store, df in zip(stores, dataframes):
        df['store'] = store
    return pd.concat(dataframes)


def combine_files(files: List[str], delimiter: str = ";") -> pd.DataFrame:
    """ Combine results from different files into one DataFrame.

    Parameters 
    ----------
    files : List[str]
        List of file names.

    delimiter : str
        Delimiter to use for reading the files.

    Returns
    -------
        DataFrame containing the combined results.

    """
    store_names = [os.path.basename(file).split("_")[0] for file in files]
    dataframes = [pd.read_csv(file, delimiter=delimiter) for file in files]

    return combine_results(store_names, dataframes)


def boxplot_metrics(data: pd.DataFrame,
                    plot_filename: str,
                    metric_name: str,
                    group_name: str,
                    ) -> None:
    """ Create a boxplot for the given metrics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.

    metric_name : str
        Name of the column containing the metric names.

    group_name : str
        Name of the column containing the group names.    

    Returns
    -------
        None
    """
    sns.boxplot(x=metric_name, y=group_name, data=data)
    plt.xticks(rotation=45)
    plt.savefig(plot_filename)


def boxplot_files(files: List[str],
                  delimiter: str,
                  plot_filename: str,
                  metric_name: str,
                  group_name: str = 'store') -> None:
    """ Create a boxplot for the given metrics in the files.

    Parameters
    ----------
    files : List[str]
        List of file names.

    delimiter : str
        Delimiter to use for reading the files.

    plot_filename : str
        Filename to save the plot.

    metrics : List[str]
        List of metrics to include in the boxplot.

    metric_name : str
        Name of the column containing the metric names.

    value_name : str
        Name of the column containing the metric values.

    Returns
    -------
        None
    """
    data = combine_files(files, delimiter)
    boxplot_metrics(data, plot_filename, metric_name, group_name)
