from typing import List
import pandas as pd


def unpivot(dataframe: pd.DataFrame,
            value_vars: List[str],
            var_name: str = 'group',
            value_name: str = 'value') -> pd.DataFrame:
    """Unpivot a DataFrame from wide to long format.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe to unpivot

    value_vars : List[str]
        The columns to unpivot

    var_name : str
        The name of the column containing the variable names

    value_name : str
        The name of the column containing the values

    Returns
    -------
    pd.DataFrame
        The unpivoted dataframe
    """
    return dataframe.melt(id_vars=dataframe.columns.difference(value_vars).tolist(),
                          value_vars=value_vars,
                          var_name=var_name,
                          value_name=value_name)
