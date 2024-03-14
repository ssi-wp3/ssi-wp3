from typing import List
import pandas as pd


def unpivot(dataframe: pd.DataFrame,
            value_vars: List[str],
            var_name: str = 'group',
            value_name: str = 'value') -> pd.DataFrame:
    """Unpivot a DataFrame from wide to long format."""
    print(dataframe.columns)
    return dataframe.melt(id_vars=dataframe.columns.difference(value_vars).tolist(),
                          value_vars=value_vars,
                          var_name=var_name,
                          value_name=value_name)
