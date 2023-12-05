from typing import List
import pandas as pd
import os

def filter_coicop_level(dataframe: pd.DataFrame, coicop_level_column: str, coicop_level_value: str) -> pd.DataFrame:
    return dataframe[dataframe[coicop_level_column] == coicop_level_value]

def write_filtered_coicop_level_files(dataframe: pd.DataFrame, coicop_level_columns: List[str], data_directory: str, supermarket_name: str):
    for coicop_level_column in coicop_level_columns:    
        for coicop_level_value in dataframe[coicop_level_column].unique():
            filtered_dataframe = filter_coicop_level(
                dataframe, coicop_level_column, coicop_level_value)
            filtered_dataframe.to_parquet(
                os.path.join(data_directory, f"{supermarket_name}_{coicop_level_value}.parquet"), engine="pyarrow", index=False)