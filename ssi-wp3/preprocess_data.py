from typing import List
import pandas as pd
import os


def split_coicop(coicop_column: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"coicop_number": coicop_column,
                         "coicop_division": coicop_column.str[:2],
                         "coicop_group": coicop_column.str[:3],
                         "coicop_class": coicop_column.str[:4],
                         "coicop_subclass": coicop_column.str[:5],
                         "coicop_subsubclass": coicop_column,
                         })


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    pass


def combine_revenue_files(revenue_files: List[str], sort_columns: List[str], sort_order: List[bool],  engine: str = "pyarrow") -> pd.DataFrame:
    lidl_df = pd.concat([pd.read_parquet(revenue_file, engine=engine)
                        for revenue_file in revenue_files])
    lidl_df = lidl_df.sort_values(
        by=sort_columns, ascending=sort_order).reset_index(drop=True)
    return lidl_df


def combine_revenue_files_in_folder(data_directory: str, supermarket_name: str, sort_columns: List[str], sort_order: List[bool], filename_prefix: str = "Omzet") -> pd.DataFrame:
    lidl_revenue_files = [os.path.join(data_directory, filename) for filename in os.listdir(
        data_directory) if filename.startswith(filename_prefix) and supermarket_name in filename]
    return combine_revenue_files(lidl_revenue_files, sort_columns=sort_columns)


def save_combined_revenue_files(data_directory: str, output_filename: str, supermarket_name: str, filename_prefix: str = "Omzet", engine: str = "pyarrow"):
    combined_df = combine_revenue_files_in_folder(
        data_directory, supermarket_name, filename_prefix, sort_columns=["bg_number", "month", "coicop_number"], sort_order=[True, True, True])
    combined_df.to_parquet(os.path.join(
        data_directory, output_filename), engine=engine)
