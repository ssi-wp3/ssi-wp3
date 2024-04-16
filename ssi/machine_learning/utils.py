from typing import List, Tuple
import pyarrow.parquet as pq
import pandas as pd
import itertools


def store_combinations(available_stores: List[str]) -> List[Tuple[str, str]]:
    """Create all combinations of two stores from a list of available stores.

    Parameters
    ----------
    available_stores : List[str]
        A list of available stores

    Returns
    -------
    List[Tuple[str, str]]
        A list of all combinations of two stores
    """
    return list(itertools.combinations(available_stores, 2))


def read_parquet_indices(store_file: str,
                         indices: pd.Series,
                         columns: List[str]
                         ) -> pd.DataFrame:
    """Read only the rows with the given indices from a parquet file.
    The parquet file is read in batches to reduce memory usage. For each 
    row only the columns specified are read.

    Parameters
    ----------
    store_file : str
        The filename of the parquet file

    indices : List[int]
        The indices of the rows to read from the parquet file.

    columns : List[str]
        The columns to read from the parquet file

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the rows with the given indices
    """
    parquet = pq.ParquetFile(store_file, memory_map=True)

    batch_start = 0
    data = []
    for batch in parquet.iter_batches(columns=columns):
        batch_indices = indices[(indices >= batch_start) & (
            indices < batch_start + batch.num_rows)]
        batch_indices -= batch_start
        data.append(batch.to_pandas().iloc[batch_indices])
        batch_start += batch.num_rows
    return pd.concat(data)


def read_unique_rows(
        filename: str,
        group_columns: List[str],
        value_columns: List[str] = None,
        engine: str = "pyarrow"
) -> pd.DataFrame:
    """Read only the unique rows from a file. The unique files are determined
    by the columns specified. The file is read in batches to reduce memory usage.

    Parameters
    ----------
    filename : str
        The filename of the parquet file

    group_columns : List[str]
        The columns to use to determine the unique rows

    value_columns : List[str]
        The columns to read from the parquet file. If None the group_columns are used.

    engine : str
        The engine to use to read the parquet file

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the unique rows
    """
    dataframe = pd.read_parquet(filename, columns=group_columns, engine=engine)
    unique_rows = dataframe.drop_duplicates(group_columns)
    unique_indices = unique_rows.index

    value_columns = group_columns if not value_columns else value_columns
    return read_parquet_indices(dataframe, unique_indices, value_columns)
