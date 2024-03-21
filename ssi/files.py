from typing import List, Optional, Callable, Any
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm


def get_feature_filename(feature_extractor_type: str,
                         supermarket_name: str,
                         project_prefix: str = "ssi",
                         feature_file_suffix: str = "features") -> str:
    return f"{project_prefix}_{supermarket_name.lower()}_{feature_extractor_type}_{feature_file_suffix}.parquet"


def get_features_files_in_directory(directory: str,
                                    project_prefix: str = "ssi",
                                    feature_file_suffix: str = "features",
                                    extension: str = ".parquet") -> List[str]:
    return [filename
            for filename in os.listdir(directory)
            if filename.startswith(f"{project_prefix}_") and f"_{feature_file_suffix}" in filename and filename.endswith(extension)]


def batched_writer(filename: str,
                   dataframe: pd.DataFrame,
                   batch_size: int,
                   process_batch: Callable[[pd.DataFrame], pd.DataFrame],
                   batch_statistics: Optional[Callable[[
                       pd.DataFrame], Any]] = None,
                   **process_batch_kwargs) -> List[Any]:
    """Process a DataFrame in batches and write it to a Parquet file batch per batch.
    Use this function to avoid memory issues when writing large DataFrames to Parquet files.

    Parameters:
    ----------

    filename: str
        The name of the Parquet file to write to.

    dataframe: pd.DataFrame
        The DataFrame to write to the Parquet file.

    batch_size: int
        The size of each batch to process.

    process_batch: Callable[[pd.DataFrame], pd.DataFrame]
        A function that processes a batch of the DataFrame. This function should return a DataFrame.

    batch_statistics: Optional[Callable[[pd.DataFrame], None]]
        A function that can be used to calculate statistics for each batch. 

    **process_batch_kwargs
        Additional keyword arguments to pass to the process_batch function.

    """
    pq_writer = None
    batch_statistics_results = []
    with tqdm(total=len(dataframe)) as progress_bar:
        for i in range(0, len(dataframe), batch_size):
            batch_df = dataframe.iloc[i:i+batch_size].copy()
            processed_batch_df = process_batch(
                batch_df, progress_bar=progress_bar, **process_batch_kwargs)

            if batch_statistics:
                batch_statistics_results.append(
                    batch_statistics(processed_batch_df))

            table = pa.Table.from_pandas(processed_batch_df)
            if i == 0:
                pq_writer = pq.ParquetWriter(filename, table.schema)
            pq_writer.write_table(table)

            progress_bar.update(batch_size)

    if pq_writer:
        pq_writer.close()

    return batch_statistics_results
