from typing import List
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


def combine_unique_column_values(filenames: List[str],
                                 output_filename: str,
                                 key_columns: List[str],
                                 parquet_engine: str = "pyarrow"
                                 ):
    """ Combine unique column values from multiple files into a single file.
    """
    unique_column_values = []
    # Read the files and remove duplicates per file
    for file_index, filename in enumerate(filenames):
        df = pd.read_parquet(
            filename, columns=key_columns, engine=parquet_engine)
        df["filename"] = file_index
        df = df.drop_duplicates(subset=key_columns)
        df.index.name = "row_index"
        df = df.reset_index()
        unique_column_values.append(df)

    # Drop the duplicates in the combined DataFrame
    combined_df = pd.concat(unique_column_values)
    combined_df = combined_df.drop_duplicates(subset=key_columns)

    # Write the combined DataFrame to a new file
    for filename in filenames:
        row_indices = combined_df[combined_df["filename"]
                                  == filename]["row_index"]

        number_of_rows_read = 0
        pq_writer = None
        read_dataset = pq.ParquetDataset(filename, memory_map=True)

        with tqdm.tqdm(total=read_dataset.metadata.num_rows) as progress_bar:
            progress_bar.set_description(
                f"Writing unique values from {filename}")
            for batch in read_dataset.iter_batches(row_indices):
                batch_df = batch.to_pandas()
                # Retrieve the row indices in the range of this batch
                batch_indices = row_indices[row_indices >= number_of_rows_read &
                                            row_indices < number_of_rows_read + len(batch_df)]

                # Retrieve the rows in the range of this batch
                batch_rows = combined_df.loc[batch_indices]

                batch_table = pa.Table.from_pandas(batch_rows)
                if number_of_rows_read == 0:
                    pq_writer = pq.ParquetWriter(
                        filename, batch_table.schema)

                pq_writer.write_table(batch_table)
                number_of_rows_read += len(batch)
                progress_bar.update(len(batch))

        if read_dataset is not None:
            read_dataset.close()

        if pq_writer is not None:
            pq_writer.close()
