from typing import List
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


def combine_unique_column_values(filenames: List[str],
                                 output_filename: str,
                                 key_columns: List[str],
                                 parquet_engine: str = "pyarrow",
                                 batch_size: int = 1000
                                 ):
    """ Combine unique column values from multiple files into a single file.
    """
    unique_column_values = []
    # Read the files and remove duplicates per file
    for file_index, filename in enumerate(filenames):
        df = pd.read_parquet(
            filename, columns=key_columns, engine=parquet_engine)
        df["file_index"] = file_index
        df = df.drop_duplicates(subset=key_columns)
        df.index.name = "row_index"
        df = df.reset_index()
        unique_column_values.append(df)

    # Drop the duplicates in the combined DataFrame
    combined_df = pd.concat(unique_column_values)
    combined_df = combined_df.drop_duplicates(subset=key_columns)

    # Write the combined DataFrame to a new file
    for file_index, filename in enumerate(filenames):
        row_indices = combined_df[combined_df["file_index"]
                                  == file_index]["row_index"]

        number_of_rows_read = 0
        pq_writer = None
        read_dataset = pq.ParquetFile(filename, memory_map=True)

        with tqdm.tqdm(total=read_dataset.metadata.num_rows) as progress_bar:
            progress_bar.set_description(
                f"Writing unique values from {filename}")
            current_batch = None
            for batch in read_dataset.iter_batches():
                batch_df = batch.to_pandas()
                # Retrieve the row indices in the range of this batch
                batch_indices = row_indices[(row_indices >= number_of_rows_read) &
                                            (row_indices < number_of_rows_read + len(batch_df))]
                batch_indices = batch_indices - number_of_rows_read

                # Retrieve the rows in the range of this batch
                batch_rows = batch_df.loc[batch_indices]
                progress_bar.set_description(f"Wrote {len(batch_rows)} rows")
                current_batch = pd.concat(
                    [current_batch, batch_rows]) if current_batch is not None else batch_rows

                batch_table = pa.Table.from_pandas(
                    current_batch)

                if number_of_rows_read == 0:
                    pq_writer = pq.ParquetWriter(
                        output_filename, batch_table.schema)

                if len(current_batch) % batch_size == 0:
                    pq_writer.write_table(batch_table)
                    current_batch = None

                number_of_rows_read += len(batch)
                progress_bar.update(len(batch))

            # write last rows (if any)
            if current_batch is not None:
                pq_writer.write_table(batch_table)

            if pq_writer:
                pq_writer.close()

            if read_dataset:
                read_dataset.close()
