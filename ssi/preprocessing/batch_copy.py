from typing import Callable
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import pandas as pd


def batch_copy(input_filename: str,
               row_indices_to_copy: pd.DataFrame,
               output_filename: str,
               batch_filter_function: Callable[[
                   pd.DataFrame], pd.DataFrame] = None,
               batch_size: int = 1024):
    row_indices = row_indices_to_copy["row_index"]

    read_dataset = pq.ParquetFile(input_filename)  # , memory_map=True)

    total_number_of_rows_read = 0
    total_number_of_rows_written = 0
    with tqdm.tqdm(total=read_dataset.metadata.num_rows) as progress_bar:
        progress_bar.set_description(
            f"Copying rows from {input_filename}")
        for batch in read_dataset.iter_batches():
            batch_df = batch.to_pandas()
            # Retrieve the row indices in the range of this batch
            batch_indices = row_indices[(row_indices >= total_number_of_rows_read) &
                                        (row_indices < total_number_of_rows_read + len(batch_df))]
            batch_indices = batch_indices - total_number_of_rows_read

            # Retrieve the rows in the range of this batch
            batch_rows = batch_df.loc[batch_indices]
            if batch_filter_function:
                batch_rows = batch_filter_function(batch_rows)

            if len(batch_rows) == 0:
                progress_bar.set_description("Skipping empty batch")
                progress_bar.update(len(batch))
                total_number_of_rows_read += len(batch_df)
                continue

            progress_bar.set_description(
                f"Wrote {len(batch_rows)} rows for {len(batch_indices)} unique values")
            batch_table = pa.Table.from_pandas(
                batch_rows)

            if pq_writer:
                batch_table = batch_table.cast(pq_writer.schema)

            if total_number_of_rows_written == 0:
                pq_writer = pq.ParquetWriter(
                    output_filename, batch_table.schema, write_batch_size=batch_size)

            pq_writer.write_table(batch_table)
            total_number_of_rows_written += len(batch_rows)

            total_number_of_rows_read += len(batch_df)
            total_number_of_rows_read += len(batch)

            progress_bar.update(len(batch))

        if read_dataset:
            read_dataset.close()

    if pq_writer:
        pq_writer.close()

    print(
        f"Number of rows written: {total_number_of_rows_written} out of {len(row_indices_to_copy)} unique values")
