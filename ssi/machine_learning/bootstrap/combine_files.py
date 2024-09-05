from .results import combine_files
import argparse
import os
import pandas as pd

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Combine CSV files into a Parquet file')

    # Add arguments for input directory and output filename
    parser.add_argument("-i", '--input-directory', required=True,
                        help='Path to the input directory containing CSV files')
    parser.add_argument("-o", '--output-filename',
                        default="combined_lr_bootstraps.parquet", help='Name of the output Parquet file')
    parser.add_argument('-e', '--engine', default='pyarrow',
                        help='Parquet engine to use (default: pyarrow)')
    parser.add_argument('-d', '--delimiter', default=';',
                        help='Delimiter to use for reading the files (default: ;)')
    args = parser.parse_args()

    # Call the combine_csv_files function with the provided arguments
    csv_files = [file for file in os.listdir(
        args.input_directory) if file.endswith('.csv')]
    dataframe = combine_files(csv_files)
    dataframe.to_parquet(args.output_filename, engine=args.engine)
