import argparse
import pandas as pd
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-directory", required=True,
                    help="The directory to read the csv files from")
parser.add_argument("-o", "--output-directory", default=None,
                    help="The directory to read the parquet files to")
parser.add_argument("-e", "--extension", default=".xlsx",
                    help="The extension for the xlsx files")
args = parser.parse_args()

input_filenames = [os.path.join(args.input_directory, filename)
                   for filename in os.listdir(args.input_directory)
                   if filename.endswith(args.extension)]
output_directory = args.input_directory if not args.output_directory else args.output_directory

print(
    f"Reading all {args.extension} files from {args.input_directory} and writing them to {output_directory}")
progress_bar = tqdm.tqdm(input_filenames)
for input_filename in progress_bar:
    filename = os.path.basename(input_filename).replace(args.extension, "")
    output_filename = os.path.join(output_directory, f"{filename}.parquet")
    progress_bar.set_description(
        f"Writing {input_filename} to {output_filename}")
    df = pd.read_excel(input_filename, sep=args.delimiter, engine="openpyxl")
    df.to_parquet(output_filename, engine="pyarrow")
