from dotenv import load_dotenv
import argparse
import pandas as pd
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-directory",
                    help="The directory to read the csv files from")
parser.add_argument("-o", "--output-directory", default=None,
                    help="The directory to read the parquet files to")
parser.add_argument("-d", "--delimiter", default=";",
                    help="The delimiter used in the csv files")
parser.add_argumens("-ec", "--encoding", default="ISO-8859-1",
                    help="The encoding of the csv files (default: ISO-8859-1)")
parser.add_argument("-e", "--extension", default=".csv",
                    help="The extension for the csv files")
args = parser.parse_args()

load_dotenv()

input_directory = os.getenv(
    "INPUT_DIRECTORY") if not args.input_directory else args.input_directory

# If no output directory is specified, try to get output directory from env
# If no output directory is specified in env, use input directory.
output_directory = os.getenv(
    "OUTPUT_DIRECTORY") if not args.output_directory else args.output_directory
output_directory = input_directory if not output_directory else output_directory

input_filenames = [os.path.join(input_directory, filename)
                   for filename in os.listdir(input_directory)
                   if filename.endswith(args.extension)]

print(
    f"Reading all {args.extension} files from {input_directory} and writing them to {output_directory}")
progress_bar = tqdm.tqdm(input_filenames)
for input_filename in progress_bar:
    filename = os.path.basename(input_filename).replace(args.extension, "")
    output_filename = os.path.join(output_directory, f"{filename}.parquet")
    progress_bar.set_description(
        f"Writing {input_filename} to {output_filename}")
    df = pd.read_excel(input_filename, sep=args.delimiter,
                       engine="pyarrow", encoding=args.encoding)
    df.to_parquet(output_filename, engine="pyarrow")
