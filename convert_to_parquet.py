from dotenv import load_dotenv
from typing import Dict, Any, Optional
from collections import OrderedDict
import argparse
import pandas as pd
import os
import tqdm
import numpy as np


def get_column_types(filename: str) -> Optional[OrderedDict[str, Any]]:
    # From Justin's code, but converted them to english and lower case
    column_types = OrderedDict([
        ('bg_number', str),
        ('month', str),
        ('coicop_number', str),
        ('coicop_name', str),
        ('isba_number', str),
        ('isba_name', str),
        ('esba_number', str),
        ('esba_name', str),
        ('rep_id', str),
        ('ean_number', str),
        ('ean_name', str),
        ('revenue', np.float32),
        ('amount', np.float32)
    ])

    if filename.lower().startswith("omzeteans"):
        return column_types
    elif filename.lower().startswith("output"):
        return OrderedDict([(column_name, column_types[column_name])
                            for column_name in ['bg_number', 'coicop_number', 'coicop_name', 'isba_number', 'isba_name', 'esba_number', 'esba_name', 'rep_id', 'ean_number', 'ean_name']
                            ])
    elif filename.lower().startswith("kassabon"):
        return OrderedDict([
            ('Datum_vanaf', str),
            ('Ean', str),
            ('Kassabon', str),
            ('RPK_rep_id', str)
        ])
    return None


def get_columns_to_rename(filename: str) -> Optional[Dict[str, str]]:
    if filename.lower().startswith("kassabon"):
        return {
            'Datum_vanaf': 'start_date',
            'Ean': 'ean_number',
            'Kassabon': 'receipt_text',
            'RPK_rep_id': 'rep_id'
        }
    return None


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-directory",
                    help="The directory to read the csv files from")
parser.add_argument("-o", "--output-directory", default=None,
                    help="The directory to read the parquet files to")
parser.add_argument("-d", "--delimiter", default=";",
                    help="The delimiter used in the csv files")
parser.add_argument("-ec", "--encoding", default="utf-8",
                    help="The encoding of the csv files (default: utf-8)")
parser.add_argument("-e", "--extension", default=".csv",
                    help="The extension for the csv files")
parser.add_argument("-de", "--decimal", default=",",
                    help="The decimal separator used in the csv files")
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

    first_line = pd.read_csv(input_filename, sep=args.delimiter, nrows=1)

    # Add header names and types to all but kassa files
    header_types = get_column_types(filename)
    header_names = None if not header_types else [
        name for name in header_types.keys()]
    df = pd.read_csv(input_filename, sep=args.delimiter, engine="pyarrow",
                     encoding=args.encoding, decimal=args.decimal,
                     names=header_names, dtype=header_types, parse_dates=True)

    columns_to_rename = get_columns_to_rename(filename)
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)
    df.to_parquet(output_filename, engine="pyarrow")
